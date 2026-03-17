# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


GSM8K_ONLINE_EVAL_PROMPT = (
    "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)


def _extract_gsm8k_gold(text: str) -> str:
    return text.split("#### ")[1].replace(",", "").strip()


def _extract_gsm8k_pred(text: str) -> str:
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        boxed = boxed_matches[-1].replace(",", "").strip()
        if boxed:
            return boxed

    numbers = re.findall(r"-?\d+\.\d+|-?\d+", text)
    return numbers[-1] if numbers else "NULL"


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def _get_num_transfer_tokens(block_length: int, steps: int) -> torch.LongTensor:
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def _block_diffusion_gap_generate(
    model: torch.nn.Module,
    prompt,
    mask_id: int,
    gen_length: int = 128,
    block_length: int = 8,
    denoising_steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 1.0,
    remask_threshold: float = 0.5,
    stopping_criteria_idx: Optional[list[int]] = None,
) -> torch.Tensor:
    from transformers.cache_utils import DynamicCache

    input_ids = prompt["input_ids"]
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    num_transfer_tokens = _get_num_transfer_tokens(block_length, denoising_steps)
    gap_enabled = bool(getattr(model.config, "gap_enable", False)) and hasattr(model, "gap_remask_head")

    for num_block in range(prefill_blocks, num_blocks):
        block_slice = slice(num_block * block_length, (num_block + 1) * block_length)
        cur_x = x[:, block_slice].clone()
        cur_attn_mask = block_diffusion_attention_mask[:, block_slice, : (num_block + 1) * block_length]
        cur_position_ids = position_ids[:, block_slice]

        for step in range(denoising_steps + 1):
            mask_index = cur_x.eq(mask_id)
            if mask_index.sum() == 0:
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break

            outputs = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
                output_hidden_states=gap_enabled,
            )
            logits = outputs.logits
            logits[..., mask_id] = float("-inf")

            x0, x0_p = _sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            x0 = torch.where(mask_index, x0, cur_x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row_idx in range(batch_size):
                row_mask = mask_index[row_idx]
                masked_count = int(row_mask.sum().item())
                if masked_count == 0:
                    continue

                row_scores = torch.where(row_mask, x0_p[row_idx], torch.full_like(x0_p[row_idx], float("-inf")))
                k = min(int(num_transfer_tokens[step].item()), masked_count)
                if k <= 0:
                    continue

                if remasking_strategy == "sequential":
                    masked_positions = row_mask.nonzero(as_tuple=True)[0]
                    transfer_index[row_idx, masked_positions[:k]] = True
                elif remasking_strategy == "low_confidence_static":
                    _, idx = torch.topk(row_scores, k=k)
                    transfer_index[row_idx, idx] = True
                elif remasking_strategy == "low_confidence_dynamic":
                    high_conf_mask = row_scores > confidence_threshold
                    if int(high_conf_mask.sum().item()) >= k:
                        transfer_index[row_idx] = high_conf_mask
                    else:
                        _, idx = torch.topk(row_scores, k=k)
                        transfer_index[row_idx, idx] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            if gap_enabled:
                hidden_states = outputs.hidden_states[-1]
                remask_probs = torch.sigmoid(model.gap_remask_head(hidden_states).squeeze(-1))
                remask_index = transfer_index & remask_probs.ge(remask_threshold)
                transfer_index = transfer_index & ~remask_index

                for row_idx in range(batch_size):
                    if transfer_index[row_idx].any() or not mask_index[row_idx].any():
                        continue

                    candidates = (mask_index[row_idx] & ~remask_index[row_idx]).nonzero(as_tuple=True)[0]
                    if candidates.numel() > 0:
                        candidate_scores = x0_p[row_idx, candidates]
                        best_idx = candidates[torch.argmax(candidate_scores)]
                    else:
                        candidates = mask_index[row_idx].nonzero(as_tuple=True)[0]
                        remask_scores = remask_probs[row_idx, candidates]
                        best_idx = candidates[torch.argmin(remask_scores)]
                    transfer_index[row_idx, best_idx] = True

            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_slice] = cur_x

        if stopping_criteria_idx is not None:
            generated_prefix = x[:, prompt_length : (num_block + 1) * block_length]
            if any(stop_idx in generated_prefix for stop_idx in stopping_criteria_idx):
                break

    return x


class OnlineGsm8kEvalCallback(TrainerCallback):
    def __init__(self) -> None:
        self.eval_steps = int(os.getenv("SDAR_ONLINE_EVAL_STEPS", "500"))
        self.num_samples = int(os.getenv("SDAR_ONLINE_EVAL_SAMPLES", "100"))
        self.dataset_path = os.getenv("SDAR_ONLINE_EVAL_DATA", "/work/leotsia0416/datasets/gsm8k/test.jsonl")
        self.max_new_tokens = int(os.getenv("SDAR_ONLINE_EVAL_MAX_NEW_TOKENS", "1024"))
        self.block_length = os.getenv("SDAR_ONLINE_EVAL_BLOCK_LENGTH")
        self.confidence_threshold = os.getenv("SDAR_ONLINE_EVAL_CONFIDENCE_THRESHOLD")
        self.remasking_strategy = os.getenv("SDAR_ONLINE_EVAL_REMASKING_STRATEGY", "low_confidence_dynamic")
        self.remask_threshold = os.getenv("SDAR_ONLINE_EVAL_REMASK_THRESHOLD")
        self._last_eval_step = -1
        self._examples: Optional[list[dict[str, str]]] = None

    def _load_examples(self) -> list[dict[str, str]]:
        if self._examples is None:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                examples = [json.loads(line) for line in f]
            self._examples = examples[: self.num_samples]
        return self._examples

    @torch.no_grad()
    def _run_eval(self, model: torch.nn.Module, tokenizer, output_dir: str, global_step: int) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        was_training = raw_model.training
        raw_model.eval()
        block_length = (
            int(self.block_length) if self.block_length is not None else int(getattr(raw_model.config, "block_size", 4))
        )
        confidence_threshold = (
            float(self.confidence_threshold)
            if self.confidence_threshold is not None
            else float(getattr(raw_model.config, "gap_rollout_confidence_threshold", 0.95))
        )
        remask_threshold = (
            float(self.remask_threshold)
            if self.remask_threshold is not None
            else float(getattr(raw_model.config, "gap_remask_threshold", 0.5))
        )

        examples = self._load_examples()
        sample_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "gsm8k_online_eval_samples.jsonl")

        correct = 0
        records = []
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

        for idx, example in enumerate(examples):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": GSM8K_ONLINE_EVAL_PROMPT.format(question=example["question"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = tokenizer(prompt_text, return_tensors="pt").to(raw_model.device)
            generated = _block_diffusion_gap_generate(
                model=raw_model,
                prompt=prompt,
                mask_id=raw_model.config.mask_token_id,
                gen_length=self.max_new_tokens,
                block_length=block_length,
                denoising_steps=block_length,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                remasking_strategy=self.remasking_strategy,
                confidence_threshold=confidence_threshold,
                remask_threshold=remask_threshold,
                stopping_criteria_idx=eos_ids,
            )
            output_ids = generated[:, prompt["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred = _extract_gsm8k_pred(pred_text)
            gold = _extract_gsm8k_gold(example["answer"])
            is_correct = pred == gold or (
                pred not in {"NULL", ""} and abs(float(pred) - int(gold)) < 1e-6
            )
            correct += int(is_correct)
            records.append(
                {
                    "index": idx,
                    "question": example["question"],
                    "prediction_text": pred_text,
                    "prediction": pred,
                    "answer": gold,
                    "correct": is_correct,
                }
            )

        with open(sample_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        accuracy = 100.0 * correct / len(examples)
        logger.info_rank0(
            f"online_gsm8k step={global_step} samples={len(examples)} accuracy={accuracy:.2f}"
        )

        if was_training:
            raw_model.train()

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save or self.eval_steps <= 0:
            return
        if state.global_step == 0 or state.global_step == self._last_eval_step:
            return
        if state.global_step % self.eval_steps != 0:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            logger.warning_rank0("Skip online_gsm8k eval because model/tokenizer is unavailable.")
            return

        self._last_eval_step = state.global_step
        self._run_eval(model=model, tokenizer=tokenizer, output_dir=args.output_dir, global_step=state.global_step)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )
