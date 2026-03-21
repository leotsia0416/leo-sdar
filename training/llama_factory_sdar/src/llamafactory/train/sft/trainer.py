# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        # Our text models expose `**kwargs` in forward for attention internals, but they do not
        # consume `num_items_in_batch`. Force the trainer to apply gradient-accumulation scaling
        # itself so the reported loss stays on the same scale as the component metrics.
        self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        self._loss_metric_sums: dict[str, torch.Tensor] = {}
        self._loss_metric_last_logged_step = 0
        self._puma_streaming_micro_step = 0

    @override
    def _wrap_model(self, model, training: bool = True, dataloader=None):
        model = super()._wrap_model(model, training=training, dataloader=dataloader)

        ddp_handler = getattr(self.accelerator, "ddp_handler", None)
        if training and ddp_handler is not None and getattr(model, "is_gradient_checkpointing", False):
            ddp_handler.static_graph = True
            logger.info_rank0("Enabled DDP static graph to support gradient checkpointing safely.")

        return model

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        self._configure_puma_streaming(model, inputs)
        result = super().compute_loss(model, inputs, *args, **kwargs)
        self._accumulate_loss_metrics(model)
        return result

    def _configure_puma_streaming(self, model: "torch.nn.Module", inputs: dict[str, Any]) -> None:
        raw_model = self.accelerator.unwrap_model(model, keep_torch_compile=False)
        config = getattr(raw_model, "config", None)
        if (
            config is None
            or not getattr(config, "gap_enable", False)
            or getattr(config, "gap_training_mode", None) not in {"puma", "remask"}
            or not getattr(config, "gap_puma_streaming", True)
            or not hasattr(raw_model, "set_puma_streaming_context")
        ):
            return

        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            return

        micro_batch_size = int(input_ids.size(0))
        base_batch_size = int(getattr(self.args, "per_device_train_batch_size", micro_batch_size) or micro_batch_size)
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        buffer_size = base_batch_size * grad_accum
        slot_offset = (self._puma_streaming_micro_step % grad_accum) * base_batch_size
        raw_model.set_puma_streaming_context(
            slot_offset=slot_offset,
            buffer_size=max(buffer_size, micro_batch_size),
        )
        self._puma_streaming_micro_step += 1

    def _accumulate_loss_metrics(self, model: "torch.nn.Module") -> None:
        raw_model = self.accelerator.unwrap_model(model, keep_torch_compile=False)
        metrics = getattr(raw_model, "_last_loss_metrics", None)
        if not metrics:
            return

        for name, value in metrics.items():
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.args.device)

            metric_value = value.detach().float().clone()
            if self.args.gradient_accumulation_steps > 1:
                metric_value = metric_value / self.args.gradient_accumulation_steps

            if name in self._loss_metric_sums:
                self._loss_metric_sums[name] += metric_value
            else:
                self._loss_metric_sums[name] = metric_value

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs and self._loss_metric_sums:
            steps_since_last_log = self.state.global_step - self._loss_metric_last_logged_step
            if steps_since_last_log > 0:
                for name, value in self._loss_metric_sums.items():
                    metric_scalar = self._nested_gather(value).mean().item()
                    logs[name] = round(metric_scalar / steps_since_last_log, 4)

            self._loss_metric_sums.clear()
            self._loss_metric_last_logged_step = self.state.global_step

        super().log(logs, start_time)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
