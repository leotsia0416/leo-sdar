from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import torch

from .config import BaseRemaskConfig
from .utils import SerializableMixin

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenerationContext(SerializableMixin):
    prompt_id: str
    prompt_text: str | None = None
    prompt_messages: list[dict[str, Any]] | None = None
    prompt_token_ids: list[int] | None = None
    reference_text: str | None = None
    prompt_length: int = 4096
    gen_length: int = 128
    block_length: int = 4
    denoising_steps: int = 4
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    remasking_strategy: str = "low_confidence_dynamic"
    confidence_threshold: float = 0.85
    eb_threshold: float | None = 0.35
    stopping_criteria_idx: list[int] | None = None
    mask_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: BaseRemaskConfig,
        *,
        prompt_id: str,
        prompt_text: str | None = None,
        prompt_messages: list[dict[str, Any]] | None = None,
        prompt_token_ids: list[int] | None = None,
        reference_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GenerationContext":
        return cls(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_messages=prompt_messages,
            prompt_token_ids=prompt_token_ids,
            reference_text=reference_text,
            prompt_length=config.prompt_length,
            gen_length=config.gen_length,
            block_length=config.block_length,
            denoising_steps=config.denoising_steps,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            remasking_strategy=config.remasking_strategy,
            confidence_threshold=config.confidence_threshold,
            eb_threshold=config.eb_threshold,
            stopping_criteria_idx=config.stopping_criteria_idx,
            metadata=dict(metadata or {}),
        )


@dataclass
class GeneratedBlock(SerializableMixin):
    block_index: int
    absolute_block_index: int
    token_start: int
    token_end: int
    token_ids: list[int]
    decoded_text: str | None = None
    prompt_tokens_in_block: int = 0
    generated_tokens_in_block: int = 0
    token_confidences: list[float] | None = None
    token_entropies: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockGenerationResult(SerializableMixin):
    prompt_id: str
    prompt_length: int
    requested_gen_length: int
    rounded_total_length: int
    effective_output_length: int
    decode_block_start: int
    block_length: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str | None = None
    blocks: list[GeneratedBlock] = field(default_factory=list)
    stopping_criteria_idx: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_block(self, block_index: int) -> GeneratedBlock:
        for block in self.blocks:
            if block.block_index == block_index:
                return block
        raise IndexError(f"Unknown block index: {block_index}")


class BaseBlockGenerator:
    """Thin wrapper around the repo's built-in block diffusion generator."""

    def __init__(
        self,
        config: BaseRemaskConfig,
        *,
        model: "PreTrainedModel | Any | None" = None,
        tokenizer: "PreTrainedTokenizerBase | Any | None" = None,
        generate_fn: Any | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self._generate_fn = generate_fn

    def load_components(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self._resolve_torch_dtype(self.config.dtype),
            device_map=self.config.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_dir,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._patch_model_runtime()

    def generate(self, context: GenerationContext) -> BlockGenerationResult:
        self.load_components()
        prompt = self._encode_prompt(context)
        output = self._run_generation(prompt=prompt, context=context)
        return self._build_result(prompt["input_ids"], output, context)

    def generate_from_token_ids(
        self,
        prompt_token_ids: list[int],
        context: GenerationContext,
    ) -> BlockGenerationResult:
        branch_context = replace(
            context,
            prompt_text=None,
            prompt_messages=None,
            prompt_token_ids=list(prompt_token_ids),
        )
        return self.generate(branch_context)

    def encode_prompt_token_ids(self, context: GenerationContext) -> list[int]:
        self.load_components()
        prompt = self._encode_prompt(context)
        return prompt["input_ids"][0].detach().cpu().tolist()

    def resolve_stop_ids(self, context: GenerationContext) -> list[int] | None:
        return self._resolve_stop_ids(context)

    @torch.no_grad()
    def annotate_block_scores(
        self,
        result: BlockGenerationResult,
        *,
        block_index: int | None = None,
    ) -> BlockGenerationResult:
        self.load_components()
        blocks = (
            [result.get_block(block_index)]
            if block_index is not None
            else [block for block in result.blocks if block.generated_tokens_in_block > 0]
        )
        for block in blocks:
            if block.generated_tokens_in_block <= 0:
                continue
            token_confidences, token_entropies = self._score_block_tokens(result, block)
            block.token_confidences = token_confidences
            block.token_entropies = token_entropies
            block.metadata["confidence_todo"] = False
            block.metadata["entropy_todo"] = False
        if blocks:
            result.metadata["confidence_todo"] = False
            result.metadata["entropy_todo"] = False
        return result

    def decode_token_ids(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_mask_tokens: bool = False,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if clean_mask_tokens:
            mask_token = getattr(self.tokenizer, "mask_token", None)
            if mask_token:
                text = text.replace(mask_token, "")
        return text

    def _run_generation(self, prompt: dict[str, torch.Tensor], context: GenerationContext) -> torch.Tensor:
        generate_fn = self._resolve_generate_fn()
        return generate_fn(
            self.model,
            prompt=prompt,
            mask_id=self._resolve_mask_id(context),
            gen_length=context.gen_length,
            block_length=context.block_length,
            denoising_steps=context.denoising_steps,
            temperature=context.temperature,
            top_k=context.top_k,
            top_p=context.top_p,
            remasking_strategy=context.remasking_strategy,
            confidence_threshold=context.confidence_threshold,
            eb_threshold=context.eb_threshold,
            stopping_criteria_idx=self._resolve_stop_ids(context),
        )

    def _patch_model_runtime(self) -> None:
        if self.model is None:
            return

        module_name = getattr(self.model.__class__, "__module__", "")
        if not module_name:
            return

        module = importlib.import_module(module_name)
        flex_attention = getattr(module, "flex_attention", None)
        create_block_mask = getattr(module, "create_block_mask", None)
        fused_flex_attention = getattr(module, "fused_flex_attention", None)
        if flex_attention is None or fused_flex_attention is None:
            return
        if getattr(fused_flex_attention, "_remask_eager_patch", False):
            return

        def eager_fused_flex_attention(query, key, value, attention_mask, **kwargs):
            if isinstance(attention_mask, torch.Tensor):
                q_len = query.shape[-2]
                kv_len = key.shape[-2]
                attn_mask = attention_mask.to(dtype=torch.bool, device=query.device)
                if attn_mask.dim() == 4:
                    attn_mask = attn_mask[..., -q_len:, -kv_len:]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, -q_len:, -kv_len:]
                elif attn_mask.dim() == 2:
                    attn_mask = attn_mask[-q_len:, -kv_len:]
                    attn_mask = attn_mask.unsqueeze(0)

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=kwargs.get("scale"),
                    enable_gqa=kwargs.get("enable_gqa", False),
                )
                return attn_output, None

            block_mask = attention_mask
            q_len = query.shape[-2]
            kv_len = key.shape[-2]
            adjust = getattr(block_mask, "_adjust", None)
            if adjust is not None:
                block_mask = adjust(q_len, kv_len)
            return flex_attention(query, key, value, block_mask=block_mask, **kwargs)

        eager_fused_flex_attention._remask_eager_patch = True  # type: ignore[attr-defined]
        module.fused_flex_attention = eager_fused_flex_attention

    def _encode_prompt(self, context: GenerationContext) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        if context.prompt_token_ids is not None:
            input_ids = torch.tensor([context.prompt_token_ids], dtype=torch.long, device=self.model.device)
            return {"input_ids": input_ids}

        if context.prompt_messages is not None:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not support apply_chat_template for message prompts.")
            prompt_text = self.tokenizer.apply_chat_template(
                context.prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            tokenized = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=False,
                max_length=context.prompt_length,
            )
        elif context.prompt_text is not None:
            tokenized = self.tokenizer(
                context.prompt_text,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=True,
                max_length=context.prompt_length,
            )
        else:
            raise ValueError("One of prompt_text, prompt_messages, or prompt_token_ids must be set.")

        return {key: value.to(self.model.device) for key, value in tokenized.items()}

    def _build_result(
        self,
        prompt_input_ids: torch.Tensor,
        output: torch.Tensor,
        context: GenerationContext,
    ) -> BlockGenerationResult:
        prompt_token_ids = prompt_input_ids[0].detach().cpu().tolist()
        output_token_ids = output[0].detach().cpu().tolist()
        prompt_length = len(prompt_token_ids)
        rounded_total_length = len(output_token_ids)
        effective_output_length = min(rounded_total_length, prompt_length + context.gen_length)
        decode_block_start = (prompt_length // context.block_length) * context.block_length
        generated_token_ids = output_token_ids[prompt_length:effective_output_length]

        blocks: list[GeneratedBlock] = []
        for block_index, token_start in enumerate(
            range(decode_block_start, rounded_total_length, context.block_length)
        ):
            token_end = min(token_start + context.block_length, rounded_total_length)
            token_ids = output_token_ids[token_start:token_end]
            prompt_tokens_in_block = max(0, min(prompt_length, token_end) - token_start)
            generated_tokens_in_block = max(
                0,
                min(effective_output_length, token_end) - max(prompt_length, token_start),
            )
            block_text = self.decode_token_ids(token_ids, clean_mask_tokens=False)
            blocks.append(
                GeneratedBlock(
                    block_index=block_index,
                    absolute_block_index=token_start // context.block_length,
                    token_start=token_start,
                    token_end=token_end,
                    token_ids=token_ids,
                    decoded_text=block_text,
                    prompt_tokens_in_block=prompt_tokens_in_block,
                    generated_tokens_in_block=generated_tokens_in_block,
                    metadata={
                        "token_end_is_exclusive": True,
                        "confidence_todo": True,
                        "entropy_todo": True,
                    },
                )
            )

        generated_text = self.decode_token_ids(
            generated_token_ids,
            skip_special_tokens=False,
            clean_mask_tokens=True,
        )

        return BlockGenerationResult(
            prompt_id=context.prompt_id,
            prompt_length=prompt_length,
            requested_gen_length=context.gen_length,
            rounded_total_length=rounded_total_length,
            effective_output_length=effective_output_length,
            decode_block_start=decode_block_start,
            block_length=context.block_length,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            blocks=blocks,
            stopping_criteria_idx=self._resolve_stop_ids(context),
            metadata={
                **context.metadata,
                "generation_backend": "generate.block_diffusion_generate",
                "confidence_todo": True,
                "entropy_todo": True,
            },
        )

    def _resolve_mask_id(self, context: GenerationContext) -> int:
        if context.mask_id is not None:
            return context.mask_id
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is not None:
            return int(mask_token_id)

        mask_token = getattr(self.tokenizer, "mask_token", None)
        if mask_token is None:
            raise ValueError("Tokenizer does not expose a mask token.")
        return int(self.tokenizer(mask_token)["input_ids"][0])

    def _resolve_stop_ids(self, context: GenerationContext) -> list[int] | None:
        if context.stopping_criteria_idx is not None:
            return list(context.stopping_criteria_idx)
        if self.config.stopping_criteria_idx is not None:
            return list(self.config.stopping_criteria_idx)

        try:
            from transformers import GenerationConfig

            generation_config = GenerationConfig.from_pretrained(self.config.model_dir)
        except Exception:
            return None

        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        return list(eos_token_id)

    def _resolve_generate_fn(self) -> Any:
        if self._generate_fn is None:
            module = importlib.import_module("generate")
            self._generate_fn = getattr(module, "block_diffusion_generate")
        return self._generate_fn

    def _score_block_tokens(
        self,
        result: BlockGenerationResult,
        block: GeneratedBlock,
    ) -> tuple[list[float], list[float]]:
        from transformers.cache_utils import DynamicCache

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        model_device = self._model_device()
        block_length = result.block_length
        sequence_upto_block = result.output_token_ids[:block.token_end]
        total_length = len(sequence_upto_block)
        num_blocks = total_length // block_length
        x = torch.tensor([sequence_upto_block], dtype=torch.long, device=model_device)
        past_key_values = DynamicCache()

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model_device))
        attention_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(
            block_length, dim=1
        ).unsqueeze(0)
        position_ids = torch.arange(total_length, device=model_device).unsqueeze(0)

        prefill_length = block.token_start
        if prefill_length > 0:
            self.model(
                x[:, :prefill_length],
                attention_mask=attention_mask[:, :prefill_length, :prefill_length],
                position_ids=position_ids[:, :prefill_length],
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

        cur_x = x[:, block.token_start:block.token_end]
        logits = self.model(
            cur_x,
            attention_mask=attention_mask[:, block.token_start:block.token_end, :block.token_end],
            position_ids=position_ids[:, block.token_start:block.token_end],
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=False,
        ).logits[0]

        probabilities = torch.softmax(logits, dim=-1)
        block_token_ids = torch.tensor(block.token_ids, dtype=torch.long, device=model_device).unsqueeze(-1)
        selected_probabilities = torch.gather(probabilities, -1, block_token_ids).squeeze(-1)
        entropies = -(
            probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)

        generated_start = block.prompt_tokens_in_block
        generated_end = generated_start + block.generated_tokens_in_block
        return (
            selected_probabilities[generated_start:generated_end].detach().cpu().tolist(),
            entropies[generated_start:generated_end].detach().cpu().tolist(),
        )

    def _model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        resolved = getattr(torch, dtype_name)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported torch dtype value: {dtype_name}")
        return resolved
