from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .block_runner import BaseBlockGenerator, BlockGenerationResult, GeneratedBlock, GenerationContext
from .interfaces import PolicyDecision
from .state_encoder import StateTensorEncoder, build_block_state_features
from .utils import SerializableMixin


@dataclass
class RuntimeBlockRecord(SerializableMixin):
    block_index: int
    absolute_block_index: int
    token_start: int
    token_end: int
    initial_block_token_ids: list[int]
    final_block_token_ids: list[int]
    initial_block_text: str | None
    final_block_text: str | None
    generated_tokens_in_block: int
    remasked: bool
    state_features: dict[str, float]
    policy_decision: PolicyDecision
    token_confidences: list[float] | None = None
    token_entropies: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyGuidedGenerationResult(SerializableMixin):
    prompt_id: str
    policy_mode: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    blocks: list[RuntimeBlockRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyGuidedGenerator:
    def __init__(
        self,
        base_generator: BaseBlockGenerator,
        *,
        policy_mode: str,
        policy_model: torch.nn.Module | None = None,
        state_tensor_encoder: StateTensorEncoder | None = None,
        policy_threshold: float = 0.5,
        heuristic_confidence_threshold: float = 0.5,
    ) -> None:
        self.base_generator = base_generator
        self.policy_mode = policy_mode
        self.policy_model = policy_model
        self.state_tensor_encoder = state_tensor_encoder
        self.policy_threshold = policy_threshold
        self.heuristic_confidence_threshold = heuristic_confidence_threshold

        if self.policy_mode == "learned":
            if self.policy_model is None or self.state_tensor_encoder is None:
                raise ValueError("learned mode requires policy_model and state_tensor_encoder.")

    def generate(self, context: GenerationContext) -> PolicyGuidedGenerationResult:
        self.base_generator.load_components()
        initial_prompt_token_ids = self.base_generator.encode_prompt_token_ids(context)
        current_prefix_token_ids = list(initial_prompt_token_ids)
        total_decode_blocks = self._total_decode_blocks(
            prompt_length=len(initial_prompt_token_ids),
            gen_length=context.gen_length,
            block_length=context.block_length,
        )
        stop_ids = self.base_generator.resolve_stop_ids(context)
        prompt_prefill_blocks = len(initial_prompt_token_ids) // context.block_length
        runtime_blocks: list[RuntimeBlockRecord] = []

        generated_tokens = 0
        while generated_tokens < context.gen_length:
            block_budget = self._next_block_budget(
                prefix_length=len(current_prefix_token_ids),
                remaining_gen_length=context.gen_length - generated_tokens,
                block_length=context.block_length,
            )
            if block_budget <= 0:
                break

            block_context = replace(
                context,
                prompt_text=None,
                prompt_messages=None,
                prompt_token_ids=list(current_prefix_token_ids),
                gen_length=block_budget,
            )
            initial_result = self.base_generator.generate_from_token_ids(
                current_prefix_token_ids,
                block_context,
            )
            initial_block = self._select_current_block(initial_result)
            self.base_generator.annotate_block_scores(initial_result, block_index=initial_block.block_index)
            initial_block = initial_result.get_block(initial_block.block_index)

            relative_block_index = initial_block.absolute_block_index - prompt_prefill_blocks
            state_features = build_block_state_features(
                block_index=relative_block_index,
                total_decode_blocks=total_decode_blocks,
                token_start=initial_block.token_start,
                token_end=initial_block.token_end,
                token_ids=initial_block.token_ids,
                prompt_tokens_in_block=initial_block.prompt_tokens_in_block,
                generated_tokens_in_block=initial_block.generated_tokens_in_block,
                prompt_length=len(initial_prompt_token_ids),
                requested_gen_length=context.gen_length,
                output_token_ids=initial_result.output_token_ids[:initial_result.effective_output_length],
                stopping_criteria_idx=stop_ids,
                token_confidences=initial_block.token_confidences,
                token_entropies=initial_block.token_entropies,
                verifier_score=float(context.metadata.get("verifier_score", 0.0)),
            )
            decision = self._decide(state_features, initial_block)

            final_result = initial_result
            final_block = initial_block
            remasked = False
            if decision.should_remask:
                remasked_result = self.base_generator.generate_from_token_ids(
                    current_prefix_token_ids,
                    block_context,
                )
                remasked_block = self._select_current_block(remasked_result)
                self.base_generator.annotate_block_scores(remasked_result, block_index=remasked_block.block_index)
                final_result = remasked_result
                final_block = remasked_result.get_block(remasked_block.block_index)
                remasked = True

            accepted_token_ids = final_result.output_token_ids[
                len(current_prefix_token_ids):final_result.effective_output_length
            ]
            if not accepted_token_ids:
                break
            current_prefix_token_ids.extend(accepted_token_ids)
            generated_tokens += len(accepted_token_ids)

            runtime_blocks.append(
                RuntimeBlockRecord(
                    block_index=relative_block_index,
                    absolute_block_index=final_block.absolute_block_index,
                    token_start=final_block.token_start,
                    token_end=final_block.token_end,
                    initial_block_token_ids=list(initial_block.token_ids),
                    final_block_token_ids=list(final_block.token_ids),
                    initial_block_text=initial_block.decoded_text,
                    final_block_text=final_block.decoded_text,
                    generated_tokens_in_block=final_block.generated_tokens_in_block,
                    remasked=remasked,
                    state_features=state_features,
                    policy_decision=decision,
                    token_confidences=final_block.token_confidences,
                    token_entropies=final_block.token_entropies,
                    metadata={
                        "block_budget": block_budget,
                        "stop_token_seen_in_block": bool(
                            stop_ids and any(token in stop_ids for token in accepted_token_ids)
                        ),
                    },
                )
            )

            if stop_ids and any(token in stop_ids for token in accepted_token_ids):
                break

        generated_token_ids = current_prefix_token_ids[len(initial_prompt_token_ids):]
        generated_text = self.base_generator.decode_token_ids(
            generated_token_ids,
            skip_special_tokens=False,
            clean_mask_tokens=True,
        )
        return PolicyGuidedGenerationResult(
            prompt_id=context.prompt_id,
            policy_mode=self.policy_mode,
            prompt_token_ids=initial_prompt_token_ids,
            output_token_ids=current_prefix_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            blocks=runtime_blocks,
            metadata={
                **context.metadata,
                "requested_gen_length": context.gen_length,
                "heuristic_confidence_threshold": self.heuristic_confidence_threshold,
                "policy_threshold": self.policy_threshold,
                "remasked_blocks": sum(1 for block in runtime_blocks if block.remasked),
            },
        )

    def _decide(
        self,
        state_features: dict[str, float],
        block: GeneratedBlock,
    ) -> PolicyDecision:
        if self.policy_mode == "off":
            return PolicyDecision(
                should_remask=False,
                score=0.0,
                threshold=1.0,
                block_index=block.block_index,
                reason="policy_mode_off",
            )

        if self.policy_mode == "heuristic":
            if not block.token_confidences:
                return PolicyDecision(
                    should_remask=False,
                    score=0.0,
                    threshold=self.heuristic_confidence_threshold,
                    block_index=block.block_index,
                    reason="confidence_unavailable",
                )
            mean_confidence = float(sum(block.token_confidences) / len(block.token_confidences))
            should_remask = mean_confidence < self.heuristic_confidence_threshold
            return PolicyDecision(
                should_remask=should_remask,
                score=mean_confidence,
                threshold=self.heuristic_confidence_threshold,
                block_index=block.block_index,
                reason="mean_confidence_below_threshold" if should_remask else "mean_confidence_above_threshold",
            )

        if self.policy_mode == "learned":
            assert self.policy_model is not None
            assert self.state_tensor_encoder is not None
            with torch.no_grad():
                logits = self.policy_model(self.state_tensor_encoder.encode(state_features).unsqueeze(0))
                probability = float(torch.sigmoid(logits).squeeze(0).item())
            should_remask = probability >= self.policy_threshold
            return PolicyDecision(
                should_remask=should_remask,
                score=probability,
                threshold=self.policy_threshold,
                block_index=block.block_index,
                reason="policy_probability_above_threshold" if should_remask else "policy_probability_below_threshold",
            )

        raise ValueError(f"Unsupported policy mode: {self.policy_mode}")

    @staticmethod
    def _select_current_block(result: BlockGenerationResult) -> GeneratedBlock:
        generated_blocks = [
            block for block in result.blocks if block.generated_tokens_in_block > 0
        ]
        if not generated_blocks:
            raise ValueError("No generated block found in block generation result.")
        return generated_blocks[-1]

    @staticmethod
    def _next_block_budget(
        *,
        prefix_length: int,
        remaining_gen_length: int,
        block_length: int,
    ) -> int:
        remainder = prefix_length % block_length
        block_budget = block_length if remainder == 0 else block_length - remainder
        return min(block_budget, remaining_gen_length)

    @staticmethod
    def _total_decode_blocks(*, prompt_length: int, gen_length: int, block_length: int) -> int:
        rounded_total_length = ((prompt_length + gen_length + block_length - 1) // block_length) * block_length
        decode_block_start = (prompt_length // block_length) * block_length
        return max(1, len(range(decode_block_start, rounded_total_length, block_length)))
