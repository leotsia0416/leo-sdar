from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .block_runner import BaseBlockGenerator, BlockGenerationResult, GeneratedBlock, GenerationContext
from .interfaces import BlockDecisionRecord, RolloutRecord
from .reward import BaseRewardAdapter, DummyRewardAdapter
from .state_encoder import (
    DEFAULT_BLOCK_FEATURE_DESCRIPTIONS,
    DEFAULT_BLOCK_FEATURE_NAMES,
    build_block_state_features,
)
from .utils import SerializableMixin, ensure_dir

logger = logging.getLogger("remask_policy")


@dataclass
class PromptExample(SerializableMixin):
    prompt_id: str
    prompt_text: str | None = None
    prompt_messages: list[dict[str, Any]] | None = None
    reference_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBundle(SerializableMixin):
    base_rollout: RolloutRecord
    counterfactual_rollouts: list[RolloutRecord] = field(default_factory=list)


@dataclass
class RolloutExportResult(SerializableMixin):
    supervision_dataset_path: str
    base_rollout_path: str
    branch_rollout_path: str
    feature_schema_path: str
    num_supervision_samples: int
    num_base_rollouts: int
    num_branch_rollouts: int


class DefaultStateEncoder:
    def encode(
        self,
        generation_result: BlockGenerationResult,
        block: GeneratedBlock,
    ) -> dict[str, float]:
        return build_block_state_features(
            block_index=block.block_index,
            total_decode_blocks=len(generation_result.blocks),
            token_start=block.token_start,
            token_end=block.token_end,
            token_ids=block.token_ids,
            prompt_tokens_in_block=block.prompt_tokens_in_block,
            generated_tokens_in_block=block.generated_tokens_in_block,
            prompt_length=generation_result.prompt_length,
            requested_gen_length=generation_result.requested_gen_length,
            output_token_ids=generation_result.output_token_ids[:generation_result.effective_output_length],
            stopping_criteria_idx=generation_result.stopping_criteria_idx,
            token_confidences=block.token_confidences,
            token_entropies=block.token_entropies,
            verifier_score=float(generation_result.metadata.get("verifier_score", 0.0)),
        )


class RolloutCollector:
    def __init__(
        self,
        generator: BaseBlockGenerator,
        reward_adapter: BaseRewardAdapter | None = None,
        *,
        num_counterfactual_blocks: int | None = None,
        remask_penalty_lambda: float | None = None,
        random_seed: int | None = None,
        state_encoder: DefaultStateEncoder | None = None,
    ) -> None:
        self.generator = generator
        self.reward_adapter = reward_adapter or DummyRewardAdapter()
        self.num_counterfactual_blocks = (
            num_counterfactual_blocks
            if num_counterfactual_blocks is not None
            else generator.config.num_counterfactual_blocks
        )
        self.remask_penalty_lambda = (
            remask_penalty_lambda
            if remask_penalty_lambda is not None
            else generator.config.remask_penalty_lambda
        )
        self.random = random.Random(
            generator.config.random_seed if random_seed is None else random_seed
        )
        self.state_encoder = state_encoder or DefaultStateEncoder()

    def collect_example(self, example: PromptExample) -> RolloutBundle:
        logger.info("Collecting base rollout for prompt_id=%s", example.prompt_id)
        context = GenerationContext.from_config(
            self.generator.config,
            prompt_id=example.prompt_id,
            prompt_text=example.prompt_text,
            prompt_messages=example.prompt_messages,
            reference_text=example.reference_text,
            metadata=example.metadata,
        )

        base_result = self.generator.generate(context)
        self.generator.annotate_block_scores(base_result)
        base_rollout = self._build_rollout_record(
            example=example,
            context=context,
            generation_result=base_result,
            policy_mode=self.generator.config.policy_mode,
        )
        base_reward = self.reward_adapter.evaluate(base_rollout)
        base_rollout.base_reward = base_reward
        base_rollout.final_reward = base_reward
        for record in base_rollout.decisions:
            record.base_reward = base_reward

        sampled_blocks = self._sample_counterfactual_blocks(base_result)
        counterfactual_rollouts: list[RolloutRecord] = []
        if sampled_blocks:
            logger.info(
                "Prompt %s sampled %d counterfactual block(s): %s",
                example.prompt_id,
                len(sampled_blocks),
                [block.block_index for block in sampled_blocks],
            )
        for block in sampled_blocks:
            logger.info(
                "Collecting counterfactual rollout for prompt_id=%s block_index=%d",
                example.prompt_id,
                block.block_index,
            )
            branch_rollout = self.collect_counterfactual_rollout(
                example=example,
                base_result=base_result,
                base_rollout=base_rollout,
                target_block=block,
            )
            counterfactual_rollouts.append(branch_rollout)
        logger.info(
            "Finished prompt_id=%s with %d counterfactual rollout(s)",
            example.prompt_id,
            len(counterfactual_rollouts),
        )

        return RolloutBundle(
            base_rollout=base_rollout,
            counterfactual_rollouts=counterfactual_rollouts,
        )

    def collect_many(self, examples: Iterable[PromptExample]) -> list[RolloutBundle]:
        examples = list(examples)
        total = len(examples)
        bundles: list[RolloutBundle] = []
        for index, example in enumerate(examples, start=1):
            logger.info("Progress %d/%d prompt_id=%s", index, total, example.prompt_id)
            bundles.append(self.collect_example(example))
        logger.info("Completed rollout collection for %d prompt(s)", total)
        return bundles

    def collect_counterfactual_rollout(
        self,
        *,
        example: PromptExample,
        base_result: BlockGenerationResult,
        base_rollout: RolloutRecord,
        target_block: GeneratedBlock,
    ) -> RolloutRecord:
        prefix_length = max(base_result.prompt_length, target_block.token_start)
        prefix_token_ids = base_result.output_token_ids[:prefix_length]
        remaining_length = base_result.rounded_total_length - prefix_length
        if remaining_length <= 0:
            raise ValueError("Counterfactual rollout requires positive remaining generation length.")

        branch_context = GenerationContext.from_config(
            self.generator.config,
            prompt_id=f"{example.prompt_id}::cf::{target_block.block_index}",
            prompt_token_ids=prefix_token_ids,
            reference_text=example.reference_text,
            metadata={
                **example.metadata,
                "single_intervention": True,
                "intervention_block_index": target_block.block_index,
                "fixed_prefix_length": prefix_length,
            },
        )
        branch_context.gen_length = remaining_length
        branch_result = self.generator.generate(branch_context)
        self.generator.annotate_block_scores(branch_result)
        branch_rollout = self._build_rollout_record(
            example=example,
            context=branch_context,
            generation_result=branch_result,
            policy_mode="counterfactual_single_remask",
            parent_rollout_id=base_rollout.rollout_id,
            intervention_block_index=target_block.block_index,
        )
        branch_reward = self.reward_adapter.evaluate(branch_rollout)
        branch_rollout.base_reward = base_rollout.base_reward
        branch_rollout.final_reward = branch_reward

        delta = branch_reward.reward - float(base_rollout.base_reward.reward if base_rollout.base_reward else 0.0)
        decision_record = base_rollout.decisions[target_block.block_index]
        decision_record.branch_reward = branch_reward
        decision_record.reward_delta = delta
        decision_record.label = int(delta > self.remask_penalty_lambda)
        decision_record.metadata.update(
            {
                "counterfactual_rollout_id": branch_rollout.rollout_id,
                "single_intervention": True,
                "fixed_prefix_length": prefix_length,
            }
        )
        return branch_rollout

    def _sample_counterfactual_blocks(self, generation_result: BlockGenerationResult) -> list[GeneratedBlock]:
        eligible_blocks = [
            block
            for block in generation_result.blocks
            if block.generated_tokens_in_block > 0
        ]
        if not eligible_blocks:
            return []

        sample_count = min(self.num_counterfactual_blocks, len(eligible_blocks))
        if sample_count == len(eligible_blocks):
            return list(eligible_blocks)
        return sorted(
            self.random.sample(eligible_blocks, sample_count),
            key=lambda block: block.block_index,
        )

    def _build_rollout_record(
        self,
        *,
        example: PromptExample,
        context: GenerationContext,
        generation_result: BlockGenerationResult,
        policy_mode: str,
        parent_rollout_id: str | None = None,
        intervention_block_index: int | None = None,
    ) -> RolloutRecord:
        rollout_id = self._make_rollout_id(example.prompt_id, policy_mode)
        decisions = [
            BlockDecisionRecord(
                prompt_id=example.prompt_id,
                rollout_id=rollout_id,
                block_index=block.block_index,
                block_token_start=block.token_start,
                block_token_end=block.token_end,
                state_features=self.state_encoder.encode(generation_result, block),
                remask_penalty_lambda=self.remask_penalty_lambda,
                metadata={
                    "absolute_block_index": block.absolute_block_index,
                    "prompt_tokens_in_block": block.prompt_tokens_in_block,
                    "generated_tokens_in_block": block.generated_tokens_in_block,
                    "token_end_is_exclusive": True,
                    "confidence_todo": block.token_confidences is None,
                    "entropy_todo": block.token_entropies is None,
                },
            )
            for block in generation_result.blocks
        ]

        return RolloutRecord(
            rollout_id=rollout_id,
            prompt_id=example.prompt_id,
            prompt_text=example.prompt_text or "",
            reference_text=example.reference_text,
            generated_text=generation_result.generated_text,
            model_dir=self.generator.config.model_dir,
            trust_remote_code=self.generator.config.trust_remote_code,
            device=self.generator.config.device,
            dtype=self.generator.config.dtype,
            block_length=self.generator.config.block_length,
            gen_length=context.gen_length,
            denoising_steps=self.generator.config.denoising_steps,
            policy_mode=policy_mode,
            parent_rollout_id=parent_rollout_id,
            intervention_block_index=intervention_block_index,
            decisions=decisions,
            metadata={
                **example.metadata,
                "prompt_messages": example.prompt_messages,
                "rounded_total_length": generation_result.rounded_total_length,
                "effective_output_length": generation_result.effective_output_length,
                "decode_block_start": generation_result.decode_block_start,
                "generation_backend": generation_result.metadata.get("generation_backend"),
                "single_intervention": intervention_block_index is not None,
            },
        )

    @staticmethod
    def _make_rollout_id(prompt_id: str, suffix: str) -> str:
        return f"{prompt_id}::{suffix}::{uuid.uuid4().hex[:8]}"


def load_prompt_examples(path: str | Path, *, max_samples: int | None = None) -> list[PromptExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    if path.suffix == ".jsonl":
        examples = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                examples.append(_prompt_example_from_mapping(json.loads(line)))
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            if "examples" in payload and isinstance(payload["examples"], list):
                examples = [_prompt_example_from_mapping(item) for item in payload["examples"]]
            else:
                examples = [_prompt_example_from_mapping(payload)]
        else:
            examples = [_prompt_example_from_mapping(item) for item in payload]
    else:
        examples = [
            PromptExample(
                prompt_id=path.stem,
                prompt_text=path.read_text(encoding="utf-8"),
            )
        ]

    if max_samples is not None:
        return examples[:max_samples]
    return examples


def save_rollout_bundles(
    bundles: Iterable[RolloutBundle],
    output_dir: str | Path,
    save_filename: str,
) -> RolloutExportResult:
    bundles = list(bundles)
    output_path = ensure_dir(output_dir)
    supervision_path = output_path / save_filename
    base_path = output_path / _rollout_filename(save_filename, "base_rollouts")
    branch_path = output_path / _rollout_filename(save_filename, "branch_rollouts")
    schema_path = output_path / _schema_filename(save_filename)

    supervision_samples = [
        sample
        for bundle in bundles
        for sample in rollout_bundle_to_supervision_samples(bundle)
    ]
    supervision_lines = [
        json.dumps(sample, ensure_ascii=True)
        for sample in supervision_samples
    ]
    base_lines = [json.dumps(bundle.base_rollout.to_dict(), ensure_ascii=True) for bundle in bundles]
    branch_lines = [
        json.dumps(branch.to_dict(), ensure_ascii=True)
        for bundle in bundles
        for branch in bundle.counterfactual_rollouts
    ]

    supervision_path.write_text(
        "\n".join(supervision_lines) + ("\n" if supervision_lines else ""),
        encoding="utf-8",
    )
    base_path.write_text("\n".join(base_lines) + ("\n" if base_lines else ""), encoding="utf-8")
    branch_path.write_text("\n".join(branch_lines) + ("\n" if branch_lines else ""), encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            {
                "feature_names": list(DEFAULT_BLOCK_FEATURE_NAMES),
                "descriptions": dict(DEFAULT_BLOCK_FEATURE_DESCRIPTIONS),
                "version": "remask_policy_block_features_v1",
                "label_rule": "label = 1 if delta > remask_penalty_lambda else 0",
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RolloutExportResult(
        supervision_dataset_path=str(supervision_path),
        base_rollout_path=str(base_path),
        branch_rollout_path=str(branch_path),
        feature_schema_path=str(schema_path),
        num_supervision_samples=len(supervision_samples),
        num_base_rollouts=len(bundles),
        num_branch_rollouts=sum(len(bundle.counterfactual_rollouts) for bundle in bundles),
    )


def rollout_bundle_to_supervision_samples(bundle: RolloutBundle) -> list[dict[str, Any]]:
    base_rollout = bundle.base_rollout
    branch_rollouts_by_block = {
        branch.intervention_block_index: branch
        for branch in bundle.counterfactual_rollouts
        if branch.intervention_block_index is not None
    }
    samples: list[dict[str, Any]] = []
    for decision in base_rollout.decisions:
        if decision.label is None or decision.branch_reward is None or decision.reward_delta is None:
            continue
        branch_rollout = branch_rollouts_by_block.get(decision.block_index)
        samples.append(
            {
                "sample_id": f"{base_rollout.rollout_id}::block::{decision.block_index}",
                "prompt_id": base_rollout.prompt_id,
                "prompt_text": base_rollout.prompt_text,
                "block_index": decision.block_index,
                "state_features": decision.state_features,
                "base_reward": 0.0 if decision.base_reward is None else decision.base_reward.reward,
                "branch_reward": decision.branch_reward.reward,
                "delta": decision.reward_delta,
                "label": decision.label,
                "remask_penalty_lambda": decision.remask_penalty_lambda,
                "metadata": {
                    "base_rollout_id": base_rollout.rollout_id,
                    "branch_rollout_id": None if branch_rollout is None else branch_rollout.rollout_id,
                    "block_token_start": decision.block_token_start,
                    "block_token_end": decision.block_token_end,
                    "reference_text": base_rollout.reference_text,
                    "generated_text": base_rollout.generated_text,
                    "prompt_messages": base_rollout.metadata.get("prompt_messages"),
                },
            }
        )
    return samples


def _prompt_example_from_mapping(payload: Mapping[str, Any]) -> PromptExample:
    prompt_id = payload.get("prompt_id") or payload.get("id") or uuid.uuid4().hex[:8]
    prompt_text = payload.get("prompt_text") or payload.get("prompt")
    prompt_messages = payload.get("prompt_messages") or payload.get("messages")
    reference_text = (
        payload.get("reference_text")
        or payload.get("reference")
        or payload.get("target")
        or payload.get("answer")
    )
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"prompt_id", "id", "prompt_text", "prompt", "prompt_messages", "messages", "reference_text", "reference", "target", "answer"}
    }
    return PromptExample(
        prompt_id=str(prompt_id),
        prompt_text=prompt_text,
        prompt_messages=prompt_messages,
        reference_text=reference_text,
        metadata=metadata,
    )


def _branch_filename(save_filename: str) -> str:
    path = Path(save_filename)
    if path.suffix:
        return f"{path.stem}.branches{path.suffix}"
    return f"{save_filename}.branches.jsonl"


def _rollout_filename(save_filename: str, suffix: str) -> str:
    path = Path(save_filename)
    if path.suffix:
        return f"{path.stem}.{suffix}{path.suffix}"
    return f"{save_filename}.{suffix}.jsonl"


def _schema_filename(save_filename: str) -> str:
    path = Path(save_filename)
    if path.suffix:
        return f"{path.stem}.schema.json"
    return f"{save_filename}.schema.json"
