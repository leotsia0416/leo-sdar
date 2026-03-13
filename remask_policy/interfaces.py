from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .utils import SerializableMixin


@dataclass
class RewardResult(SerializableMixin):
    reward: float
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision(SerializableMixin):
    should_remask: bool
    score: float
    threshold: float
    block_index: int
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockDecisionRecord(SerializableMixin):
    prompt_id: str
    rollout_id: str
    block_index: int
    block_token_start: int
    block_token_end: int
    state_features: dict[str, float] = field(default_factory=dict)
    base_reward: RewardResult | None = None
    branch_reward: RewardResult | None = None
    reward_delta: float | None = None
    remask_penalty_lambda: float = 0.0
    label: int | None = None
    policy_decision: PolicyDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutRecord(SerializableMixin):
    rollout_id: str
    prompt_id: str
    prompt_text: str
    reference_text: str | None = None
    generated_text: str | None = None
    model_dir: str | None = None
    trust_remote_code: bool = False
    device: str = "cuda"
    dtype: str = "float16"
    block_length: int = 4
    gen_length: int = 128
    denoising_steps: int = 4
    policy_mode: str = "disabled"
    parent_rollout_id: str | None = None
    intervention_block_index: int | None = None
    decisions: list[BlockDecisionRecord] = field(default_factory=list)
    base_reward: RewardResult | None = None
    final_reward: RewardResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StateEncoder(Protocol):
    def encode_block(self, rollout: RolloutRecord, block_index: int) -> dict[str, float]:
        """Extract policy features for one completed block."""


class RewardAdapter(Protocol):
    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        """Convert a rollout into a scalar reward signal."""


class RemaskPolicy(Protocol):
    def decide(self, block_record: BlockDecisionRecord) -> PolicyDecision:
        """Score whether a completed block should be remasked."""
