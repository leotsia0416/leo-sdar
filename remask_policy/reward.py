from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .interfaces import RewardResult, RolloutRecord

HookReturn = RewardResult | float | int | bool


class BaseRewardAdapter:
    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        raise NotImplementedError


@dataclass
class DummyRewardAdapter(BaseRewardAdapter):
    value: float = 0.0

    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        score = float(self.value)
        return RewardResult(
            reward=score,
            components={"dummy": score},
            metadata={"adapter": "dummy", "prompt_id": rollout.prompt_id},
        )


@dataclass
class ExactMatchRewardAdapter(BaseRewardAdapter):
    case_sensitive: bool = False
    strip: bool = True
    normalize_whitespace: bool = True

    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        reference = rollout.reference_text or rollout.metadata.get("reference_text")
        prediction = rollout.generated_text or ""
        if reference is None:
            return RewardResult(
                reward=0.0,
                components={"exact_match": 0.0},
                metadata={"adapter": "exact_match", "missing_reference": True},
            )

        normalized_prediction = self._normalize(prediction)
        normalized_reference = self._normalize(reference)
        score = 1.0 if normalized_prediction == normalized_reference else 0.0
        return RewardResult(
            reward=score,
            components={"exact_match": score},
            metadata={
                "adapter": "exact_match",
                "normalized_prediction": normalized_prediction,
                "normalized_reference": normalized_reference,
            },
        )

    def _normalize(self, text: str) -> str:
        if self.strip:
            text = text.strip()
        if self.normalize_whitespace:
            text = " ".join(text.split())
        if not self.case_sensitive:
            text = text.lower()
        return text


@dataclass
class FormatValidityRewardAdapter(BaseRewardAdapter):
    pattern: str | None = None
    fullmatch: bool = False
    require_non_empty: bool = True

    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        prediction = rollout.generated_text or ""
        if self.require_non_empty and not prediction.strip():
            return RewardResult(
                reward=0.0,
                components={"format_validity": 0.0},
                metadata={"adapter": "format_validity", "reason": "empty_prediction"},
            )

        if self.pattern is None:
            score = 1.0
            is_valid = True
        else:
            matcher = re.fullmatch if self.fullmatch else re.search
            is_valid = matcher(self.pattern, prediction) is not None
            score = 1.0 if is_valid else 0.0

        return RewardResult(
            reward=score,
            components={"format_validity": score},
            metadata={
                "adapter": "format_validity",
                "pattern": self.pattern,
                "is_valid": is_valid,
            },
        )


@dataclass
class HookRewardAdapter(BaseRewardAdapter):
    hook: Callable[[RolloutRecord], HookReturn]

    def evaluate(self, rollout: RolloutRecord) -> RewardResult:
        result = self.hook(rollout)
        if isinstance(result, RewardResult):
            return result
        score = float(result)
        return RewardResult(
            reward=score,
            components={"hook": score},
            metadata={"adapter": "hook"},
        )


def build_reward_adapter(kind: str, **kwargs: Any) -> BaseRewardAdapter:
    normalized = kind.lower()
    if normalized == "dummy":
        return DummyRewardAdapter(value=float(kwargs.get("value", 0.0)))
    if normalized == "exact_match":
        return ExactMatchRewardAdapter(
            case_sensitive=bool(kwargs.get("case_sensitive", False)),
            strip=bool(kwargs.get("strip", True)),
            normalize_whitespace=bool(kwargs.get("normalize_whitespace", True)),
        )
    if normalized in {"format_validity", "regex_validity"}:
        return FormatValidityRewardAdapter(
            pattern=kwargs.get("pattern"),
            fullmatch=bool(kwargs.get("fullmatch", False)),
            require_non_empty=bool(kwargs.get("require_non_empty", True)),
        )
    raise ValueError(f"Unknown reward adapter: {kind}")
