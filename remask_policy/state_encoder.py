from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

DEFAULT_BLOCK_FEATURE_NAMES = (
    "normalized_block_index",
    "normalized_progress",
    "prefix_length",
    "current_block_length",
    "mean_confidence",
    "min_confidence",
    "max_confidence",
    "mean_entropy",
    "min_entropy",
    "max_entropy",
    "stop_token_seen",
    "verifier_score",
)

DEFAULT_BLOCK_FEATURE_DESCRIPTIONS = {
    "normalized_block_index": "Block index normalized to [0, 1] within the decoded block sequence.",
    "normalized_progress": "Generated-token progress normalized by the requested generation length.",
    "prefix_length": "Number of tokens available before the current block starts.",
    "current_block_length": "Token count of the current block.",
    "mean_confidence": "Mean token confidence for the current block, or 0 if unavailable.",
    "min_confidence": "Minimum token confidence for the current block, or 0 if unavailable.",
    "max_confidence": "Maximum token confidence for the current block, or 0 if unavailable.",
    "mean_entropy": "Mean token entropy for the current block, or 0 if unavailable.",
    "min_entropy": "Minimum token entropy for the current block, or 0 if unavailable.",
    "max_entropy": "Maximum token entropy for the current block, or 0 if unavailable.",
    "stop_token_seen": "1 if any stopping token has appeared in the generated prefix up to this block, else 0.",
    "verifier_score": "Optional verifier signal copied from prompt or rollout metadata, defaulting to 0.",
}


def build_block_state_features(
    *,
    block_index: int,
    total_decode_blocks: int,
    token_start: int,
    token_end: int,
    token_ids: list[int],
    prompt_tokens_in_block: int,
    generated_tokens_in_block: int,
    prompt_length: int,
    requested_gen_length: int,
    output_token_ids: list[int],
    stopping_criteria_idx: list[int] | None,
    token_confidences: list[float] | None = None,
    token_entropies: list[float] | None = None,
    verifier_score: float = 0.0,
) -> dict[str, float]:
    total_blocks_denominator = max(1, total_decode_blocks - 1)
    generated_target = max(1, requested_gen_length)
    generated_progress = max(
        0,
        min(requested_gen_length, len(output_token_ids) - prompt_length),
    )

    stop_ids = set(stopping_criteria_idx or [])
    stop_slice = output_token_ids[prompt_length:min(len(output_token_ids), token_end)]
    confidence_values = list(token_confidences or [])
    entropy_values = list(token_entropies or [])

    return {
        "normalized_block_index": float(block_index) / float(total_blocks_denominator),
        "normalized_progress": float(generated_progress) / float(generated_target),
        "prefix_length": float(token_start),
        "current_block_length": float(len(token_ids)),
        "mean_confidence": _safe_mean(confidence_values),
        "min_confidence": _safe_min(confidence_values),
        "max_confidence": _safe_max(confidence_values),
        "mean_entropy": _safe_mean(entropy_values),
        "min_entropy": _safe_min(entropy_values),
        "max_entropy": _safe_max(entropy_values),
        "stop_token_seen": 1.0 if stop_ids and any(token in stop_ids for token in stop_slice) else 0.0,
        "verifier_score": float(verifier_score),
    }


@dataclass
class StateTensorEncoder:
    feature_names: list[str]
    mean: list[float] | None = None
    std: list[float] | None = None
    normalize: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._device = self._resolve_device(self.device)
        if self.mean is None:
            self.mean = [0.0] * len(self.feature_names)
        if self.std is None:
            self.std = [1.0] * len(self.feature_names)
        if len(self.mean) != len(self.feature_names):
            raise ValueError("mean length must match feature_names length.")
        if len(self.std) != len(self.feature_names):
            raise ValueError("std length must match feature_names length.")
        self._mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=self._device)
        self._std_tensor = torch.tensor(self.std, dtype=torch.float32, device=self._device)
        self._std_tensor = torch.where(
            self._std_tensor < 1e-6,
            torch.ones_like(self._std_tensor),
            self._std_tensor,
        )

    def encode(self, state_features: Mapping[str, Any]) -> torch.Tensor:
        vector = torch.tensor(
            [float(state_features.get(name, 0.0)) for name in self.feature_names],
            dtype=torch.float32,
            device=self._device,
        )
        if self.normalize:
            vector = (vector - self._mean_tensor) / self._std_tensor
        return vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean or []),
            "std": list(self.std or []),
            "normalize": self.normalize,
            "device": str(self._device),
        }

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_min(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(min(values))


def _safe_max(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))
