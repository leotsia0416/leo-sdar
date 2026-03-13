from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset

from .state_encoder import DEFAULT_BLOCK_FEATURE_DESCRIPTIONS, DEFAULT_BLOCK_FEATURE_NAMES
from .utils import SerializableMixin


@dataclass
class FeatureSchema(SerializableMixin):
    feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCK_FEATURE_NAMES))
    descriptions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_BLOCK_FEATURE_DESCRIPTIONS)
    )
    version: str = "remask_policy_block_features_v1"
    label_rule: str | None = None

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)


@dataclass
class NormalizationStats(SerializableMixin):
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    enabled: bool = True

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.mean, dtype=torch.float32),
            torch.tensor(self.std, dtype=torch.float32),
        )


class BlockSupervisionDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        feature_schema: FeatureSchema,
        normalization_stats: NormalizationStats,
        *,
        normalize: bool = True,
    ) -> None:
        self.samples = samples
        self.feature_schema = feature_schema
        self.normalization_stats = normalization_stats
        self.normalize = normalize and normalization_stats.enabled

        features = torch.tensor(
            [vectorize_state_features(sample["state_features"], feature_schema) for sample in samples],
            dtype=torch.float32,
        )
        labels = torch.tensor([float(sample["label"]) for sample in samples], dtype=torch.float32)

        if self.normalize and len(samples) > 0:
            mean, std = normalization_stats.to_tensors()
            features = (features - mean) / std

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass
class RemaskDatasetBundle:
    train_dataset: BlockSupervisionDataset
    val_dataset: BlockSupervisionDataset | None
    feature_schema: FeatureSchema
    normalization_stats: NormalizationStats
    train_samples: list[dict[str, Any]]
    val_samples: list[dict[str, Any]]


def load_supervision_samples(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            _validate_sample(payload, path=path, line_number=line_number)
            records.append(payload)
    if not records:
        raise ValueError(f"No labeled supervision samples found in {path}.")
    return records


def load_feature_schema(path: str | Path) -> FeatureSchema:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureSchema.from_dict(payload)


def save_feature_schema(path: str | Path, schema: FeatureSchema) -> None:
    Path(path).write_text(schema.to_json() + "\n", encoding="utf-8")


def build_dataset_bundle(
    train_data_path: str | Path,
    *,
    eval_data_path: str | Path | None = None,
    feature_schema_path: str | Path | None = None,
    feature_names: Iterable[str] | None = None,
    val_split: float = 0.1,
    random_seed: int = 0,
    normalize_features: bool = True,
) -> RemaskDatasetBundle:
    all_train_samples = load_supervision_samples(train_data_path)
    feature_schema = _resolve_feature_schema(
        feature_schema_path=feature_schema_path,
        feature_names=feature_names,
    )

    if eval_data_path is not None:
        train_samples = all_train_samples
        val_samples = load_supervision_samples(eval_data_path)
    else:
        train_samples, val_samples = split_samples(
            all_train_samples,
            val_split=val_split,
            random_seed=random_seed,
        )

    normalization_stats = compute_normalization_stats(
        train_samples,
        feature_schema,
        enabled=normalize_features,
    )
    train_dataset = BlockSupervisionDataset(
        train_samples,
        feature_schema,
        normalization_stats,
        normalize=normalize_features,
    )
    val_dataset = (
        BlockSupervisionDataset(
            val_samples,
            feature_schema,
            normalization_stats,
            normalize=normalize_features,
        )
        if val_samples
        else None
    )

    return RemaskDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        feature_schema=feature_schema,
        normalization_stats=normalization_stats,
        train_samples=train_samples,
        val_samples=val_samples,
    )


def split_samples(
    samples: list[dict[str, Any]],
    *,
    val_split: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(samples) <= 1 or val_split <= 0.0:
        return list(samples), []

    indices = list(range(len(samples)))
    random.Random(random_seed).shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_split)))
    if val_count >= len(samples):
        val_count = len(samples) - 1

    val_indices = set(indices[:val_count])
    train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
    val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
    return train_samples, val_samples


def compute_normalization_stats(
    samples: list[dict[str, Any]],
    feature_schema: FeatureSchema,
    *,
    enabled: bool,
) -> NormalizationStats:
    if not enabled:
        return NormalizationStats(
            feature_names=list(feature_schema.feature_names),
            mean=[0.0] * feature_schema.input_dim,
            std=[1.0] * feature_schema.input_dim,
            enabled=False,
        )

    if not samples:
        raise ValueError("Cannot compute normalization stats without training samples.")

    features = torch.tensor(
        [vectorize_state_features(sample["state_features"], feature_schema) for sample in samples],
        dtype=torch.float32,
    )
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return NormalizationStats(
        feature_names=list(feature_schema.feature_names),
        mean=mean.tolist(),
        std=std.tolist(),
        enabled=True,
    )


def vectorize_state_features(
    state_features: Mapping[str, Any],
    feature_schema: FeatureSchema,
) -> list[float]:
    return [float(state_features.get(name, 0.0)) for name in feature_schema.feature_names]


def _resolve_feature_schema(
    *,
    feature_schema_path: str | Path | None,
    feature_names: Iterable[str] | None,
) -> FeatureSchema:
    if feature_schema_path is not None:
        return load_feature_schema(feature_schema_path)

    names = [name for name in (feature_names or []) if name]
    if not names:
        names = list(DEFAULT_BLOCK_FEATURE_NAMES)

    descriptions = {
        name: DEFAULT_BLOCK_FEATURE_DESCRIPTIONS.get(name, "")
        for name in names
    }
    return FeatureSchema(feature_names=names, descriptions=descriptions)


def _validate_sample(payload: Mapping[str, Any], *, path: str | Path, line_number: int) -> None:
    required_keys = {
        "prompt_id",
        "prompt_text",
        "block_index",
        "state_features",
        "base_reward",
        "branch_reward",
        "delta",
        "label",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise ValueError(
            f"Missing keys in supervision sample {path}:{line_number}: {', '.join(missing)}"
        )
