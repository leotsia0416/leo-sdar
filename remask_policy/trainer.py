from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import RemaskTrainConfig
from .dataset import FeatureSchema, NormalizationStats
from .policy_net import RemaskPolicyMLP
from .utils import SerializableMixin, ensure_dir


@dataclass
class EpochMetrics(SerializableMixin):
    epoch: int
    train_loss: float
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float
    train_positive_prediction_rate: float
    val_loss: float | None = None
    val_accuracy: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None
    val_positive_prediction_rate: float | None = None


@dataclass
class TrainingSummary(SerializableMixin):
    checkpoint_path: str
    metrics_path: str
    best_epoch: int
    best_score: float
    train_size: int
    val_size: int
    history: list[EpochMetrics] = field(default_factory=list)


class RemaskPolicyTrainer:
    def __init__(
        self,
        config: RemaskTrainConfig,
        model: RemaskPolicyMLP,
        feature_schema: FeatureSchema,
        normalization_stats: NormalizationStats,
    ) -> None:
        self.config = config
        self.model = model
        self.feature_schema = feature_schema
        self.normalization_stats = normalization_stats
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(
        self,
        train_dataset,
        val_dataset=None,
    ) -> TrainingSummary:
        output_dir = ensure_dir(self.config.output_dir)
        checkpoint_path = output_dir / self.config.checkpoint_name
        metrics_path = output_dir / self.config.metrics_filename

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            if val_dataset is not None and len(val_dataset) > 0
            else None
        )

        history: list[EpochMetrics] = []
        best_epoch = 0
        best_score = float("inf")

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False) if val_loader is not None else None
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                train_precision=train_metrics["precision"],
                train_recall=train_metrics["recall"],
                train_f1=train_metrics["f1"],
                train_positive_prediction_rate=train_metrics["positive_prediction_rate"],
                val_loss=None if val_metrics is None else val_metrics["loss"],
                val_accuracy=None if val_metrics is None else val_metrics["accuracy"],
                val_precision=None if val_metrics is None else val_metrics["precision"],
                val_recall=None if val_metrics is None else val_metrics["recall"],
                val_f1=None if val_metrics is None else val_metrics["f1"],
                val_positive_prediction_rate=None
                if val_metrics is None
                else val_metrics["positive_prediction_rate"],
            )
            history.append(epoch_metrics)

            selection_score = (
                epoch_metrics.val_loss
                if epoch_metrics.val_loss is not None
                else epoch_metrics.train_loss
            )
            if selection_score < best_score:
                best_score = float(selection_score)
                best_epoch = epoch
                self.save_checkpoint(
                    checkpoint_path,
                    metrics=epoch_metrics.to_dict(),
                    train_size=len(train_dataset),
                    val_size=0 if val_dataset is None else len(val_dataset),
                )

        summary = TrainingSummary(
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
            best_epoch=best_epoch,
            best_score=best_score,
            train_size=len(train_dataset),
            val_size=0 if val_dataset is None else len(val_dataset),
            history=history,
        )
        metrics_path.write_text(summary.to_json() + "\n", encoding="utf-8")
        return summary

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metrics: dict[str, Any],
        train_size: int,
        val_size: int,
    ) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "model_kwargs": self.model.architecture_config(),
            "config": self.config.to_dict(),
            "feature_schema": self.feature_schema.to_dict(),
            "normalization_stats": self.normalization_stats.to_dict(),
            "metrics": metrics,
            "train_size": train_size,
            "val_size": val_size,
        }
        torch.save(payload, Path(path))

    def _run_epoch(self, data_loader, *, training: bool) -> dict[str, float]:
        if data_loader is None:
            raise ValueError("data_loader must not be None when running an epoch.")

        self.model.train(training)
        total_loss = 0.0
        total_examples = 0
        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []

        for features, labels in data_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(features)
            loss = self.loss_fn(logits, labels)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            logits_batches.append(logits.detach().cpu())
            label_batches.append(labels.detach().cpu())

        logits_tensor = torch.cat(logits_batches, dim=0)
        labels_tensor = torch.cat(label_batches, dim=0)
        metrics = _binary_metrics(
            logits_tensor,
            labels_tensor,
            threshold=self.config.policy_threshold,
        )
        metrics["loss"] = total_loss / max(1, total_examples)
        return metrics

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)


def load_trained_policy(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RemaskPolicyMLP, FeatureSchema, NormalizationStats, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    feature_schema = FeatureSchema.from_dict(payload["feature_schema"])
    normalization_stats = NormalizationStats.from_dict(payload["normalization_stats"])
    model = RemaskPolicyMLP(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, feature_schema, normalization_stats, payload


def _binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(torch.float32)
    labels = labels.to(torch.float32)

    true_positive = float(((predictions == 1.0) & (labels == 1.0)).sum().item())
    false_positive = float(((predictions == 1.0) & (labels == 0.0)).sum().item())
    false_negative = float(((predictions == 0.0) & (labels == 1.0)).sum().item())
    true_negative = float(((predictions == 0.0) & (labels == 0.0)).sum().item())

    total = max(1.0, true_positive + false_positive + false_negative + true_negative)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    positive_prediction_rate = float(predictions.mean().item())

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_prediction_rate": positive_prediction_rate,
    }
