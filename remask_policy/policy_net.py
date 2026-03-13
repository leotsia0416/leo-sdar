from __future__ import annotations

import torch
from torch import nn


class RemaskPolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims]
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(previous_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        logits = self.output_layer(hidden)
        return logits.squeeze(-1)

    def architecture_config(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }
