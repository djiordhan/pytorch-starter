"""Simple MLP for tabular binary classification."""

from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)
