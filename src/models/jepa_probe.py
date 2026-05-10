"""Scaffolding for JEPA feature probing."""

from __future__ import annotations

from torch import Tensor, nn


class JepaLinearProbe(nn.Module):
    """Lightweight MLP probe over frozen JEPA features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return logits from pre-extracted feature tensors."""
        return self.probe(features)
