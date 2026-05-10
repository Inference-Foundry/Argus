"""Baseline models for Argus."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNBaseline(nn.Module):
    """ResNet18 baseline adapted for binary classification."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, images: Tensor) -> Tensor:
        """Return logits for each image."""
        return self.backbone(images)
