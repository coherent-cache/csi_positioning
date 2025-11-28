"""Model definitions for the Location CNN workflow."""

from __future__ import annotations

import math

import torch
from torch import nn


class LocationCNN(nn.Module):
    """Compact CNN that predicts 3D positions from CSI maps."""

    def __init__(self, in_channels: int = 4, num_outputs: int = 3) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=(1, 2)),  Removed to reduce complexity
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=(2, 2)),  Removed to reduce complexity
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 12, 128),  # Adjusted input size for new strides
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass that keeps operations Concrete-ML friendly."""

        features = self.feature_extractor(inputs)
        return self.regressor(features)
