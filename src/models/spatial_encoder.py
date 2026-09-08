"""3D-CNN / ResNet backbone: structural MRI volume -> spatial tokens."""

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SpatialEncoder3D(nn.Module):
    """(B, 1, D, H, W) -> (B, spatial_tokens, d_model) spatial token sequence.

    `self.last_conv` exposes the final residual block so Grad-CAM3D can hook
    its output activation map.
    """

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock3D(16, 32, stride=2)
        self.layer2 = ResidualBlock3D(32, 64, stride=2)
        self.layer3 = ResidualBlock3D(64, cfg.d_model, stride=2)
        self.last_conv = self.layer3

        self.pool = nn.AdaptiveAvgPool3d(cfg.spatial_pool)
        self.d_model = cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)  # (B, d_model, *spatial_pool)
        tokens = x.flatten(2).transpose(1, 2)  # (B, tokens, d_model)
        return tokens
