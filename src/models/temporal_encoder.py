"""1D-CNN latency backbone: VEP waveform -> temporal tokens."""

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG


class TemporalEncoder1D(nn.Module):
    """(B, 1, L) -> (B, temporal_tokens, d_model) temporal token sequence."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, cfg.d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(cfg.d_model),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(cfg.temporal_tokens)
        self.d_model = cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)          # (B, d_model, L')
        x = self.pool(x)         # (B, d_model, temporal_tokens)
        tokens = x.transpose(1, 2)  # (B, temporal_tokens, d_model)
        return tokens
