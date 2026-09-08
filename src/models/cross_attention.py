"""Multi-Head Cross-Attention Fusion Module (CAFM).

Spatial 3D tokens are projected as Queries; temporal 1D latency tokens are
projected as Keys and Values, so each spatial brain region dynamically
queries the electrophysiological latency bins:

    Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
"""

import math

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG


class MultiHeadCrossAttention(nn.Module):
    """Q from `query_tokens`, K/V from `kv_tokens`. Sequence lengths may differ."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.d_model = cfg.d_model

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        b = query_tokens.shape[0]
        q = self._split_heads(self.w_q(query_tokens))
        k = self._split_heads(self.w_k(kv_tokens))
        v = self._split_heads(self.w_v(kv_tokens))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # (B, H, Nq, Nk)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, Nq, Dh)

        context = context.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.out_proj(context)
        return out, attn


class CrossAttentionFusion(nn.Module):
    """CAFM: cross-attention -> residual + LayerNorm -> feed-forward block."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, spatial_tokens: torch.Tensor, temporal_tokens: torch.Tensor):
        attn_out, attn_weights = self.cross_attn(spatial_tokens, temporal_tokens)
        fused = self.norm1(spatial_tokens + attn_out)
        ffn_out = self.ffn(fused)
        fused = self.norm2(fused + ffn_out)
        return fused, attn_weights
