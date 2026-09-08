"""Cross-Attention Fusion Network (CAFN): full model wiring.

Dual-stream encoders feed the cross-attention fusion module; a fused
classification head and two auxiliary unimodal heads train jointly
(multi-task supervision) so the network cannot collapse onto a single
dominant modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig, DEFAULT_CONFIG
from src.models.spatial_encoder import SpatialEncoder3D
from src.models.temporal_encoder import TemporalEncoder1D
from src.models.cross_attention import CrossAttentionFusion


class CAFN(nn.Module):
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        self.cfg = cfg
        self.spatial_encoder = SpatialEncoder3D(cfg)
        self.temporal_encoder = TemporalEncoder1D(cfg)
        self.fusion = CrossAttentionFusion(cfg)

        self.fused_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.spatial_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.temporal_head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, mri_volume: torch.Tensor, vep_signal: torch.Tensor) -> dict:
        spatial_tokens = self.spatial_encoder(mri_volume)    # (B, S, d)
        temporal_tokens = self.temporal_encoder(vep_signal)  # (B, T, d)

        fused_tokens, attn_weights = self.fusion(spatial_tokens, temporal_tokens)

        fused_embedding = fused_tokens.mean(dim=1)
        spatial_embedding = spatial_tokens.mean(dim=1)
        temporal_embedding = temporal_tokens.mean(dim=1)

        return {
            "logits_fused": self.fused_head(fused_embedding),
            "logits_spatial": self.spatial_head(spatial_embedding),
            "logits_temporal": self.temporal_head(temporal_embedding),
            "attn_weights": attn_weights,  # (B, num_heads, S, T)
            "fused_tokens": fused_tokens,
            "spatial_tokens": spatial_tokens,
            "temporal_tokens": temporal_tokens,
        }

    def multi_task_loss(self, outputs: dict, labels: torch.Tensor, aux_weight: float = 0.3):
        loss_fused = F.cross_entropy(outputs["logits_fused"], labels)
        loss_spatial = F.cross_entropy(outputs["logits_spatial"], labels)
        loss_temporal = F.cross_entropy(outputs["logits_temporal"], labels)
        total = loss_fused + aux_weight * (loss_spatial + loss_temporal)
        return total, {
            "loss_fused": loss_fused.item(),
            "loss_spatial": loss_spatial.item(),
            "loss_temporal": loss_temporal.item(),
            "loss_total": total.item(),
        }
