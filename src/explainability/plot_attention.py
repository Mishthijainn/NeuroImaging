"""2D spatial-temporal cross-attention correlation heatmaps.

Reduces the multi-head cross-attention weights (B, H, S, T) produced by
`CrossAttentionFusion` down to a single (S, T) matrix -- spatial cortex
tokens against VEP temporal latency bins -- so clinicians can see which
brain regions the model correlated with which millisecond latency window.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def cross_attention_heatmap(attn_weights: torch.Tensor) -> np.ndarray:
    """Average (B, H, S, T) attention weights over batch and heads -> (S, T)."""
    if attn_weights.dim() != 4:
        raise ValueError(
            f"Expected (B, H, S, T) attention weights, got shape {tuple(attn_weights.shape)}"
        )
    reduced = attn_weights.mean(dim=(0, 1))
    return reduced.detach().cpu().numpy()


def plot_cross_attention(
    attn_weights: torch.Tensor,
    save_path: str,
    xlabel: str = "Temporal latency bin",
    ylabel: str = "Spatial cortex token",
    title: str = "Spatial-Temporal Cross-Attention",
) -> np.ndarray:
    """Render and save the cross-attention heatmap; returns the plotted array."""
    heatmap = cross_attention_heatmap(attn_weights)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return heatmap
