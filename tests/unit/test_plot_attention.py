import numpy as np
import pytest
import torch

from src.explainability.plot_attention import cross_attention_heatmap, plot_cross_attention


class TestCrossAttentionHeatmap:
    def test_output_shape_averages_batch_and_heads(self):
        weights = torch.rand(4, 8, 64, 16)  # (B, H, S, T)
        heatmap = cross_attention_heatmap(weights)
        assert heatmap.shape == (64, 16)
        assert isinstance(heatmap, np.ndarray)

    def test_values_are_nonnegative(self):
        weights = torch.softmax(torch.randn(2, 4, 10, 5), dim=-1)
        heatmap = cross_attention_heatmap(weights)
        assert np.all(heatmap >= 0.0)

    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(ValueError, match="B, H, S, T"):
            cross_attention_heatmap(torch.rand(4, 16))


class TestPlotCrossAttention:
    def test_saves_png_and_returns_heatmap(self, tmp_path):
        weights = torch.softmax(torch.randn(3, 4, 20, 10), dim=-1)
        save_path = tmp_path / "attn.png"

        heatmap = plot_cross_attention(weights, str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        assert heatmap.shape == (20, 10)

    def test_heatmap_matches_manual_reduction(self, tmp_path):
        weights = torch.softmax(torch.randn(2, 4, 8, 6), dim=-1)
        heatmap = plot_cross_attention(weights, str(tmp_path / "attn.png"))
        expected = weights.mean(dim=(0, 1)).numpy()
        np.testing.assert_allclose(heatmap, expected, rtol=1e-5)
