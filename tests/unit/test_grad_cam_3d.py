import os

import numpy as np
import torch

from src.explainability.grad_cam_3d import GradCAM3D, compute_gradcam, plot_gradcam_slices
from src.models.cafn import CAFN


class TestGradCAM3D:
    def test_cam_shape_matches_input_volume(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(2, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(2, 1, tiny_config.vep.length)

        cam, target_class = compute_gradcam(model, mri, vep)
        assert cam.shape == (2, *tiny_config.mri.target_shape)
        assert target_class.shape == (2,)

    def test_cam_values_in_unit_range(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(1, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(1, 1, tiny_config.vep.length)

        cam, _ = compute_gradcam(model, mri, vep)
        assert torch.all(cam >= 0.0)
        assert torch.all(cam <= 1.0 + 1e-5)
        assert torch.all(torch.isfinite(cam))

    def test_explicit_target_class_is_respected(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(2, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(2, 1, tiny_config.vep.length)

        _, target_class = compute_gradcam(model, mri, vep, target_class=1)
        assert torch.equal(target_class, torch.tensor([1, 1]))

    def test_hooks_are_removed_after_generate(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(1, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(1, 1, tiny_config.vep.length)
        compute_gradcam(model, mri, vep)

        layer = model.spatial_encoder.last_conv
        assert len(layer._forward_hooks) == 0
        assert len(layer._backward_hooks) == 0

    def test_model_left_in_original_training_mode(self, tiny_config):
        model = CAFN(tiny_config.model)
        model.train()
        mri = torch.randn(1, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(1, 1, tiny_config.vep.length)
        compute_gradcam(model, mri, vep)
        assert model.training is True

    def test_manual_engine_generate_and_remove(self, tiny_config):
        model = CAFN(tiny_config.model)
        engine = GradCAM3D(model)
        mri = torch.randn(1, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(1, 1, tiny_config.vep.length)
        cam, _ = engine.generate(mri, vep)
        engine.remove_hooks()
        assert cam.shape == (1, *tiny_config.mri.target_shape)


class TestPlotGradcamSlices:
    def test_saves_png_file(self, tmp_path, tiny_config):
        shape = tiny_config.mri.target_shape
        volume = np.random.rand(*shape).astype(np.float32)
        cam = np.random.rand(*shape).astype(np.float32)
        save_path = tmp_path / "gradcam.png"

        plot_gradcam_slices(volume, cam, str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_rejects_mismatched_shapes(self, tmp_path):
        volume = np.random.rand(10, 10, 10).astype(np.float32)
        cam = np.random.rand(8, 8, 8).astype(np.float32)
        try:
            plot_gradcam_slices(volume, cam, str(tmp_path / "out.png"))
            assert False, "expected ValueError for mismatched shapes"
        except ValueError:
            pass
