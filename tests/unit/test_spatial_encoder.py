import torch

from src.models.spatial_encoder import SpatialEncoder3D


class TestSpatialEncoder3D:
    def test_output_shape(self, tiny_config):
        model = SpatialEncoder3D(tiny_config.model)
        x = torch.randn(2, 1, *tiny_config.mri.target_shape)
        tokens = model(x)
        assert tokens.shape == (2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)

    def test_batch_size_one(self, tiny_config):
        model = SpatialEncoder3D(tiny_config.model)
        model.eval()
        x = torch.randn(1, 1, *tiny_config.mri.target_shape)
        tokens = model(x)
        assert tokens.shape == (1, tiny_config.model.spatial_tokens, tiny_config.model.d_model)

    def test_gradients_flow_to_all_parameters(self, tiny_config):
        model = SpatialEncoder3D(tiny_config.model)
        x = torch.randn(2, 1, *tiny_config.mri.target_shape)
        tokens = model(x)
        tokens.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_last_conv_hook_target_participates_in_forward(self, tiny_config):
        model = SpatialEncoder3D(tiny_config.model)
        captured = {}

        def hook(module, inp, out):
            captured["shape"] = out.shape

        handle = model.last_conv.register_forward_hook(hook)
        x = torch.randn(1, 1, *tiny_config.mri.target_shape)
        model(x)
        handle.remove()

        assert "shape" in captured
        assert captured["shape"][1] == tiny_config.model.d_model

    def test_deterministic_in_eval_mode(self, tiny_config):
        model = SpatialEncoder3D(tiny_config.model)
        model.eval()
        x = torch.randn(1, 1, *tiny_config.mri.target_shape)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)
