import torch

from src.models.temporal_encoder import TemporalEncoder1D


class TestTemporalEncoder1D:
    def test_output_shape(self, tiny_config):
        model = TemporalEncoder1D(tiny_config.model)
        x = torch.randn(2, 1, tiny_config.vep.length)
        tokens = model(x)
        assert tokens.shape == (2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)

    def test_batch_size_one(self, tiny_config):
        model = TemporalEncoder1D(tiny_config.model)
        model.eval()
        x = torch.randn(1, 1, tiny_config.vep.length)
        tokens = model(x)
        assert tokens.shape == (1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)

    def test_gradients_flow_to_all_parameters(self, tiny_config):
        model = TemporalEncoder1D(tiny_config.model)
        x = torch.randn(2, 1, tiny_config.vep.length)
        tokens = model(x)
        tokens.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_robust_to_varying_input_length(self, tiny_config):
        model = TemporalEncoder1D(tiny_config.model)
        model.eval()
        for length in (100, 200, 333):
            x = torch.randn(1, 1, length)
            tokens = model(x)
            assert tokens.shape == (1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
