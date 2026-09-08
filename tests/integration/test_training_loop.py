"""Integration test for the multi-task training loop: verifies the system
doesn't just run without crashing, but actually learns from the
class-conditioned synthetic data, and that checkpoints round-trip.
"""

import copy

import torch

from src.config import Config, ModelConfig, TrainConfig, VEPConfig, MRIConfig
from src.data.dataset import SyntheticNeuroDataset
from src.models.cafn import CAFN
from src.train import evaluate, load_checkpoint, save_checkpoint, train_model, train_val_split


def _learnable_config(epochs: int) -> Config:
    return Config(
        mri=MRIConfig(target_shape=(32, 32, 32)),
        vep=VEPConfig(length=200, sample_rate=250.0),
        model=ModelConfig(
            d_model=32,
            num_heads=2,
            spatial_tokens=8,
            temporal_tokens=8,
            spatial_pool=(2, 2, 2),
            ffn_hidden=64,
            dropout=0.1,
            num_classes=3,
        ),
        train=TrainConfig(
            batch_size=4,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-4,
            aux_loss_weight=0.3,
            val_split=0.25,
            seed=0,
            checkpoint_dir="checkpoints",
        ),
    )


class TestTrainModel:
    def test_training_loss_decreases_substantially(self):
        cfg = _learnable_config(epochs=15)
        dataset = SyntheticNeuroDataset(num_samples=24, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(dataset, cfg.train.val_split, cfg.train.seed)

        model = CAFN(cfg.model)
        history = train_model(model, train_ds, val_ds, cfg=cfg)

        assert len(history["train_loss"]) == cfg.train.epochs
        first_loss, last_loss = history["train_loss"][0], history["train_loss"][-1]
        assert last_loss < first_loss * 0.7, (
            f"expected loss to drop substantially, got {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_model_learns_to_separate_classes(self):
        cfg = _learnable_config(epochs=15)
        dataset = SyntheticNeuroDataset(num_samples=24, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(dataset, cfg.train.val_split, cfg.train.seed)

        model = CAFN(cfg.model)
        train_model(model, train_ds, val_ds, cfg=cfg)

        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.batch_size)
        metrics = evaluate(model, val_loader, torch.device("cpu"))
        assert metrics["accuracy"] > 0.5

    def test_runs_without_validation_set(self):
        cfg = _learnable_config(epochs=2)
        dataset = SyntheticNeuroDataset(num_samples=8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        model = CAFN(cfg.model)
        history = train_model(model, dataset, val_dataset=None, cfg=cfg)
        assert len(history["train_loss"]) == cfg.train.epochs
        assert history["val_loss"] == []


class TestCheckpointRoundTrip:
    def test_save_and_load_preserves_behavior(self, tmp_path):
        cfg = _learnable_config(epochs=2)
        dataset = SyntheticNeuroDataset(num_samples=8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        model = CAFN(cfg.model)
        train_model(model, dataset, cfg=cfg)
        model.eval()

        checkpoint_path = str(tmp_path / "model.pt")
        save_checkpoint(model, checkpoint_path, cfg)

        reloaded = load_checkpoint(checkpoint_path, cfg)
        reloaded.eval()

        item = dataset[0]
        mri = item["mri"].unsqueeze(0)
        vep = item["vep"].unsqueeze(0)
        with torch.no_grad():
            out_before = model(mri, vep)["logits_fused"]
            out_after = reloaded(mri, vep)["logits_fused"]

        torch.testing.assert_close(out_before, out_after)
