import nibabel as nib
import numpy as np
import pytest
import torch

from src.config import Config, MRIConfig, ModelConfig, TrainConfig, VEPConfig


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tiny_config() -> Config:
    """Small-scale config so model/training tests run in well under a second."""
    mri = MRIConfig(target_shape=(32, 32, 32), foreground_percentile=60.0)
    vep = VEPConfig(length=200, sample_rate=250.0)
    model = ModelConfig(
        d_model=32,
        num_heads=2,
        spatial_tokens=8,
        temporal_tokens=8,
        spatial_pool=(2, 2, 2),
        ffn_hidden=64,
        dropout=0.1,
        num_classes=3,
    )
    train = TrainConfig(
        batch_size=4,
        epochs=3,
        lr=1e-3,
        weight_decay=1e-4,
        aux_loss_weight=0.3,
        val_split=0.25,
        seed=0,
        checkpoint_dir="checkpoints",
    )
    return Config(mri=mri, vep=vep, model=model, train=train)


@pytest.fixture
def synthetic_nifti_path(tmp_path, rng):
    """Factory: writes a synthetic NIfTI file with non-isotropic spacing to disk."""

    def _make(shape=(80, 96, 64), affine=None, filename="sample.nii.gz"):
        data = rng.random(shape).astype(np.float32) * 200.0
        aff = affine if affine is not None else np.diag([1.5, 1.5, 2.0, 1.0])
        img = nib.Nifti1Image(data, aff)
        path = tmp_path / filename
        nib.save(img, str(path))
        return str(path)

    return _make


@pytest.fixture
def synthetic_vep_array(rng):
    """A VEP-like signal with a known 10Hz component and 50Hz line noise."""
    fs = 250.0
    t = np.arange(0, 2.0, 1 / fs)
    signal = (
        np.sin(2 * np.pi * 10 * t)
        + 0.5 * np.sin(2 * np.pi * 50 * t)
        + 0.2 * rng.standard_normal(len(t))
        + 3.0
    )
    return signal, fs
