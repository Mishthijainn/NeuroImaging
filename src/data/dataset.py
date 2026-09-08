"""PyTorch Dataset classes for paired MRI + VEP samples.

`PairedNeuroDataset` reads a manifest CSV (subject_id, mri_path, vep_path,
fs, label) and applies the real preprocessing pipeline -- this is what a
labeled clinical cohort would plug into. `SyntheticNeuroDataset` generates
class-conditioned synthetic samples in-memory (no disk I/O) through the same
preprocessing pipeline, used for fast unit tests, integration tests, and
development/training-loop validation ahead of real labeled data.
"""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import Config, DEFAULT_CONFIG
from src.data.synthetic import synthetic_mri_volume, synthetic_vep_signal
from src.preprocessing.mri_transforms import preprocess_mri_file, preprocess_mri_volume
from src.preprocessing.signal_cleaner import clean_vep_signal


class PairedNeuroDataset(Dataset):
    """Loads (MRI volume, VEP signal, label) triples from a manifest CSV."""

    def __init__(self, manifest_path: str, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        with open(manifest_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} has no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        mri = preprocess_mri_file(row["mri_path"], self.cfg.mri)

        raw_signal = np.loadtxt(row["vep_path"], delimiter=",", skiprows=1)
        fs = float(row.get("fs") or self.cfg.vep.sample_rate)
        vep = clean_vep_signal(raw_signal, fs, self.cfg.vep)

        return {
            "mri": torch.from_numpy(mri),
            "vep": torch.from_numpy(vep).unsqueeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": row["subject_id"],
        }


class SyntheticNeuroDataset(Dataset):
    """In-memory class-conditioned synthetic dataset, real preprocessing applied."""

    def __init__(
        self,
        num_samples: int,
        cfg: Config = DEFAULT_CONFIG,
        seed: int = 0,
        raw_mri_shape: tuple[int, int, int] = (80, 96, 64),
    ):
        self.cfg = cfg
        self.num_samples = num_samples
        self.raw_mri_shape = raw_mri_shape
        self.rng = np.random.default_rng(seed)
        num_classes = cfg.model.num_classes
        # Balanced labels, deterministically shuffled.
        base_labels = [i % num_classes for i in range(num_samples)]
        self.rng.shuffle(base_labels)
        self.labels = base_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        raw_volume = synthetic_mri_volume(label, shape=self.raw_mri_shape, rng=self.rng)
        mri = preprocess_mri_volume(raw_volume, zooms=(1.0, 1.0, 1.0), cfg=self.cfg.mri)

        raw_signal = synthetic_vep_signal(label, fs=self.cfg.vep.sample_rate, rng=self.rng)
        vep = clean_vep_signal(raw_signal, self.cfg.vep.sample_rate, self.cfg.vep)

        return {
            "mri": torch.from_numpy(mri),
            "vep": torch.from_numpy(vep).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": f"synthetic-{idx:04d}",
        }
