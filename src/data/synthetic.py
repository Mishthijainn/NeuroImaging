"""Synthetic MRI + VEP generators.

The datasets named in Project.md (OpenNeuro, HCP Retinotopy, OASIS, MNE
sample, PhysioNet) provide structural/functional scans and generic
electrophysiology, but none carry the clinical Healthy/Glaucoma/Optic
Neuritis-CVI labels this model targets -- that requires a curated clinical
cohort. This module generates class-conditioned synthetic MRI volumes and
VEP waveforms with the same interface real data would have, so the rest of
the system (preprocessing, model, training, explainability) can be built,
exercised, and tested end-to-end while a labeled cohort is sourced
separately.

Class-conditioning is deliberately simplistic but structured so a model can
learn from it: label 0 (Healthy) is unstructured background; label 1
(Glaucoma) has focal MRI intensity loss (atrophy proxy) and reduced VEP
amplitude; label 2 (Optic Neuritis / CVI) has focal MRI hyperintensity
(lesion proxy) and delayed VEP P100 latency (the clinical hallmark of optic
neuritis).
"""

import csv
import os

import nibabel as nib
import numpy as np

from src.config import Config, DEFAULT_CONFIG

NUM_CLASSES = DEFAULT_CONFIG.model.num_classes


def synthetic_mri_volume(
    label: int, shape: tuple[int, int, int] = (80, 96, 64), rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a class-conditioned synthetic 3D MRI-like volume."""
    rng = rng or np.random.default_rng()
    volume = rng.normal(loc=100.0, scale=15.0, size=shape).clip(min=0)
    d, h, w = shape
    if label == 1:  # Glaucoma: focal atrophy (intensity loss)
        volume[d // 2 :, : h // 2, : w // 2] *= 0.4
    elif label == 2:  # Optic Neuritis / CVI: focal lesion (hyperintensity)
        volume[: d // 2, h // 2 :, w // 2 :] *= 1.8
    return volume.astype(np.float32)


def synthetic_vep_signal(
    label: int, fs: float = 250.0, duration_s: float = 2.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a class-conditioned synthetic VEP waveform (N75-P100-N145)."""
    rng = rng or np.random.default_rng()
    t = np.arange(0.0, duration_s, 1.0 / fs)

    if label == 0:  # Healthy
        latency, amplitude = 0.100, 6.0
    elif label == 1:  # Glaucoma: reduced amplitude, near-normal latency
        latency, amplitude = 0.108, 2.5
    else:  # Optic Neuritis / CVI: delayed P100 latency
        latency, amplitude = 0.140, 4.0

    p100 = amplitude * np.exp(-((t - latency) ** 2) / (2 * 0.012**2))
    n75 = -0.5 * amplitude * np.exp(-((t - (latency - 0.025)) ** 2) / (2 * 0.010**2))
    n145 = -0.4 * amplitude * np.exp(-((t - (latency + 0.045)) ** 2) / (2 * 0.015**2))
    waveform = p100 + n75 + n145

    drift = 0.3 * np.sin(2 * np.pi * 0.3 * t)
    mains = 0.15 * np.sin(2 * np.pi * 50.0 * t)
    emg_noise = rng.normal(0.0, 0.3, size=t.shape)

    return (waveform + drift + mains + emg_noise).astype(np.float64)


def write_synthetic_dataset(
    root: str,
    samples_per_class: int = 4,
    cfg: Config = DEFAULT_CONFIG,
    seed: int = 0,
) -> str:
    """Write synthetic NIfTI + VEP CSV files and a manifest.csv under `root`.

    Mirrors the data/raw/{mri,vep} layout described in Project.md. Returns
    the manifest CSV path, with columns: subject_id, mri_path, vep_path, fs, label.
    """
    rng = np.random.default_rng(seed)
    mri_dir = os.path.join(root, "mri")
    vep_dir = os.path.join(root, "vep")
    os.makedirs(mri_dir, exist_ok=True)
    os.makedirs(vep_dir, exist_ok=True)

    manifest_path = os.path.join(root, "manifest.csv")
    rows = []
    subject_idx = 0
    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            subject_id = f"sub-{subject_idx:04d}"
            volume = synthetic_mri_volume(label, rng=rng)
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            img = nib.Nifti1Image(volume, affine)
            mri_path = os.path.join(mri_dir, f"{subject_id}_mri.nii.gz")
            nib.save(img, mri_path)

            signal = synthetic_vep_signal(label, fs=cfg.vep.sample_rate, rng=rng)
            vep_path = os.path.join(vep_dir, f"{subject_id}_vep.csv")
            with open(vep_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["amplitude"])
                for value in signal:
                    writer.writerow([value])

            rows.append(
                {
                    "subject_id": subject_id,
                    "mri_path": mri_path,
                    "vep_path": vep_path,
                    "fs": cfg.vep.sample_rate,
                    "label": label,
                }
            )
            subject_idx += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "mri_path", "vep_path", "fs", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path
