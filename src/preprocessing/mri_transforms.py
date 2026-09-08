"""3D sMRI spatial standardization: RAS reorientation, 1mm isotropic
resampling, intensity-threshold foreground masking, and normalization.

Note on skull-stripping: true skull-stripping (FSL BET / HD-BET / SynthStrip)
requires an external binary or a pretrained segmentation network that is not
pip-installable. This module ships a lightweight intensity-percentile
foreground mask as a prototype-stage approximation -- it zeroes low-intensity
background/skull voxels without needing an external dependency. Swap in
HD-BET or SynthStrip for clinical-grade deployment; the rest of the pipeline
is agnostic to which foreground mask produced its input.
"""

import nibabel as nib
import numpy as np
from monai.transforms import Resize, ResizeWithPadOrCrop, ScaleIntensity

from src.config import MRIConfig, DEFAULT_CONFIG


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume, reoriented to RAS (nibabel's closest-canonical).

    Returns (data, affine, voxel_zooms).
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if any(z <= 0 for z in zooms):
        raise ValueError(f"Invalid voxel spacing in header: {zooms}")
    return data, img.affine, zooms


def foreground_mask_threshold(volume: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out background/skull voxels below an intensity percentile.

    Percentile is computed over strictly-positive voxels so that an
    already-large air/background region does not collapse the threshold to
    zero.
    """
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return volume
    threshold = np.percentile(nonzero, percentile)
    masked = volume.copy()
    masked[masked < threshold] = 0.0
    return masked


def resample_isotropic(
    volume: np.ndarray,
    zooms: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Resample a volume to isotropic spacing by scaling voxel counts.

    MONAI's shape-based `Resize` combined with a zoom-derived target shape
    is mathematically equivalent to physical-spacing resampling and avoids
    MetaTensor/affine bookkeeping for a standalone array pipeline.
    """
    new_shape = tuple(
        max(1, round(dim * zoom / target))
        for dim, zoom, target in zip(volume.shape, zooms, target_spacing)
    )
    resize = Resize(spatial_size=new_shape, mode="trilinear", align_corners=False)
    resized = resize(volume[np.newaxis, ...])  # add channel dim
    return np.asarray(resized[0])


def preprocess_mri_volume(
    volume: np.ndarray,
    zooms: tuple[float, float, float],
    cfg: MRIConfig = DEFAULT_CONFIG.mri,
) -> np.ndarray:
    """Full spatial standardization pipeline for an already-RAS volume.

    Returns a float32 array of shape (1, *cfg.target_shape) scaled to [0, 1].
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    isotropic = resample_isotropic(volume, zooms, cfg.target_spacing)
    masked = foreground_mask_threshold(isotropic, cfg.foreground_percentile)

    if masked.max() > masked.min():
        scaler = ScaleIntensity(minv=0.0, maxv=1.0)
        normalized = np.asarray(scaler(masked[np.newaxis, ...]))
    else:
        normalized = masked[np.newaxis, ...].astype(np.float32)

    pad_crop = ResizeWithPadOrCrop(spatial_size=cfg.target_shape)
    standardized = np.asarray(pad_crop(normalized))
    return standardized.astype(np.float32)


def preprocess_mri_file(path: str, cfg: MRIConfig = DEFAULT_CONFIG.mri) -> np.ndarray:
    """Convenience wrapper: load a NIfTI file and run full standardization."""
    data, _, zooms = load_nifti(path)
    return preprocess_mri_volume(data, zooms, cfg)
