"""Central configuration for shapes, hyperparameters, and class labels.

Keeping these in one place guarantees the preprocessing stream, both
encoders, the fusion module, and the training loop all agree on tensor
shapes without hardcoding magic numbers in multiple files.
"""

from dataclasses import dataclass, field


CLASS_NAMES = ("Healthy Control", "Glaucoma", "Optic Neuritis / CVI")


@dataclass(frozen=True)
class MRIConfig:
    target_shape: tuple[int, int, int] = (64, 64, 64)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "RAS"
    foreground_percentile: float = 60.0  # intensity-threshold skull-strip approx


@dataclass(frozen=True)
class VEPConfig:
    length: int = 500
    sample_rate: float = 250.0  # Hz -> 500 samples spans 2.0s per ISCEV sweep
    bandpass_low: float = 1.0
    bandpass_high: float = 100.0
    notch_freq: float = 50.0
    notch_quality: float = 30.0


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    spatial_tokens: int = 64   # 4x4x4 adaptive pool
    temporal_tokens: int = 16
    spatial_pool: tuple[int, int, int] = (4, 4, 4)
    ffn_hidden: int = 256
    dropout: float = 0.1
    num_classes: int = len(CLASS_NAMES)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    aux_loss_weight: float = 0.3
    val_split: float = 0.2
    seed: int = 42
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class Config:
    mri: MRIConfig = field(default_factory=MRIConfig)
    vep: VEPConfig = field(default_factory=VEPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT_CONFIG = Config()
