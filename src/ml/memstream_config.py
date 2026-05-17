"""
MemStream Configuration for NYC Taxi Anomaly Detection.

Hyperparameter configuration for the MemStream anomaly detection model.
These settings are verified against the benchmark v10 implementation.

Key Design Decisions:
    - in_dim=34: Verified canonical dimensionality from feature_extractor.py
    - hidden_dim=68: 2x input dimension (standard autoencoder practice)
    - out_dim=34: Symmetric autoencoder (encode and decode same dimension)
    - memory_len=2048: Minimum 2,048 slots per neighborhood for production
    - gamma=0.0: Prevents memory poisoning (fresh anomaly scores dominate)
    - warmup_epochs=500: Extended warmup for production quality
    - k=10: kNN neighbors for scoring

Scoring Method:
    - L1 kNN distance on AE output (no reconstruction error)
    - Memory stores AE output at out_dim=34
    - score >= 1.0 indicates anomaly

NOTE: PCA is PROHIBITED per MemStream Proposition 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["MemStreamConfig", "MemStreamArchitecture", "MemStreamDefaults"]


@dataclass
class MemStreamConfig:
    """
    Hyperparameters for MemStream anomaly detection.

    This configuration defines a symmetric autoencoder (34D -> 68D -> 34D)
    with a large memory module for storing normal patterns.

    Attributes:
        in_dim: Input feature dimension (34 for NYC taxi)
        hidden_dim: Hidden layer dimension (2x in_dim)
        out_dim: Output dimension (matches in_dim for symmetric AE)
        memory_len: Number of memory slots (minimum 2,048 for production)
        k: Number of kNN neighbors for scoring
        gamma: Decay factor for weighted kNN (0.0 = no decay)
        default_beta: Default anomaly threshold
        warmup_epochs: Number of warmup training epochs
        warmup_lr: Learning rate for warmup
        warmup_batch_size: Batch size for warmup
        warmup_noise_std: Noise standard deviation for denoising
        warmup_gradient_clip: Gradient clipping threshold
        warmup_early_stop_patience: Early stopping patience
        seed: Random seed for determinism

    Example:
        >>> config = MemStreamConfig()
        >>> print(f"Input: {config.in_dim}, Hidden: {config.hidden_dim}")
        Input: 34, Hidden: 68
    """

    # Architecture (verified 34D from benchmark v10)
    in_dim: int = 34
    hidden_dim: int = 68
    out_dim: int = 34

    # Memory (production scale)
    memory_len: int = 2048
    memory_init_fraction: float = 0.1

    # kNN scoring (v10 benchmark)
    k: int = 10
    gamma: float = 0.0  # No decay — prevents memory poisoning

    # Training (warmup)
    warmup_lr: float = 1e-3
    warmup_epochs: int = 500
    warmup_batch_size: int = 256
    warmup_noise_std: float = 0.1
    warmup_gradient_clip: float = 1.0
    warmup_early_stop_patience: int = 20

    # Scoring
    default_beta: float = 0.5
    latency_warning_ms: float = 50.0

    # Determinism
    seed: int = 42

    # Alias for compatibility with older code
    @property
    def latent_dim(self) -> int:
        """Alias for hidden_dim."""
        return self.hidden_dim

    def validate(self) -> list[str]:
        """
        Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.in_dim != 34:
            errors.append(f"Expected in_dim=34, got {self.in_dim}")

        if self.hidden_dim != 68:
            errors.append(f"Expected hidden_dim=68, got {self.hidden_dim}")

        if self.out_dim != 34:
            errors.append(f"Expected out_dim=34, got {self.out_dim}")

        if self.memory_len < 2048:
            errors.append(
                f"memory_len={self.memory_len} is below minimum 2,048"
            )

        if self.gamma != 0.0:
            errors.append(
                f"gamma={self.gamma} is not 0.0 — memory poisoning risk"
            )

        if self.k < 1:
            errors.append(f"k must be >= 1, got {self.k}")

        if self.warmup_epochs < 1:
            errors.append(f"warmup_epochs must be >= 1, got {self.warmup_epochs}")

        return errors

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "memory_len": self.memory_len,
            "memory_init_fraction": self.memory_init_fraction,
            "k": self.k,
            "gamma": self.gamma,
            "warmup_lr": self.warmup_lr,
            "warmup_epochs": self.warmup_epochs,
            "warmup_batch_size": self.warmup_batch_size,
            "warmup_noise_std": self.warmup_noise_std,
            "warmup_gradient_clip": self.warmup_gradient_clip,
            "warmup_early_stop_patience": self.warmup_early_stop_patience,
            "default_beta": self.default_beta,
            "latency_warning_ms": self.latency_warning_ms,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemStreamConfig":
        """Create config from dictionary."""
        known_fields = {
            f.name for f in cls.__dataclass_fields__.values()
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class MemStreamArchitecture:
    """
    Autoencoder architecture specification.

    Architecture: 34 -> 68 -> 34 (symmetric)
    - Encoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU
    - Decoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU

    NOTE: ReLU activation (not Tanh) per benchmark_v10.py line 612
    """

    input_dim: int = 34
    hidden_dim: int = 68
    output_dim: int = 34
    activation: str = "relu"  # Must be ReLU per v10 benchmark

    encoder_layers: list[int] = field(default_factory=lambda: [34, 68, 34])
    decoder_layers: list[int] = field(default_factory=lambda: [34, 68, 34])

    def __post_init__(self):
        """Validate architecture."""
        if self.encoder_layers != [34, 68, 34]:
            raise ValueError(
                f"Encoder must be [34, 68, 34], got {self.encoder_layers}"
            )
        if self.decoder_layers != [34, 68, 34]:
            raise ValueError(
                f"Decoder must be [34, 68, 34], got {self.decoder_layers}"
            )
        if self.activation.lower() != "relu":
            raise ValueError(
                f"Activation must be ReLU per benchmark v10, got {self.activation}"
            )


class MemStreamDefaults:
    """
    Default hyperparameter constants.

    These values are verified against benchmark v10.
    """

    # Architecture
    IN_DIM: int = 34
    HIDDEN_DIM: int = 68
    OUT_DIM: int = 34

    # Memory
    MEMORY_LEN: int = 2048
    MEMORY_INIT_FRACTION: float = 0.1

    # kNN
    K: int = 10
    GAMMA: float = 0.0

    # Warmup
    WARMUP_LR: float = 1e-3
    WARMUP_EPOCHS: int = 500
    WARMUP_BATCH_SIZE: int = 256
    WARMUP_NOISE_STD: float = 0.1
    WARMUP_GRADIENT_CLIP: float = 1.0
    WARMUP_EARLY_STOP_PATIENCE: int = 20

    # Scoring
    DEFAULT_BETA: float = 0.5
    LATENCY_WARNING_MS: float = 50.0

    # Determinism
    SEED: int = 42

    @classmethod
    def as_config(cls) -> MemStreamConfig:
        """Create a MemStreamConfig with all defaults."""
        return MemStreamConfig(
            in_dim=cls.IN_DIM,
            hidden_dim=cls.HIDDEN_DIM,
            out_dim=cls.OUT_DIM,
            memory_len=cls.MEMORY_LEN,
            memory_init_fraction=cls.MEMORY_INIT_FRACTION,
            k=cls.K,
            gamma=cls.GAMMA,
            warmup_lr=cls.WARMUP_LR,
            warmup_epochs=cls.WARMUP_EPOCHS,
            warmup_batch_size=cls.WARMUP_BATCH_SIZE,
            warmup_noise_std=cls.WARMUP_NOISE_STD,
            warmup_gradient_clip=cls.WARMUP_GRADIENT_CLIP,
            warmup_early_stop_patience=cls.WARMUP_EARLY_STOP_PATIENCE,
            default_beta=cls.DEFAULT_BETA,
            latency_warning_ms=cls.LATENCY_WARNING_MS,
            seed=cls.SEED,
        )


# Convenience function
def create_memstream_config(**overrides) -> MemStreamConfig:
    """
    Create a MemStreamConfig with optional overrides.

    Args:
        **overrides: Configuration fields to override

    Returns:
        MemStreamConfig instance
    """
    config = MemStreamDefaults.as_config()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
