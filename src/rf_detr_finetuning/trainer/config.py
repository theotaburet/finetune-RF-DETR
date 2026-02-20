"""Training configuration dataclasses.

Provides structured configuration for:
- Training hyperparameters
- Optimizer settings
- Learning rate scheduler
- Checkpointing behavior

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        name: Optimizer name ('adamw', 'sgd', 'adam').
        lr: Base learning rate.
        weight_decay: L2 regularization factor.
        momentum: Momentum for SGD (ignored for Adam variants).
        betas: Betas for Adam variants.
        eps: Epsilon for numerical stability.

    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0001
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for optimizer construction."""
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "betas": list(self.betas),
            "eps": self.eps,
        }


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        name: Scheduler name ('cosine', 'step', 'plateau', 'warmup_cosine').
        warmup_epochs: Number of warmup epochs.
        warmup_lr: Initial learning rate during warmup.
        min_lr: Minimum learning rate.
        step_size: Step size for StepLR.
        gamma: Decay factor for StepLR.
        patience: Patience for ReduceLROnPlateau.

    """

    name: str = "cosine"
    warmup_epochs: int = 0
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5


@dataclass
class CheckpointConfig:
    """Checkpoint configuration.

    Attributes:
        save_dir: Directory for saving checkpoints.
        save_every_n_epochs: Save checkpoint every N epochs.
        save_best: Save best model based on validation metric.
        best_metric: Metric to track for best model ('val_loss', 'mAP').
        best_mode: 'min' or 'max' for best metric comparison.
        keep_last_n: Keep only last N checkpoints.
        resume_from: Path to checkpoint to resume from.

    """

    save_dir: str | Path = "output"
    save_every_n_epochs: int = 10
    save_best: bool = True
    best_metric: str = "val_loss"
    best_mode: str = "min"
    keep_last_n: int = 3
    resume_from: str | Path | None = None

    def __post_init__(self) -> None:
        """Ensure save_dir is a Path."""
        self.save_dir = Path(self.save_dir)


@dataclass
class TrainerConfig:
    """Full training configuration.

    Attributes:
        epochs: Total training epochs.
        batch_size: Training batch size.
        val_batch_size: Validation batch size (defaults to batch_size).
        num_workers: DataLoader worker processes.
        device: Training device ('cuda', 'cpu', 'auto').
        mixed_precision: Use automatic mixed precision (AMP).
        gradient_clip_val: Max gradient norm for clipping (0 = no clipping).
        accumulate_grad_batches: Gradient accumulation steps.
        val_check_interval: Validation every N training steps (0 = per epoch).
        log_every_n_steps: Log metrics every N steps.
        seed: Random seed for reproducibility.
        deterministic: Use deterministic algorithms.
        optimizer: Optimizer configuration.
        scheduler: Scheduler configuration.
        checkpoint: Checkpoint configuration.

    """

    epochs: int = 10
    batch_size: int = 8
    val_batch_size: int | None = None
    num_workers: int = 4
    device: str = "auto"
    mixed_precision: bool = True
    gradient_clip_val: float = 0.1
    accumulate_grad_batches: int = 1
    val_check_interval: int = 0
    log_every_n_steps: int = 50
    seed: int = 42
    deterministic: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> TrainerConfig:
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            TrainerConfig instance.

        """
        # Extract nested configs (copy to avoid mutating caller's dict)
        config_dict = dict(config_dict)
        optimizer_dict = config_dict.pop("optimizer", {})
        scheduler_dict = config_dict.pop("scheduler", {})
        checkpoint_dict = config_dict.pop("checkpoint", {})

        optimizer = OptimizerConfig(**optimizer_dict)
        scheduler = SchedulerConfig(**scheduler_dict)
        checkpoint = CheckpointConfig(**checkpoint_dict)

        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint=checkpoint,
            **config_dict,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TrainerConfig:
        """Load config from YAML file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            TrainerConfig instance.

        """
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Handle nested 'training' key
        if "training" in config_dict:
            config_dict = config_dict["training"]

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Includes all fields so that ``from_dict(config.to_dict())`` produces an
        equivalent configuration (roundtrip-safe).

        """
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "val_check_interval": self.val_check_interval,
            "log_every_n_steps": self.log_every_n_steps,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "optimizer": self.optimizer.to_dict(),
            "scheduler": {
                "name": self.scheduler.name,
                "warmup_epochs": self.scheduler.warmup_epochs,
                "warmup_lr": self.scheduler.warmup_lr,
                "min_lr": self.scheduler.min_lr,
                "step_size": self.scheduler.step_size,
                "gamma": self.scheduler.gamma,
                "patience": self.scheduler.patience,
            },
            "checkpoint": {
                "save_dir": str(self.checkpoint.save_dir),
                "save_every_n_epochs": self.checkpoint.save_every_n_epochs,
                "save_best": self.checkpoint.save_best,
                "best_metric": self.checkpoint.best_metric,
                "best_mode": self.checkpoint.best_mode,
                "keep_last_n": self.checkpoint.keep_last_n,
                "resume_from": str(self.checkpoint.resume_from) if self.checkpoint.resume_from else None,
            },
        }
