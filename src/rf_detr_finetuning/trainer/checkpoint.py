"""Checkpoint management for training.

Provides:
- Save/load checkpoint state
- Best model tracking
- Automatic checkpoint cleanup

"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from rf_detr_finetuning.trainer.config import CheckpointConfig
from rf_detr_finetuning.trainer.loop import TrainingState

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    state: TrainingState | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Save training checkpoint.

    Args:
        path: Output path for checkpoint file.
        model: Model to save.
        optimizer: Optimizer state (optional).
        scheduler: LR scheduler state (optional).
        state: Training state (optional).
        config: Training config dict (optional).
        extra: Additional data to save (optional).

    Returns:
        Path to saved checkpoint.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if state is not None:
        checkpoint["training_state"] = asdict(state)

    if config is not None:
        checkpoint["config"] = config

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to load tensors to.
        strict: Strict mode for model.load_state_dict().

    Returns:
        Full checkpoint dictionary.

    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Loaded model weights from {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Loaded optimizer state")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Loaded scheduler state")

    return checkpoint


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup.

    Handles:
    - Saving checkpoints at regular intervals
    - Tracking and saving best models
    - Cleaning up old checkpoints
    - Resuming from checkpoints

    Args:
        config: Checkpoint configuration.

    """

    def __init__(self, config: CheckpointConfig) -> None:
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration.

        """
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = float("inf") if config.best_mode == "min" else float("-inf")
        self.saved_checkpoints: list[Path] = []

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved this epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            True if checkpoint should be saved.

        """
        return (epoch + 1) % self.config.save_every_n_epochs == 0

    def is_best(self, metric: float) -> bool:
        """Check if current metric is best.

        Args:
            metric: Current metric value.

        Returns:
            True if this is the best metric so far.

        """
        if self.config.best_mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        state: TrainingState | None = None,
        config: dict | None = None,
        metric: float | None = None,
        epoch: int | None = None,
    ) -> Path | None:
        """Save checkpoint with automatic naming and cleanup.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            state: Training state.
            config: Training config.
            metric: Current validation metric (for best tracking).
            epoch: Current epoch number.

        Returns:
            Path to saved checkpoint, or None if not saved.

        """
        saved_path = None

        # Regular checkpoint
        if epoch is not None and self.should_save(epoch):
            checkpoint_name = f"checkpoint{epoch:04d}.pth"
            checkpoint_path = self.save_dir / checkpoint_name

            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config=config,
            )

            self.saved_checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()
            saved_path = checkpoint_path

        # Best checkpoint
        if metric is not None and self.config.save_best and self.is_best(metric):
            self.best_metric = metric
            best_path = self.save_dir / "checkpoint_best.pth"

            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config=config,
                extra={"best_metric": metric},
            )

            logger.info(f"New best model! {self.config.best_metric}: {metric:.4f}")
            saved_path = best_path

        # Always save latest
        latest_path = self.save_dir / "checkpoint.pth"
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=config,
        )

        return saved_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only last N."""
        if self.config.keep_last_n <= 0:
            return

        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_path = self.saved_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> TrainingState | None:
        """Load the latest checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.
            device: Device to load to.

        Returns:
            Training state if found, None otherwise.

        """
        latest_path = self.save_dir / "checkpoint.pth"

        if not latest_path.exists():
            logger.info("No checkpoint found to resume from")
            return None

        checkpoint = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        if "training_state" in checkpoint:
            state_dict = checkpoint["training_state"]
            state = TrainingState(**state_dict)
            logger.info(f"Resumed from epoch {state.epoch}")
            return state

        return None

    def load_best(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ) -> dict | None:
        """Load the best checkpoint (model only).

        Args:
            model: Model to load weights into.
            device: Device to load to.

        Returns:
            Checkpoint dict if found, None otherwise.

        """
        best_path = self.save_dir / "checkpoint_best.pth"

        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None

        return load_checkpoint(best_path, model=model, device=device)

    def export_model(
        self,
        model: nn.Module,
        output_path: str | Path,
        include_config: bool = True,
    ) -> Path:
        """Export model weights only (no optimizer/scheduler).

        Args:
            model: Model to export.
            output_path: Output file path.
            include_config: Include training config in export.

        Returns:
            Path to exported model.

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_dict = {
            "model_state_dict": model.state_dict(),
        }

        if include_config:
            # Try to load config from latest checkpoint
            latest = self.save_dir / "checkpoint.pth"
            if latest.exists():
                checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
                if "config" in checkpoint:
                    export_dict["config"] = checkpoint["config"]

        torch.save(export_dict, output_path)
        logger.info(f"Exported model to {output_path}")

        return output_path
