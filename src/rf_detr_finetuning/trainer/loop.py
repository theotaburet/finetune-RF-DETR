"""Training loop implementation.

Provides a generic training loop with:
- Mixed precision support
- Gradient accumulation
- Validation integration
- Callback hooks

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn
from torch.amp.autocast_mode import autocast as amp_autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from rf_detr_finetuning.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state.

    Tracks epoch, step counts, and best metrics for checkpointing.

    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class TrainerCallback(Protocol):
    """Protocol for trainer callbacks."""

    def on_train_start(self, trainer: Trainer) -> None:
        """Called at start of training."""
        ...

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at end of training."""
        ...

    def on_epoch_start(self, trainer: Trainer, epoch: int) -> None:
        """Called at start of each epoch."""
        ...

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        """Called at end of each epoch."""
        ...

    def on_batch_end(self, trainer: Trainer, batch_idx: int, loss: float) -> None:
        """Called after each batch."""
        ...


class Trainer:
    """Generic training loop for PyTorch models.

    Handles:
    - Training and validation loops
    - Mixed precision (AMP)
    - Gradient accumulation and clipping
    - Learning rate scheduling
    - Checkpointing via callbacks

    Args:
        model: PyTorch model to train.
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        optimizer: Optimizer (created if not provided).
        scheduler: LR scheduler (created if not provided).
        loss_fn: Loss function (model.forward must accept targets if None).
        callbacks: List of callbacks.

    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        loss_fn: Any | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        """Initialize training loop.

        Args:
            model: PyTorch model to train.
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            optimizer: Optional optimizer (created if None).
            scheduler: Optional learning rate scheduler.
            loss_fn: Optional loss function.
            callbacks: Optional list of training callbacks.

        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Create optimizer if not provided
        self.optimizer = optimizer or self._create_optimizer()

        # Create scheduler if not provided
        self.scheduler = scheduler or self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Training state
        self.state = TrainingState()
        # Initialize best_metric based on checkpoint config mode
        if config.checkpoint.best_mode == "max":
            self.state.best_metric = float("-inf")

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optimizer

        if opt_config.name.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")

    def _create_scheduler(self) -> _LRScheduler | None:
        """Create learning rate scheduler from config."""
        sched_config = self.config.scheduler
        # Account for gradient accumulation: effective steps per epoch = batches / accumulate
        steps_per_epoch = len(self.train_loader) // self.config.accumulate_grad_batches
        total_steps = steps_per_epoch * self.config.epochs

        if sched_config.name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=sched_config.min_lr,
            )
        elif sched_config.name.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma,
            )
        elif sched_config.name.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=sched_config.patience,
                factor=sched_config.gamma,
            )
        elif sched_config.name.lower() == "none":
            return None
        else:
            logger.warning(f"Unknown scheduler: {sched_config.name}, using none")
            return None

    def fit(self) -> TrainingState:
        """Run full training loop.

        Returns:
            Final training state.

        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.config.mixed_precision}")

        # Callbacks: training start
        for callback in self.callbacks:
            if hasattr(callback, "on_train_start"):
                callback.on_train_start(self)

        try:
            for epoch in range(self.config.epochs):
                self.state.epoch = epoch

                # Callbacks: epoch start
                for callback in self.callbacks:
                    if hasattr(callback, "on_epoch_start"):
                        callback.on_epoch_start(self, epoch)

                # Training epoch
                train_loss = self._train_epoch(epoch)
                self.state.train_loss = train_loss

                # Validation
                metrics = {"train_loss": train_loss}
                if self.val_loader is not None:
                    val_loss = self._validate_epoch(epoch)
                    self.state.val_loss = val_loss
                    metrics["val_loss"] = val_loss

                    # Track best metric using checkpoint config
                    ckpt_cfg = self.config.checkpoint
                    tracked_metric = metrics.get(ckpt_cfg.best_metric, val_loss)
                    is_better = (
                        tracked_metric < self.state.best_metric
                        if ckpt_cfg.best_mode == "min"
                        else tracked_metric > self.state.best_metric
                    )
                    if is_better:
                        self.state.best_metric = tracked_metric
                        self.state.best_epoch = epoch

                self.state.metrics = metrics

                # Callbacks: epoch end
                for callback in self.callbacks:
                    if hasattr(callback, "on_epoch_end"):
                        callback.on_epoch_end(self, epoch, metrics)

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"train_loss: {train_loss:.4f}"
                    + (f" - val_loss: {self.state.val_loss:.4f}" if self.val_loader else "")
                )

        finally:
            # Callbacks: training end
            for callback in self.callbacks:
                if hasattr(callback, "on_train_end"):
                    callback.on_train_end(self)

        return self.state

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.

        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch, batch_idx)
            total_loss += loss

            # Callbacks: batch end
            for callback in self.callbacks:
                if hasattr(callback, "on_batch_end"):
                    callback.on_batch_end(self, batch_idx, loss)

            self.state.global_step += 1
            num_batches += 1

            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] loss: {loss:.4f} lr: {lr:.2e}")

        return total_loss / max(num_batches, 1)

    def _targets_to_device(self, targets: Any) -> Any:
        """Move targets (list of dicts or dict) to the training device.

        Args:
            targets: Batch targets from the data loader.

        Returns:
            Targets with all tensors moved to self.device.

        """
        if isinstance(targets, list):
            return [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        if isinstance(targets, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        return targets

    def _train_step(self, batch: Any, batch_idx: int) -> float:
        """Single training step.

        Args:
            batch: Batch from data loader.
            batch_idx: Batch index.

        Returns:
            Batch loss value.

        """
        # Unpack batch
        images, targets = batch
        images = images.to(self.device)
        targets = self._targets_to_device(targets)

        # Forward pass with optional AMP
        if self.config.mixed_precision:
            with amp_autocast(device_type=self.device.type):
                if self.loss_fn is not None:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)
                else:
                    # Model computes loss internally
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
        else:
            if self.loss_fn is not None:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
            else:
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

        # Scale loss for gradient accumulation
        loss = loss / self.config.accumulate_grad_batches

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with accumulation)
        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            if self.config.gradient_clip_val > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # LR scheduler step (if per-step)
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

        return loss.item() * self.config.accumulate_grad_batches

    @torch.no_grad()
    def _validate_epoch(self, _epoch: int) -> float:
        """Run validation epoch.

        Args:
            _epoch: Current epoch number (unused, kept for API symmetry with ``_train_epoch``).

        Returns:
            Average validation loss.

        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images, targets = batch
            images = images.to(self.device)
            targets = self._targets_to_device(targets)

            # Forward pass
            if self.loss_fn is not None:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
            else:
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # LR scheduler step (if per-epoch / plateau)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return avg_loss
