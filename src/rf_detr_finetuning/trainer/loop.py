"""Training state tracking.

Provides TrainingState dataclass used by checkpointing to track epoch, step counts, and best metrics during training.

"""

from __future__ import annotations

from dataclasses import dataclass, field


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
