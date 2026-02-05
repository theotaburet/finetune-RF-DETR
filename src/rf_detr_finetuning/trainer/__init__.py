"""Trainer module for RF-DETR fine-tuning.

Provides training loop, loss handling, optimization, checkpointing, and logging.

"""

from rf_detr_finetuning.trainer.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from rf_detr_finetuning.trainer.config import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from rf_detr_finetuning.trainer.logger import (
    MetricsTracker,
    TrainingLogger,
)
from rf_detr_finetuning.trainer.loop import (
    Trainer,
    TrainingState,
)
from rf_detr_finetuning.trainer.rfdetr_wrapper import (
    RFDETRConfig,
    RFDETRTrainer,
    create_rfdetr_trainer,
)

__all__ = [
    # Config
    "TrainerConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    # Training loop
    "Trainer",
    "TrainingState",
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    # Logging
    "TrainingLogger",
    "MetricsTracker",
    # RF-DETR wrapper
    "RFDETRTrainer",
    "RFDETRConfig",
    "create_rfdetr_trainer",
]
