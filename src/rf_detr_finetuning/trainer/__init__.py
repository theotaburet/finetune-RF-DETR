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
    TrainingState,
)
from rf_detr_finetuning.trainer.metrics import (
    compute_coco_metrics,
)
from rf_detr_finetuning.trainer.rfdetr_wrapper import (
    RFDETRConfig,
    RFDETRTrainer,
    create_rfdetr_trainer,
    get_model_sizes,
)

__all__ = [
    # Config
    "TrainerConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    # Training state
    "TrainingState",
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    # Logging
    "TrainingLogger",
    "MetricsTracker",
    # Metrics
    "compute_coco_metrics",
    # RF-DETR wrapper
    "RFDETRTrainer",
    "RFDETRConfig",
    "create_rfdetr_trainer",
    "get_model_sizes",
]
