"""Finetuning utilities for RF-DETR models.

.. deprecated:: 0.3.0
    Use :class:`rf_detr_finetuning.trainer.RFDETRTrainer` instead.
    This module is a thin compatibility shim and will be removed in a future release.

"""

import warnings

from rf_detr_finetuning.trainer.config import CheckpointConfig, OptimizerConfig, TrainerConfig
from rf_detr_finetuning.trainer.rfdetr_wrapper import (
    RFDETRConfig,
    RFDETRTrainer,
    _get_model_classes,
    get_model_sizes,
)

# Backward-compatible mapping; prefer get_model_sizes() or _get_model_classes() for new code.
MAP_MODEL_SIZE: dict = {}
try:
    MAP_MODEL_SIZE = _get_model_classes()
except ImportError:
    pass


def finetune_model(model_size: str, dataset_path: str, config: dict) -> dict:
    """Finetune an RF-DETR model on a custom dataset.

    .. deprecated:: 0.3.0
        Use :class:`rf_detr_finetuning.trainer.RFDETRTrainer` instead.

    Args:
        model_size: Size of the RF-DETR model to use (one of 'nano', 'small', 'base', 'medium', 'large').
        dataset_path: Path to the dataset directory containing images and annotations.
        config: Dictionary of training configuration parameters.

    Returns:
        Training results dictionary.

    """
    warnings.warn(
        "finetune_model() is deprecated and will be removed in v0.3.0. "
        "Use rf_detr_finetuning.trainer.RFDETRTrainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    valid_sizes = get_model_sizes()
    if model_size.lower() not in valid_sizes:
        raise ValueError(f"Model size must be one of {valid_sizes}, got {model_size!r}")

    trainer_config = TrainerConfig(
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("workers", 4),
        device=config.get("device", "auto"),
        optimizer=OptimizerConfig(lr=config.get("lr", 1e-4)),
        checkpoint=CheckpointConfig(
            save_dir=config.get("project", "output") + "/" + config.get("name", "run"),
        ),
    )

    model_config = RFDETRConfig(
        model_size=model_size,
        image_size=config.get("imgsz", 640),
    )

    trainer = RFDETRTrainer(model_config=model_config, trainer_config=trainer_config)

    # Pass through any extra keys from the config as train_kwargs
    known_keys = {"epochs", "batch_size", "workers", "lr", "project", "name", "imgsz", "device"}
    extra_kwargs = {k: v for k, v in config.items() if k not in known_keys}

    return trainer.train(dataset_path=dataset_path, suppress_warnings=True, **extra_kwargs)
