"""RF-DETR specific training wrapper.

Provides a high-level interface for RF-DETR fine-tuning that wraps the rfdetr library's training capabilities.

"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rf_detr_finetuning.trainer.checkpoint import CheckpointManager
from rf_detr_finetuning.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)

# Warnings commonly emitted by rfdetr/DINOv2 during model loading and training
_RFDETR_WARNING_PATTERNS = (
    ".*positional encodings.*",
    ".*patch size.*",
    ".*meshgrid.*",
    ".*multidimensional indexing.*",
    ".*lightning.*",
)


def _get_model_classes() -> dict[str, type]:
    """Import and return the rfdetr model class mapping.

    Lazily imports from rfdetr to avoid import-time failures when
    the rfdetr package is not installed.

    Returns:
        Dictionary mapping model size names to rfdetr model classes.

    Raises:
        ImportError: If rfdetr package is not installed.

    """
    try:
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError:
        raise ImportError("rfdetr package not found. Install with: pip install rfdetr")

    return {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }


def get_model_sizes() -> list[str]:
    """Return sorted list of supported RF-DETR model size names.

    This can be used without importing rfdetr (returns hardcoded names).

    Returns:
        List of model size strings: ['base', 'large', 'medium', 'nano', 'small'].

    """
    return ["nano", "small", "base", "medium", "large"]


@dataclass
class RFDETRConfig:
    """RF-DETR specific configuration.

    Attributes:
        model_size: Model variant ('nano', 'small', 'base', 'medium', 'large').
        pretrained_weights: Path to pretrained weights or 'coco'.
        num_classes: Number of object classes (excluding background). Used for logging
            and metadata; rfdetr auto-detects this from the dataset at training time.
        image_size: Input image size used for spectrogram generation. NOT passed to
            RF-DETR training — each model variant uses its own default resolution
            (e.g. 560 for base, 384 for nano) to satisfy backbone divisibility
            constraints. RF-DETR's transforms resize input images accordingly.

    """

    model_size: str = "base"
    pretrained_weights: str | Path | None = None
    num_classes: int = 1
    image_size: int = 640


class RFDETRTrainer:
    """High-level RF-DETR fine-tuning trainer.

    Wraps the rfdetr library for easy fine-tuning on custom datasets.

    Args:
        model_config: RF-DETR model configuration.
        trainer_config: Training configuration.

    """

    def __init__(
        self,
        model_config: RFDETRConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        """Initialize RF-DETR trainer wrapper.

        Args:
            model_config: RF-DETR model configuration.
            trainer_config: Training configuration.

        """
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.model = None
        self.checkpoint_manager = CheckpointManager(trainer_config.checkpoint)

    def setup_model(self) -> Any:
        """Initialize RF-DETR model.

        Returns:
            Initialized model.

        Raises:
            ImportError: If rfdetr package is not installed.
            ValueError: If model_size is not one of the supported sizes.

        """
        model_classes = _get_model_classes()
        size = self.model_config.model_size.lower()

        model_class = model_classes.get(size)
        if model_class is None:
            raise ValueError(
                f"Unknown model size: {self.model_config.model_size!r}. Choose from: {list(model_classes.keys())}"
            )

        # Initialize model
        self.model = model_class(
            pretrain_weights=self.model_config.pretrained_weights,
        )

        logger.info(f"Initialized RF-DETR {self.model_config.model_size} with {self.model_config.num_classes} classes")

        return self.model

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path | None = None,
        suppress_warnings: bool = True,
        **train_kwargs: Any,
    ) -> dict[str, Any]:
        """Run RF-DETR fine-tuning.

        Uses the rfdetr library's built-in training pipeline.

        Args:
            dataset_path: Path to dataset in RF-DETR format.
            output_dir: Output directory for checkpoints.
            suppress_warnings: Suppress common rfdetr/DINOv2 warnings during training.
            **train_kwargs: Additional arguments passed to model.train().

        Returns:
            Training results dictionary.

        """
        if self.model is None:
            self.setup_model()

        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir or self.trainer_config.checkpoint.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare training arguments.
        # Keys must match rfdetr.config.TrainConfig fields and rfdetr.main.populate_args
        # parameters — mismatched names are silently dropped by Pydantic.
        # NOTE: We intentionally do NOT pass "resolution" here. Each RF-DETR model
        # variant has a default resolution that satisfies its backbone's divisibility
        # constraint (e.g. base=560 for patch_size=14, num_windows=4 → block_size=56).
        # Passing resolution=640 would crash the base model (640 % 56 != 0).
        # RF-DETR's transforms resize our input images to the model's native resolution.
        train_args = {
            "dataset_dir": str(dataset_path),
            "coco_path": str(dataset_path),
            "epochs": self.trainer_config.epochs,
            "batch_size": self.trainer_config.batch_size,
            "grad_accum_steps": self.trainer_config.accumulate_grad_batches,
            "lr": self.trainer_config.optimizer.lr,
            "output_dir": str(output_dir),
            "device": self.trainer_config.device,
            "seed": self.trainer_config.seed,
            "num_workers": self.trainer_config.num_workers,
        }

        # Override with user kwargs
        train_args.update(train_kwargs)

        logger.info(f"Starting RF-DETR training with config: {train_args}")

        # Optionally suppress noisy warnings from rfdetr/DINOv2
        if suppress_warnings:
            for pattern in _RFDETR_WARNING_PATTERNS:
                warnings.filterwarnings("ignore", message=pattern)

        try:
            results = self.model.train(**train_args)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        logger.info("Training complete!")

        return results or {}

    def load_weights(self, checkpoint_path: str | Path) -> None:
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        """
        if self.model is None:
            self.setup_model()

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {checkpoint_path}")

    def export(
        self,
        output_path: str | Path,
        format: str = "pytorch",
    ) -> Path:
        """Export trained model.

        Args:
            output_path: Output file path.
            format: Export format ('pytorch', 'onnx', 'torchscript').

        Returns:
            Path to exported model.

        """
        if self.model is None:
            raise RuntimeError("No model to export. Call setup_model() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "pytorch":
            torch.save(self.model.state_dict(), output_path)
        elif format == "torchscript":
            scripted = torch.jit.script(self.model)
            scripted.save(str(output_path))
        elif format == "onnx":
            # ONNX export would require dummy input
            raise NotImplementedError("ONNX export not yet implemented")
        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Exported model to {output_path}")
        return output_path


def create_rfdetr_trainer(
    config_path: str | Path | None = None,
    model_size: str = "base",
    num_classes: int = 1,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    output_dir: str | Path = "output",
    pretrained_weights: str | Path | None = None,
    **kwargs: Any,
) -> RFDETRTrainer:
    """Factory function to create RF-DETR trainer.

    Args:
        config_path: Optional path to YAML config file.
        model_size: Model variant.
        num_classes: Number of classes.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        output_dir: Output directory.
        pretrained_weights: Path to pretrained weights or None.
        **kwargs: Additional config overrides.

    Returns:
        Configured RFDETRTrainer instance.

    """
    if config_path is not None:
        trainer_config = TrainerConfig.from_yaml(config_path)
    else:
        from rf_detr_finetuning.trainer.config import (
            CheckpointConfig,
            OptimizerConfig,
        )

        trainer_config = TrainerConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=OptimizerConfig(lr=lr),
            checkpoint=CheckpointConfig(save_dir=output_dir),
        )

    model_config = RFDETRConfig(
        model_size=model_size,
        num_classes=num_classes,
        pretrained_weights=pretrained_weights,
    )

    return RFDETRTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
    )
