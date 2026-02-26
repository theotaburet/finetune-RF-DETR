"""RF-DETR specific training wrapper.

Provides a high-level interface for RF-DETR fine-tuning that wraps the rfdetr library's training capabilities.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rf_detr_finetuning.trainer.checkpoint import CheckpointManager
from rf_detr_finetuning.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class RFDETRConfig:
    """RF-DETR specific configuration.

    Attributes:
        model_size: Model variant ('small', 'base', 'large').
        pretrained_weights: Path to pretrained weights or 'coco'.
        num_classes: Number of object classes (excluding background).
        image_size: Input image size (square).
        freeze_backbone: Freeze backbone during training.
        freeze_batch_norm: Freeze batch normalization layers.

    """

    model_size: str = "base"
    pretrained_weights: str | Path | None = None
    num_classes: int = 1
    image_size: int = 640
    freeze_backbone: bool = False
    freeze_batch_norm: bool = True


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

        """
        try:
            from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall
        except ImportError:
            raise ImportError("rfdetr package not found. Install with: pip install rfdetr")

        # Select model class based on size
        model_classes = {
            "small": RFDETRSmall,
            "base": RFDETRBase,
            "large": RFDETRLarge,
        }

        model_class = model_classes.get(self.model_config.model_size.lower())
        if model_class is None:
            raise ValueError(
                f"Unknown model size: {self.model_config.model_size}. Choose from: {list(model_classes.keys())}"
            )

        # Initialize model
        model_kwargs: dict[str, Any] = {}
        if self.model_config.pretrained_weights:
            model_kwargs["pretrain_weights"] = self.model_config.pretrained_weights
        if self.model_config.num_classes != 80:  # Non-COCO class count
            model_kwargs["num_classes"] = self.model_config.num_classes

        self.model = model_class(**model_kwargs)

        # Apply freeze settings
        if self.model_config.freeze_backbone:
            self._freeze_backbone()
        if self.model_config.freeze_batch_norm:
            self._freeze_batch_norm()

        logger.info(f"Initialized RF-DETR {self.model_config.model_size} with {self.model_config.num_classes} classes")

        return self.model

    def _get_nn_module(self) -> Any | None:
        """Get the underlying nn.Module from the rfdetr wrapper.

        RFDETRBase/Large/Small are not nn.Modules themselves.
        The actual PyTorch module lives at model.model.model (LWDETR).

        Returns:
            The nn.Module, or None if not accessible.

        """
        if self.model is None:
            return None
        # RFDETRBase -> .model (Model wrapper) -> .model (LWDETR nn.Module)
        inner = getattr(self.model, "model", None)
        if inner is not None:
            inner = getattr(inner, "model", inner)
        # Verify it's actually an nn.Module
        if inner is not None and hasattr(inner, "modules") and hasattr(inner, "named_parameters"):
            return inner
        return None

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters to prevent updates during training."""
        nn_module = self._get_nn_module()
        if nn_module is None:
            logger.warning("Cannot freeze backbone: unable to access inner nn.Module")
            return
        frozen = 0
        for name, param in nn_module.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen += 1
        logger.info(f"Froze {frozen} backbone parameters")

    def _freeze_batch_norm(self) -> None:
        """Freeze batch normalization layers (set to eval mode)."""
        nn_module = self._get_nn_module()
        if nn_module is None:
            logger.warning("Cannot freeze batch norm: unable to access inner nn.Module")
            return
        import torch.nn as nn

        frozen = 0
        for module in nn_module.modules():
            if isinstance(module, nn.BatchNorm2d | nn.SyncBatchNorm):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                frozen += 1
        if frozen > 0:
            logger.info(f"Froze {frozen} batch norm layers")
        else:
            logger.debug("No batch norm layers found to freeze")

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path | None = None,
        **train_kwargs: Any,
    ) -> dict[str, Any]:
        """Run RF-DETR fine-tuning.

        Uses the rfdetr library's built-in training pipeline.

        Args:
            dataset_path: Path to dataset in RF-DETR format.
            output_dir: Output directory for checkpoints.
            **train_kwargs: Additional arguments passed to model.train().

        Returns:
            Training results dictionary.

        """
        if self.model is None:
            self.setup_model()

        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir or self.trainer_config.checkpoint.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare training arguments
        train_args = {
            "dataset_dir": str(dataset_path),
            "epochs": self.trainer_config.epochs,
            "batch_size": self.trainer_config.batch_size,
            "grad_accum_steps": self.trainer_config.accumulate_grad_batches,
            "lr": self.trainer_config.optimizer.lr,
            "output_dir": str(output_dir),
            "device": self.trainer_config.device,
            "seed": self.trainer_config.seed,
            "workers": self.trainer_config.num_workers,
        }

        # Add image size if supported
        train_args["image_size"] = self.model_config.image_size

        # Override with user kwargs
        train_args.update(train_kwargs)

        logger.info(f"Starting RF-DETR training with config: {train_args}")

        # Run training
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

    # Apply any additional config overrides from kwargs
    for key, value in kwargs.items():
        if hasattr(trainer_config, key):
            setattr(trainer_config, key, value)
        else:
            logger.warning(f"Unknown trainer config key ignored: {key}")

    model_config = RFDETRConfig(
        model_size=model_size,
        num_classes=num_classes,
        pretrained_weights=pretrained_weights,
    )

    return RFDETRTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
    )
