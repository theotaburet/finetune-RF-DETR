"""Core inference functionality.

Provides base prediction classes for running inference on images.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result.

    Attributes:
        bbox: Bounding box in XYXY format [x1, y1, x2, y2].
        score: Confidence score (0-1).
        class_id: Predicted class index.
        class_name: Predicted class name (if available).

    """

    bbox: list[float]
    score: float
    class_id: int
    class_name: str | None = None

    @property
    def x1(self) -> float:
        """Left x coordinate."""
        return self.bbox[0]

    @property
    def y1(self) -> float:
        """Top y coordinate."""
        return self.bbox[1]

    @property
    def x2(self) -> float:
        """Right x coordinate."""
        return self.bbox[2]

    @property
    def y2(self) -> float:
        """Bottom y coordinate."""
        return self.bbox[3]

    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Bounding box center (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    def to_coco(self) -> dict[str, Any]:
        """Convert to COCO annotation format."""
        return {
            "bbox": [self.x1, self.y1, self.width, self.height],
            "score": self.score,
            "category_id": self.class_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class PredictionResult:
    """Prediction result for a single image.

    Attributes:
        detections: List of detections.
        image_id: Optional image identifier.
        image_path: Optional source image path.
        metadata: Additional metadata.

    """

    detections: list[Detection] = field(default_factory=list)
    image_id: int | str | None = None
    image_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of detections."""
        return len(self.detections)

    def filter_by_score(self, min_score: float) -> PredictionResult:
        """Filter detections by minimum score.

        Args:
            min_score: Minimum confidence threshold.

        Returns:
            New PredictionResult with filtered detections.

        """
        filtered = [d for d in self.detections if d.score >= min_score]
        return PredictionResult(
            detections=filtered,
            image_id=self.image_id,
            image_path=self.image_path,
            metadata=self.metadata,
        )

    def filter_by_class(self, class_ids: list[int]) -> PredictionResult:
        """Filter detections by class.

        Args:
            class_ids: List of class IDs to keep.

        Returns:
            New PredictionResult with filtered detections.

        """
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return PredictionResult(
            detections=filtered,
            image_id=self.image_id,
            image_path=self.image_path,
            metadata=self.metadata,
        )

    def to_supervision(self) -> Any:
        """Convert to supervision Detections format.

        Returns:
            supervision.Detections object.

        """
        try:
            import supervision as sv
        except ImportError:
            raise ImportError("supervision package required: pip install supervision")

        if not self.detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in self.detections])
        confidence = np.array([d.score for d in self.detections])
        class_id = np.array([d.class_id for d in self.detections])

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )


class Predictor:
    """Base predictor class for running inference.

    Args:
        model: PyTorch model for inference.
        device: Device to run inference on.
        class_names: Optional list of class names.

    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model: PyTorch model.
            device: Device to run inference on.
            class_names: Optional list of class names.

        """
        self.model = model
        self.device = torch.device(device)
        self.class_names = class_names

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | torch.Tensor,
        confidence_threshold: float = 0.5,
    ) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) or tensor (C, H, W).
            confidence_threshold: Minimum confidence for detections.

        Returns:
            PredictionResult with detections.

        """
        # Prepare input
        if isinstance(image, np.ndarray):
            # Assume HWC format, convert to CHW tensor
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = image

        # Add batch dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        # Run inference
        outputs = self.model(tensor)

        # Parse outputs (format depends on model)
        detections = self._parse_outputs(outputs, confidence_threshold)

        return PredictionResult(detections=detections)

    def _parse_outputs(
        self,
        outputs: Any,
        confidence_threshold: float,
    ) -> list[Detection]:
        """Parse model outputs into detections.

        Override this method for specific model output formats.

        Args:
            outputs: Raw model outputs.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            List of Detection objects.

        """
        detections = []

        # Generic DETR-style output parsing
        if isinstance(outputs, dict):
            pred_logits = outputs.get("pred_logits", outputs.get("logits"))
            pred_boxes = outputs.get("pred_boxes", outputs.get("boxes"))

            if pred_logits is not None and pred_boxes is not None:
                # Apply softmax to get probabilities
                probs = pred_logits.softmax(-1)[0, :, :-1]  # Exclude no-object class
                scores, labels = probs.max(-1)

                # Filter by threshold
                keep = scores > confidence_threshold
                scores = scores[keep]
                labels = labels[keep]
                boxes = pred_boxes[0, keep]

                for score, label, box in zip(scores, labels, boxes):
                    # Convert from center format to XYXY if needed
                    cx, cy, w, h = box.tolist()
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2

                    class_name = None
                    if self.class_names and label < len(self.class_names):
                        class_name = self.class_names[label]

                    detections.append(
                        Detection(
                            bbox=[x1, y1, x2, y2],
                            score=score.item(),
                            class_id=label.item(),
                            class_name=class_name,
                        )
                    )

        return detections

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type,
        device: str = "cuda",
        class_names: list[str] | None = None,
        **model_kwargs: Any,
    ) -> Predictor:
        """Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            model_class: Model class to instantiate.
            device: Inference device.
            class_names: Optional class names.
            **model_kwargs: Arguments for model construction.

        Returns:
            Predictor instance.

        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        model = model_class(**model_kwargs)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        return cls(model=model, device=device, class_names=class_names)


class RFDETRPredictor(Predictor):
    """RF-DETR specific predictor.

    Wraps the rfdetr library's inference capabilities.

    Args:
        model_size: Model variant ('nano', 'small', 'base', 'medium', 'large').
        weights_path: Path to fine-tuned weights.
        device: Inference device.
        class_names: List of class names.

    """

    def __init__(
        self,
        model_size: str = "base",
        weights_path: str | Path | None = None,
        device: str = "cuda",
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize RF-DETR predictor.

        Note: We intentionally skip ``Predictor.__init__`` because RF-DETR
        manages its own model loading and device placement.  We only set the
        base-class attributes that downstream code may reference.

        Args:
            model_size: Model size (e.g., 'base', 'small').
            weights_path: Path to model weights.
            device: Device to run inference on.
            class_names: Optional list of class names.

        """
        self.model_size = model_size
        self.weights_path = weights_path

        # Set base-class attributes without calling super().__init__
        self.device = torch.device(device)
        self.class_names = class_names
        self.model = None  # RF-DETR model is accessed via _rfdetr_model

        # Load RF-DETR model (handles its own device placement)
        self._rfdetr_model = self._load_model()

    def _load_model(self) -> Any:
        """Load RF-DETR model with checkpoint introspection.

        Uses the canonical model class mapping from ``_get_model_classes()``
        (supports all 5 sizes: nano, small, base, medium, large).

        When a checkpoint path is provided, the checkpoint is introspected to
        extract ``num_classes`` (from the classification head bias shape) and
        ``class_names`` (from saved args metadata). This avoids shape mismatches
        when loading fine-tuned models.

        """
        from rf_detr_finetuning.trainer.rfdetr_wrapper import _get_model_classes

        model_classes = _get_model_classes()

        model_class = model_classes.get(self.model_size.lower())
        if model_class is None:
            valid = list(model_classes.keys())
            raise ValueError(f"Unknown model size: {self.model_size!r}. Choose from: {valid}")

        # Introspect checkpoint for num_classes and class_names
        num_classes = None
        checkpoint_class_names = None

        if self.weights_path and Path(self.weights_path).exists():
            try:
                checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    # Extract num_classes from classification head bias
                    model_state = checkpoint.get("model", {})
                    bias = model_state.get("class_embed.bias")
                    if bias is not None:
                        num_classes = bias.shape[0] - 1

                    # Extract class_names from saved args
                    args = checkpoint.get("args")
                    if args is not None and hasattr(args, "class_names"):
                        checkpoint_class_names = args.class_names
            except Exception as exc:
                logger.warning("Failed to inspect checkpoint metadata: %s", exc)

        # Build constructor kwargs
        kwargs: dict[str, Any] = {}
        if self.weights_path:
            kwargs["pretrain_weights"] = str(self.weights_path)
        if num_classes is not None:
            kwargs["num_classes"] = num_classes

        model = model_class(**kwargs)

        # Apply checkpoint class_names if caller didn't provide them
        if self.class_names is None and checkpoint_class_names is not None:
            self.class_names = checkpoint_class_names

        # Optimize for inference (disable training-only ops)
        try:
            model.optimize_for_inference(compile=False)
        except Exception:
            pass

        logger.info(
            "Loaded RF-DETR %s%s",
            self.model_size,
            f" (num_classes={num_classes})" if num_classes is not None else "",
        )
        return model

    def predict(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: float = 0.5,
    ) -> PredictionResult:
        """Run RF-DETR inference.

        Args:
            image: Input image (array or path).
            confidence_threshold: Minimum confidence.

        Returns:
            PredictionResult with detections.

        """
        # Use RF-DETR's built-in predict method
        results = self._rfdetr_model.predict(
            image,
            threshold=confidence_threshold,
        )

        # Convert to our format
        detections = []
        if hasattr(results, "xyxy"):
            for i in range(len(results.xyxy)):
                bbox = results.xyxy[i].tolist()
                score = results.confidence[i] if hasattr(results, "confidence") else 1.0
                class_id = results.class_id[i] if hasattr(results, "class_id") else 0

                class_name = None
                if self.class_names and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]

                detections.append(
                    Detection(
                        bbox=bbox,
                        score=float(score),
                        class_id=int(class_id),
                        class_name=class_name,
                    )
                )

        return PredictionResult(
            detections=detections,
            image_path=str(image) if isinstance(image, str | Path) else None,
        )

    def predict_batch(
        self,
        images: list[np.ndarray | str | Path],
        confidence_threshold: float = 0.5,
    ) -> list[PredictionResult]:
        """Run inference on multiple images.

        Args:
            images: List of images or paths.
            confidence_threshold: Minimum confidence.

        Returns:
            List of PredictionResults.

        """
        return [self.predict(img, confidence_threshold) for img in images]
