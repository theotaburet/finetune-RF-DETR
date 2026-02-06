"""Prediction script for RF-DETR finetuned models."""

import contextlib
import io
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from supervision import Color

from rf_detr_finetuning.finetune import MAP_MODEL_SIZE


def prediction(
    image_path: str,
    model_size: str,
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] = None,
) -> np.ndarray:
    """Predict on an image using a pretrained or checkpoint RF-DETR model.

    Args:
        model_size: Size of the RF-DETR model to use (one of 'base', 'small', 'nano', 'large', 'medium').
        model_path: Path to the model checkpoint or pretrained model name.
        image_path: Path to the input image.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.

    """
    assert model_size.lower() in MAP_MODEL_SIZE.keys(), f"Model size must be one of {list(MAP_MODEL_SIZE.keys())}"
    logging.info(f"Loading model from {model_size}")
    ModelClass = MAP_MODEL_SIZE[model_size.lower()]

    checkpoint_class_names = None
    checkpoint_num_classes = None
    if model_path and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                bias = checkpoint["model"].get("class_embed.bias")
                if bias is not None:
                    checkpoint_num_classes = bias.shape[0] - 1
            if isinstance(checkpoint, dict) and "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
                checkpoint_class_names = checkpoint["args"].class_names
        except Exception as exc:
            logging.warning("Failed to inspect checkpoint metadata: %s", exc)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.meshgrid:.*")
        # Suppress noisy stdout from DINOv2 weight checks
        with contextlib.redirect_stdout(io.StringIO()):
            if model_path and Path(model_path).exists():
                # Assume it's a checkpoint path
                if checkpoint_num_classes is not None:
                    model = ModelClass(pretrain_weights=model_path, num_classes=checkpoint_num_classes)
                else:
                    model = ModelClass(pretrain_weights=model_path)
            else:
                model = ModelClass()
    # Optimize for inference to avoid runtime warning
    try:
        model.optimize_for_inference(compile=False)
    except Exception:
        pass

    # Perform inference
    logging.info(f"Processing image: {image_path}")

    # Load and convert image (handle grayscale spectrograms)
    image = plt.imread(image_path)

    # Convert grayscale to RGB if needed
    if image.ndim == 2:  # Grayscale
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:  # Grayscale with channel dimension
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:  # RGBA
        image = image[..., :3]

    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.meshgrid:.*")
        predictions = model.predict(image, confidence=confidence)
    logging.info(f"{predictions=}")

    # Get labels from predictions
    if not class_names:
        class_names = checkpoint_class_names or model.class_names
    if isinstance(class_names, list | tuple):
        labels = [
            class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
            for cls_id in predictions.class_id
        ]
    else:
        labels = [class_names.get(int(cls_id), str(int(cls_id))) for cls_id in predictions.class_id]

    # Prepare image for annotation (ensure BGR and contiguous for OpenCV)
    annotated_image = np.ascontiguousarray(image[:, :, ::-1])

    annotated_image = sv.BoxAnnotator().annotate(annotated_image, predictions)
    annotated_image = sv.LabelAnnotator(text_color=Color.RED).annotate(annotated_image, predictions, labels=labels)

    return annotated_image
