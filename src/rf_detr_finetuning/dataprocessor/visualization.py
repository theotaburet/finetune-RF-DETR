"""Visualization utilities for audio spectrograms and bounding boxes.

This module provides functions for visualizing audio data with bounding boxes for debugging and validation purposes.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from rf_detr_finetuning.dataprocessor.chunking import ChunkBbox
    from rf_detr_finetuning.predictor.inference import Detection

logger = logging.getLogger(__name__)

# Color constants (RGB)
COLOR_GT = (0, 255, 0)  # Green for ground truth
COLOR_PRED = (255, 80, 80)  # Red for predictions
COLOR_MATCH = (80, 180, 255)  # Blue for matched predictions


def _to_rgb_uint8(spectrogram: np.ndarray) -> np.ndarray:
    """Convert spectrogram to RGB uint8 image.

    Args:
        spectrogram: Spectrogram image array (H, W) or (H, W, 3).

    Returns:
        RGB uint8 image (H, W, 3).

    """
    if spectrogram.ndim == 2:
        img = np.stack([spectrogram] * 3, axis=-1)
    else:
        img = spectrogram.copy()

    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    return img


def _draw_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a rectangle on an image array (in-place).

    Args:
        img: Image array (H, W, 3), modified in place.
        x1: Left x coordinate.
        y1: Top y coordinate.
        x2: Right x coordinate.
        y2: Bottom y coordinate.
        color: RGB color tuple.
        thickness: Line thickness in pixels.

    """
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    img[y1 : y1 + thickness, x1:x2] = color
    img[max(y2 - thickness, y1) : y2, x1:x2] = color
    img[y1:y2, x1 : x1 + thickness] = color
    img[y1:y2, max(x2 - thickness, x1) : x2] = color


def draw_bboxes_on_spectrogram(
    spectrogram: np.ndarray,
    bboxes: list[ChunkBbox],
    output_path: Path | None = None,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes on spectrogram using supervision library.

    Args:
        spectrogram: Spectrogram image array (H, W) or (H, W, 3).
        bboxes: List of ChunkBbox objects.
        output_path: Optional path to save annotated image.
        class_names: Optional mapping from category_id to category name.

    Returns:
        Annotated image as numpy array (H, W, 3).

    Example:
        >>> chunks = chunker.chunk_audio_file(audio_path, metadata_path)
        >>> for chunk in chunks:
        ...     debug_img = draw_bboxes_on_spectrogram(
        ...         chunk.spectrogram, chunk.bboxes
        ...     )
        ...     Image.fromarray(debug_img).save(f"debug_{i}.png")

    """
    try:
        import supervision as sv
    except ImportError:
        logger.warning("supervision not installed, drawing boxes manually")
        return _draw_bboxes_manual(spectrogram, bboxes, output_path)

    img = _to_rgb_uint8(spectrogram)

    # Handle empty bboxes
    if not bboxes:
        if output_path:
            Image.fromarray(img).save(output_path)
        return img

    # Prepare detection data
    xyxy = np.array([bbox.to_xyxy() for bbox in bboxes])
    class_ids = np.array([bbox.category_id for bbox in bboxes])

    # Create labels with overlap ratio
    labels = []
    for bbox in bboxes:
        name = class_names.get(bbox.category_id, bbox.category) if class_names else bbox.category
        labels.append(f"{name} ({bbox.overlap_ratio:.0%})")

    # Create detections object
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_ids,
    )

    # Annotate
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    img = box_annotator.annotate(img, detections)
    img = label_annotator.annotate(img, detections, labels)

    if output_path:
        Image.fromarray(img).save(output_path)

    return img


def draw_gt_and_predictions_on_spectrogram(
    spectrogram: np.ndarray,
    gt_bboxes: list[ChunkBbox] | None = None,
    predictions: list[Detection] | None = None,
    output_path: Path | None = None,
    class_names: dict[int, str] | None = None,
    gt_color: tuple[int, int, int] = COLOR_GT,
    pred_color: tuple[int, int, int] = COLOR_PRED,
) -> np.ndarray:
    """Draw ground truth and prediction bboxes on a spectrogram.

    GT boxes are drawn in green, predictions in red. Each box is labeled
    with its class name and confidence (for predictions) or overlap ratio
    (for GT).

    Args:
        spectrogram: Spectrogram image array (H, W) or (H, W, 3).
        gt_bboxes: List of ChunkBbox ground truth objects (green).
        predictions: List of Detection prediction objects (red).
        output_path: Optional path to save annotated image.
        class_names: Optional mapping from class_id/category_id to name.
        gt_color: RGB color for ground truth boxes.
        pred_color: RGB color for prediction boxes.

    Returns:
        Annotated image as numpy array (H, W, 3).

    """
    try:
        import supervision as sv

        return _draw_gt_pred_supervision(
            spectrogram, gt_bboxes, predictions, output_path, class_names, gt_color, pred_color, sv
        )
    except ImportError:
        logger.warning("supervision not installed, drawing boxes manually")
        return _draw_gt_pred_manual(spectrogram, gt_bboxes, predictions, output_path, class_names, gt_color, pred_color)


def _draw_gt_pred_supervision(
    spectrogram: np.ndarray,
    gt_bboxes: list[ChunkBbox] | None,
    predictions: list[Detection] | None,
    output_path: Path | None,
    class_names: dict[int, str] | None,
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
    sv: object,
) -> np.ndarray:
    """Draw GT and predictions using supervision library.

    Args:
        spectrogram: Spectrogram image array.
        gt_bboxes: Ground truth bounding boxes.
        predictions: Prediction detections.
        output_path: Optional save path.
        class_names: Optional class name mapping.
        gt_color: RGB color for GT boxes.
        pred_color: RGB color for prediction boxes.
        sv: The supervision module (passed to avoid re-import).

    Returns:
        Annotated image as numpy array (H, W, 3).

    """
    img = _to_rgb_uint8(spectrogram)

    # Draw GT boxes (green)
    if gt_bboxes:
        gt_xyxy = np.array([bbox.to_xyxy() for bbox in gt_bboxes])
        gt_class_ids = np.array([bbox.category_id for bbox in gt_bboxes])
        gt_labels = []
        for bbox in gt_bboxes:
            name = class_names.get(bbox.category_id, bbox.category) if class_names else bbox.category
            gt_labels.append(f"GT: {name}")

        gt_detections = sv.Detections(xyxy=gt_xyxy, class_id=gt_class_ids)
        gt_box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(*gt_color))
        gt_label_annotator = sv.LabelAnnotator(text_scale=0.35, text_thickness=1, color=sv.Color(*gt_color))

        img = gt_box_annotator.annotate(img, gt_detections)
        img = gt_label_annotator.annotate(img, gt_detections, gt_labels)

    # Draw prediction boxes (red)
    if predictions:
        pred_xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in predictions])
        pred_class_ids = np.array([d.class_id for d in predictions])
        pred_scores = np.array([d.score for d in predictions])
        pred_labels = []
        for det in predictions:
            name = (
                class_names.get(det.class_id, det.class_name or str(det.class_id))
                if class_names
                else (det.class_name or str(det.class_id))
            )
            pred_labels.append(f"Pred: {name} {det.score:.2f}")

        pred_detections = sv.Detections(xyxy=pred_xyxy, class_id=pred_class_ids, confidence=pred_scores)
        pred_box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(*pred_color))
        pred_label_annotator = sv.LabelAnnotator(text_scale=0.35, text_thickness=1, color=sv.Color(*pred_color))

        img = pred_box_annotator.annotate(img, pred_detections)
        img = pred_label_annotator.annotate(img, pred_detections, pred_labels)

    if output_path:
        Image.fromarray(img).save(output_path)

    return img


def _draw_gt_pred_manual(
    spectrogram: np.ndarray,
    gt_bboxes: list[ChunkBbox] | None,
    predictions: list[Detection] | None,
    output_path: Path | None,
    class_names: dict[int, str] | None,
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
) -> np.ndarray:
    """Fallback GT+prediction drawing without supervision library.

    Args:
        spectrogram: Spectrogram image array.
        gt_bboxes: Ground truth bounding boxes.
        predictions: Prediction detections.
        output_path: Optional save path.
        class_names: Optional class name mapping.
        gt_color: RGB color for GT boxes.
        pred_color: RGB color for prediction boxes.

    Returns:
        Annotated image as numpy array (H, W, 3).

    """
    img = _to_rgb_uint8(spectrogram)

    # Draw GT boxes
    if gt_bboxes:
        for bbox in gt_bboxes:
            x1, y1, x2, y2 = map(int, bbox.to_xyxy())
            _draw_rect(img, x1, y1, x2, y2, gt_color, thickness=2)

    # Draw prediction boxes
    if predictions:
        for det in predictions:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            _draw_rect(img, x1, y1, x2, y2, pred_color, thickness=2)

    if output_path:
        Image.fromarray(img).save(output_path)

    return img


def _draw_bboxes_manual(
    spectrogram: np.ndarray,
    bboxes: list[ChunkBbox],
    output_path: Path | None = None,
) -> np.ndarray:
    """Fallback bbox drawing without supervision library.

    Draws simple green rectangles when supervision is not available.

    Args:
        spectrogram: Spectrogram image array (H, W) or (H, W, 3).
        bboxes: List of ChunkBbox objects.
        output_path: Optional path to save annotated image.

    Returns:
        Annotated image as numpy array (H, W, 3).

    """
    img = _to_rgb_uint8(spectrogram)

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox.to_xyxy())
        _draw_rect(img, x1, y1, x2, y2, COLOR_GT, thickness=2)

    if output_path:
        Image.fromarray(img).save(output_path)

    return img
