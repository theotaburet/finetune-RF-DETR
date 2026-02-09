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

logger = logging.getLogger(__name__)


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

    # Convert to RGB if needed
    if spectrogram.ndim == 2:
        img = np.stack([spectrogram] * 3, axis=-1)
    else:
        img = spectrogram.copy()

    # Normalize to uint8
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

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
    # Convert to RGB if needed
    if spectrogram.ndim == 2:
        img = np.stack([spectrogram] * 3, axis=-1)
    else:
        img = spectrogram.copy()

    # Normalize to uint8
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    # Draw bounding boxes
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox.to_xyxy())

        # Clamp to image bounds
        x1 = max(0, min(x1, img.shape[1] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        # Draw rectangle borders (green)
        color = (0, 255, 0)
        thickness = 2
        img[y1 : y1 + thickness, x1:x2] = color
        img[y2 - thickness : y2, x1:x2] = color
        img[y1:y2, x1 : x1 + thickness] = color
        img[y1:y2, x2 - thickness : x2] = color

    if output_path:
        Image.fromarray(img).save(output_path)

    return img
