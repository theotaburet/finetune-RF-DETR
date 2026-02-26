"""COCO evaluation metrics for object detection.

Uses pycocotools to compute standard COCO metrics (mAP, mAP50, mAP75, etc.) from prediction and ground truth data.

"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def compute_coco_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_sizes: dict[int, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute COCO detection metrics from predictions and ground truths.

    Args:
        predictions: List of prediction dicts with keys:
            - image_id: int
            - category_id: int
            - bbox: [x, y, width, height] (COCO format)
            - score: float
        ground_truths: List of ground truth annotation dicts with keys:
            - id: int
            - image_id: int
            - category_id: int
            - bbox: [x, y, width, height] (COCO format)
            - area: float
            - iscrowd: int
        categories: List of category dicts with keys:
            - id: int
            - name: str
        image_sizes: Optional mapping of image_id to (width, height).
            If None, dimensions are inferred from ground truth bounding boxes.

    Returns:
        Dictionary with metrics:
            - mAP: Mean average precision (IoU=0.50:0.95)
            - mAP50: mAP at IoU=0.50
            - mAP75: mAP at IoU=0.75
            - mAP_small: mAP for small objects
            - mAP_medium: mAP for medium objects
            - mAP_large: mAP for large objects
            - mAR_1: Mean average recall (max 1 detection)
            - mAR_10: Mean average recall (max 10 detections)
            - mAR_100: Mean average recall (max 100 detections)

    """
    if not predictions or not ground_truths:
        logger.warning("Empty predictions or ground truths, returning zero metrics")
        return {
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "mAR_1": 0.0,
            "mAR_10": 0.0,
            "mAR_100": 0.0,
        }

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.error("pycocotools is required for COCO metrics. Install with: pip install pycocotools")
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    # Build a COCO-format ground truth dataset
    image_ids = set()
    for ann in ground_truths:
        image_ids.add(ann["image_id"])

    # Build image entries with actual dimensions when available
    gt_images = []
    for img_id in sorted(image_ids):
        if image_sizes and img_id in image_sizes:
            w, h = image_sizes[img_id]
        else:
            # Infer from bounding boxes: at minimum the image must contain all bboxes
            max_x = 0
            max_y = 0
            for ann in ground_truths:
                if ann["image_id"] == img_id:
                    bx, by, bw, bh = ann["bbox"]
                    max_x = max(max_x, bx + bw)
                    max_y = max(max_y, by + bh)
            # Use inferred size with a small margin, minimum 1
            w = max(int(max_x) + 1, 1)
            h = max(int(max_y) + 1, 1)
        gt_images.append({"id": img_id, "width": w, "height": h})

    # Ensure all ground truth annotations have required fields
    gt_annotations = []
    for ann in ground_truths:
        gt_ann = dict(ann)
        if "area" not in gt_ann:
            bbox = gt_ann["bbox"]
            gt_ann["area"] = bbox[2] * bbox[3]
        if "iscrowd" not in gt_ann:
            gt_ann["iscrowd"] = 0
        gt_annotations.append(gt_ann)

    gt_dataset = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": categories,
    }

    # Load ground truth via temp file (pycocotools API)
    gt_path = None
    dt_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_dataset, f)
            gt_path = f.name

        coco_gt = COCO(gt_path)

        # Load predictions
        if not predictions:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(predictions, f)
            dt_path = f.name

        coco_dt = coco_gt.loadRes(dt_path)

        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics from COCOeval stats
        stats = coco_eval.stats
        return {
            "mAP": float(stats[0]),
            "mAP50": float(stats[1]),
            "mAP75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
            "mAR_1": float(stats[6]),
            "mAR_10": float(stats[7]),
            "mAR_100": float(stats[8]),
        }
    finally:
        if gt_path is not None:
            os.unlink(gt_path)
        if dt_path is not None:
            os.unlink(dt_path)
