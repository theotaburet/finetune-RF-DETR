"""Collate functions for batching detection data.

Provides custom collate functions that handle variable numbers of detections per image while maintaining proper batching
for DETR-style models.

"""

from __future__ import annotations

from typing import Any

import torch

from rf_detr_finetuning.dataloader.dataset import DetectionTarget


def collate_detections(
    batch: list[tuple[torch.Tensor, DetectionTarget]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Collate function for detection batches.

    Stacks images into a batch tensor while keeping targets as a list
    of dictionaries (required by DETR-style loss functions).

    Args:
        batch: List of (image, target) tuples from dataset.

    Returns:
        Tuple of (batched_images, list_of_target_dicts).

    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target.to_dict())

    # Stack images: (B, C, H, W)
    batched_images = torch.stack(images, dim=0)

    return batched_images, targets


def collate_with_targets(
    batch: list[tuple[torch.Tensor, DetectionTarget]],
    pad_to_max: bool = False,
    max_detections: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Collate with padded target tensors.

    Creates batched tensors for both images and targets by padding
    targets to the maximum number of detections in the batch.

    Args:
        batch: List of (image, target) tuples.
        pad_to_max: If True, pad all targets to max_detections.
        max_detections: Maximum number of detections to pad to.
            If None and pad_to_max is True, uses max in batch.

    Returns:
        Tuple of (batched_images, batched_targets_dict).
        batched_targets_dict contains:
            - boxes: (B, N, 4) padded boxes
            - labels: (B, N) padded labels (-1 for padding)
            - num_boxes: (B,) actual number of boxes per image
            - image_ids: (B,) image IDs

    """
    images = []
    all_boxes = []
    all_labels = []
    all_image_ids = []
    num_boxes = []

    for image, target in batch:
        images.append(image)
        all_boxes.append(target.boxes)
        all_labels.append(target.labels)
        all_image_ids.append(target.image_id)
        num_boxes.append(len(target.boxes))

    # Stack images
    batched_images = torch.stack(images, dim=0)

    # Determine padding size
    max_boxes = max(num_boxes) if num_boxes else 0
    if pad_to_max and max_detections is not None:
        max_boxes = max(max_boxes, max_detections)

    # Pad boxes and labels
    batch_size = len(batch)
    padded_boxes = torch.zeros(batch_size, max_boxes, 4)
    padded_labels = torch.full((batch_size, max_boxes), -1, dtype=torch.int64)

    for i, (boxes, labels) in enumerate(zip(all_boxes, all_labels)):
        n = len(boxes)
        if n > 0:
            padded_boxes[i, :n] = boxes
            padded_labels[i, :n] = labels

    batched_targets = {
        "boxes": padded_boxes,
        "labels": padded_labels,
        "num_boxes": torch.tensor(num_boxes, dtype=torch.int64),
        "image_ids": torch.tensor(all_image_ids, dtype=torch.int64),
    }

    return batched_images, batched_targets


def collate_for_inference(
    batch: list[torch.Tensor],
) -> torch.Tensor:
    """Simple collate for inference (images only).

    Args:
        batch: List of image tensors.

    Returns:
        Batched image tensor (B, C, H, W).

    """
    return torch.stack(batch, dim=0)


def create_collate_fn(
    mode: str = "list",
    **kwargs: Any,
) -> Any:
    """Factory function to create appropriate collate function.

    Args:
        mode: Collate mode:
            - "list": Returns targets as list of dicts (DETR-style)
            - "padded": Returns padded target tensors
            - "inference": Returns only images
        **kwargs: Additional arguments passed to collate function.

    Returns:
        Collate function.

    """
    if mode == "list":
        return collate_detections
    elif mode == "padded":
        return lambda batch: collate_with_targets(batch, **kwargs)
    elif mode == "inference":
        return collate_for_inference
    else:
        raise ValueError(f"Unknown collate mode: {mode}")
