"""Shared utilities for the dataloader module."""

from __future__ import annotations

from typing import Any

NO_LABEL_SENTINEL = -1


def extract_default_label(item: Any) -> int:
    """Extract label from a dataset item using the default convention.

    Assumes the item is a tuple of (input, target) where target has a
    ``labels`` attribute. Returns the first label, or ``NO_LABEL_SENTINEL``
    if no labels exist.

    Args:
        item: A dataset item, typically ``(image, target)``.

    Returns:
        Integer label extracted from the target.

    """
    _, target = item
    if hasattr(target, "labels") and len(target.labels) > 0:
        return int(target.labels[0])
    return NO_LABEL_SENTINEL
