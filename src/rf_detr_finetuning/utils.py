"""Shared utilities for RF-DETR fine-tuning pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_class_names(
    class_names: list[str] | None = None,
    classes_file: str | Path | None = None,
) -> list[str] | None:
    """Load class names from an explicit list or a JSON file.

    Supports multiple JSON layouts:

    * Plain list: ``["bird", "frog", ...]``
    * COCO format: ``{"categories": [{"id": 0, "name": "bird"}, ...]}``
      (categories are sorted by ``id`` so order matches model class indices)
    * Dict with key: ``{"class_names": ["bird", "frog", ...]}``

    Args:
        class_names: Explicit list of class names (takes priority).
        classes_file: Path to a JSON file containing class names.

    Returns:
        Ordered list of class names, or ``None`` if neither source is available.

    """
    if class_names:
        return class_names

    if classes_file is not None:
        classes_path = Path(classes_file)
        if classes_path.exists():
            with open(classes_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "categories" in data:
                    cats = sorted(data["categories"], key=lambda x: x["id"])
                    return [c["name"] for c in cats]
                if "class_names" in data:
                    return data["class_names"]

    return None
