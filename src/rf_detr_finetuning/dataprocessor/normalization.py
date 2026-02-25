"""Spectrogram normalization and resizing module."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_spectrogram(
    spec: np.ndarray,
    target_width: int,
    target_height: int,
    order: int = 1,
) -> np.ndarray:
    """Resize spectrogram to target dimensions.

    Strategy to preserve time resolution:
    - HEIGHT (frequency axis): Interpolate - visual scaling is acceptable
    - WIDTH (time axis): Pad with zeros - preserves time resolution

    Args:
        spec: Spectrogram array (H, W) or (H, W, C)
        target_width: Target width in pixels
        target_height: Target height in pixels
        order: Kept for API compatibility (ignored; uses INTER_LINEAR)

    Returns:
        Resized/padded spectrogram as numpy array

    """
    orig_h, orig_w = spec.shape[:2]

    # Already correct size
    if orig_h == target_height and orig_w == target_width:
        return spec

    # Step 1: Resize HEIGHT to target (interpolate frequency axis)
    if orig_h != target_height:
        spec = cv2.resize(spec, (orig_w, target_height), interpolation=cv2.INTER_LINEAR)

    # Step 2: PAD WIDTH to target (preserves time resolution)
    current_h, current_w = spec.shape[:2]
    if current_w < target_width:
        if spec.ndim == 2:
            padded = np.zeros((current_h, target_width), dtype=spec.dtype)
            padded[:, :current_w] = spec
        else:
            padded = np.zeros((current_h, target_width, spec.shape[2]), dtype=spec.dtype)
            padded[:, :current_w, :] = spec
        return padded
    elif current_w > target_width:
        return spec[:, :target_width]
    else:
        return spec


def resize_spectrogram_full(
    spec: np.ndarray,
    target_width: int,
    target_height: int,
    order: int = 1,
) -> np.ndarray:
    """Resize spectrogram to target dimensions with full interpolation.

    Unlike resize_spectrogram(), this interpolates both axes.
    Use when time-stretching is acceptable (e.g., visualization).

    Args:
        spec: Spectrogram array (H, W) or (H, W, C)
        target_width: Target width
        target_height: Target height
        order: Kept for API compatibility (ignored; uses INTER_LINEAR)

    Returns:
        Resized spectrogram

    """
    orig_h, orig_w = spec.shape[:2]

    if orig_h == target_height and orig_w == target_width:
        return spec

    return cv2.resize(spec, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def spectrogram_to_image_array(
    spec: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert spectrogram to RGB colored image array (0-255, uint8).

    Args:
        spec: Spectrogram array (H, W)
        normalize: Whether to normalize to 0-255 range

    Returns:
        RGB Image array (H, W, 3) as uint8

    """
    import torch
    from ezakodio.viz import spectrogram_to_image

    spec_t = torch.from_numpy(spec.copy())
    return spectrogram_to_image(spec_t, cmap="jet", normalize=normalize)


def normalize_to_range(
    spec: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """Normalize spectrogram to specified range.

    Args:
        spec: Spectrogram array
        out_min: Output minimum value
        out_max: Output maximum value

    Returns:
        Normalized spectrogram

    """
    spec_min = spec.min()
    spec_max = spec.max()

    if spec_max > spec_min:
        normalized = (spec - spec_min) / (spec_max - spec_min)
        return normalized * (out_max - out_min) + out_min
    else:
        return np.full_like(spec, (out_min + out_max) / 2)


def grayscale_to_rgb(spec: np.ndarray) -> np.ndarray:
    """Convert grayscale spectrogram to RGB.

    Args:
        spec: Grayscale spectrogram (H, W)

    Returns:
        RGB image (H, W, 3)

    """
    if spec.ndim == 3:
        return spec
    return np.stack([spec] * 3, axis=-1)
