"""Spectrogram normalization and resizing module.

Uses scipy.ndimage for proper interpolation without aliasing artifacts.

"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def resize_spectrogram(
    spec: np.ndarray,
    target_width: int,
    target_height: int,
    order: int = 1,
) -> np.ndarray:
    """Resize spectrogram to target dimensions using scipy.ndimage.zoom.

    Strategy to preserve time resolution:
    - HEIGHT (frequency axis): Interpolate - visual scaling is acceptable
    - WIDTH (time axis): Pad with zeros - preserves time resolution

    Uses scipy.ndimage.zoom for proper interpolation without aliasing.

    Args:
        spec: Spectrogram array (H, W) or (H, W, C)
        target_width: Target width in pixels
        target_height: Target height in pixels
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)
            Default 1 (bilinear) is a good balance of quality and speed

    Returns:
        Resized/padded spectrogram as numpy array

    """
    orig_h, orig_w = spec.shape[:2]

    # Already correct size
    if orig_h == target_height and orig_w == target_width:
        return spec

    # Step 1: Resize HEIGHT to target (interpolate frequency axis)
    if orig_h != target_height:
        zoom_h = target_height / orig_h
        if spec.ndim == 2:
            spec = zoom(spec, (zoom_h, 1.0), order=order)
        else:
            # For multi-channel (H, W, C), zoom only spatial dims
            spec = zoom(spec, (zoom_h, 1.0, 1.0), order=order)

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
        # Crop if width is larger (shouldn't happen normally)
        return spec[:, :target_width]
    else:
        return spec


def spectrogram_to_image_array(
    spec: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert spectrogram to image array (0-255, uint8).

    Args:
        spec: Spectrogram array (H, W)
        normalize: Whether to normalize to 0-255 range

    Returns:
        Image array (H, W) as uint8

    """
    if normalize:
        spec = normalize_to_range(spec, 0.0, 255.0)
    else:
        spec = np.clip(spec, 0, 255)

    return spec.astype(np.uint8)


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
