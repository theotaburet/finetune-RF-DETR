"""Tests for visualization module."""

from __future__ import annotations

import numpy as np

from rf_detr_finetuning.dataprocessor import ChunkBbox
from rf_detr_finetuning.dataprocessor.visualization import (
    _draw_bboxes_manual,
    draw_bboxes_on_spectrogram,
)


def _make_bbox(**kwargs) -> ChunkBbox:
    """Create a ChunkBbox with sensible defaults."""
    defaults = {
        "x": 10,
        "y": 20,
        "width": 50,
        "height": 30,
        "category": "test",
        "category_id": 0,
        "original_time_start_ms": 0,
        "original_time_end_ms": 1000,
        "hz_min": 100,
        "hz_max": 500,
    }
    defaults.update(kwargs)
    return ChunkBbox(**defaults)


class TestDrawBboxesOnSpectrogram:
    """Tests for draw_bboxes_on_spectrogram function."""

    def test_draw_on_grayscale(self):
        """Test drawing bboxes on grayscale spectrogram returns RGB uint8."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        result = draw_bboxes_on_spectrogram(spec, [_make_bbox()])
        assert result.shape == (128, 256, 3)
        assert result.dtype == np.uint8

    def test_float_spectrogram_normalized(self):
        """Test that float spectrograms are normalized to uint8."""
        spec = np.random.randn(128, 256).astype(np.float32)
        result = draw_bboxes_on_spectrogram(spec, [])
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_multiple_bboxes(self):
        """Test drawing multiple bounding boxes."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        bboxes = [_make_bbox(), _make_bbox(x=100, y=50, category="test2", category_id=1)]
        result = draw_bboxes_on_spectrogram(spec, bboxes)
        assert result.shape == (128, 256, 3)

    def test_manual_clamping(self):
        """Test that bboxes extending past image bounds are clamped."""
        spec = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bbox = _make_bbox(x=90, y=90, width=50, height=50)
        result = _draw_bboxes_manual(spec, [bbox])
        assert result.shape == (100, 100, 3)
