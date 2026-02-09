"""Tests for visualization module.

These tests verify that the visualization functions work correctly after migration from legacy audio_chunking module.

"""

from __future__ import annotations

import numpy as np

from rf_detr_finetuning.dataprocessor import ChunkBbox
from rf_detr_finetuning.dataprocessor.visualization import (
    _draw_bboxes_manual,
    draw_bboxes_on_spectrogram,
)


class TestDrawBboxesOnSpectrogram:
    """Tests for draw_bboxes_on_spectrogram function."""

    def test_draw_on_grayscale(self):
        """Test drawing bboxes on grayscale spectrogram."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=10,
                y=20,
                width=50,
                height=30,
                category="test",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            ),
        ]
        result = draw_bboxes_on_spectrogram(spec, bboxes)

        assert result.shape == (128, 256, 3)
        assert result.dtype == np.uint8

    def test_draw_on_rgb(self):
        """Test drawing bboxes on RGB spectrogram."""
        spec = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=10,
                y=20,
                width=50,
                height=30,
                category="test",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            ),
        ]
        result = draw_bboxes_on_spectrogram(spec, bboxes)

        assert result.shape == (128, 256, 3)
        assert result.dtype == np.uint8

    def test_draw_empty_bboxes(self):
        """Test drawing with no bboxes."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        result = draw_bboxes_on_spectrogram(spec, [])

        assert result.shape == (128, 256, 3)
        assert result.dtype == np.uint8

    def test_normalization(self):
        """Test that float spectrograms are normalized."""
        spec = np.random.randn(128, 256).astype(np.float32)
        bboxes = []
        result = draw_bboxes_on_spectrogram(spec, bboxes)

        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_multiple_bboxes(self):
        """Test drawing multiple bounding boxes."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=10,
                y=20,
                width=50,
                height=30,
                category="test1",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            ),
            ChunkBbox(
                x=100,
                y=50,
                width=40,
                height=20,
                category="test2",
                category_id=1,
                original_time_start_ms=1000,
                original_time_end_ms=2000,
                hz_min=500,
                hz_max=1000,
            ),
        ]
        result = draw_bboxes_on_spectrogram(spec, bboxes)

        assert result.shape == (128, 256, 3)


class TestDrawBboxesManual:
    """Tests for manual bbox drawing (fallback without supervision)."""

    def test_manual_draw(self):
        """Test manual bbox drawing."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=10,
                y=20,
                width=50,
                height=30,
                category="test",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            ),
        ]
        result = _draw_bboxes_manual(spec, bboxes)

        assert result.shape == (128, 256, 3)
        assert result.dtype == np.uint8

    def test_manual_empty(self):
        """Test manual drawing with no bboxes."""
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        result = _draw_bboxes_manual(spec, [])

        assert result.shape == (128, 256, 3)

    def test_manual_clamping(self):
        """Test that bboxes are clamped to image bounds."""
        spec = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=90,  # Near edge
                y=90,
                width=50,  # Would extend past edge
                height=50,
                category="test",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            ),
        ]
        result = _draw_bboxes_manual(spec, bboxes)

        assert result.shape == (100, 100, 3)
        # Should not raise error
