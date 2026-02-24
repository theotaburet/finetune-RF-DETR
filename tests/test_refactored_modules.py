"""Regression tests for the refactored module structure.

Verifies that key functionality works correctly after module reorganization. Only tests behavior not covered by
dedicated unit test files.

"""

from __future__ import annotations

import numpy as np
import pytest


class TestDataprocessor:
    """Test dataprocessor functionality not covered elsewhere."""

    def test_resize_spectrogram_no_aliasing(self):
        """Test scipy-based resize preserves content (no PIL aliasing)."""
        from rf_detr_finetuning.dataprocessor import resize_spectrogram

        spec = np.random.rand(128, 500).astype(np.float32)
        resized = resize_spectrogram(spec, target_width=640, target_height=128)

        assert resized.shape == (128, 640)
        np.testing.assert_array_almost_equal(resized[:, :500], spec)

    def test_grayscale_to_rgb(self):
        """Test grayscale to RGB conversion produces correct 3-channel output."""
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb

        gray = np.random.randint(0, 256, (128, 640), dtype=np.uint8)
        rgb = grayscale_to_rgb(gray)

        assert rgb.shape == (128, 640, 3)
        assert rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb[:, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, 1], gray)
        np.testing.assert_array_equal(rgb[:, :, 2], gray)


class TestDataloader:
    """Test dataloader components not covered elsewhere."""

    def test_split_config_validation(self):
        """Test SplitConfig validates that ratios sum to 1.0."""
        from rf_detr_finetuning.dataloader import SplitConfig

        config = SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert abs(total - 1.0) < 1e-9

        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_collate_detections(self):
        """Test collate function batches images and preserves targets."""
        import torch

        from rf_detr_finetuning.dataloader import collate_detections
        from rf_detr_finetuning.dataloader.dataset import DetectionTarget

        batch = [
            (
                torch.rand(3, 64, 64),
                DetectionTarget(
                    boxes=torch.tensor([[10, 10, 50, 50]]),
                    labels=torch.tensor([0]),
                    image_id=0,
                ),
            ),
            (
                torch.rand(3, 64, 64),
                DetectionTarget(
                    boxes=torch.tensor([[20, 20, 60, 60], [30, 30, 70, 70]]),
                    labels=torch.tensor([1, 0]),
                    image_id=1,
                ),
            ),
        ]

        images, targets = collate_detections(batch)

        assert images.shape == (2, 3, 64, 64)
        assert len(targets) == 2
        assert "boxes" in targets[0]
        assert "labels" in targets[0]
