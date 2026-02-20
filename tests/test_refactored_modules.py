"""Regression tests for the refactored module structure.

These tests verify that:
1. All new modules can be imported
2. Key functionality works the same as before
3. No breaking changes in the API

"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleImports:
    """Test that all new modules can be imported."""

    def test_import_dataprocessor(self):
        """Test dataprocessor module imports."""
        from rf_detr_finetuning.dataprocessor import (
            AudioChunker,
            ChunkConfig,
            TimeBasedFFTConfig,
            compute_mel_spectrogram,
            load_audio_file,
            resize_spectrogram,
        )

        assert callable(AudioChunker)
        assert callable(compute_mel_spectrogram)
        assert callable(load_audio_file)
        assert callable(resize_spectrogram)
        assert callable(ChunkConfig)
        assert callable(TimeBasedFFTConfig)

    def test_import_dataloader(self):
        """Test dataloader module imports."""
        from rf_detr_finetuning.dataloader import (
            AudioChunkDataset,
            COCODatasetBuilder,
            SplitConfig,
            collate_detections,
            convert_audio_to_coco,
            parse_frequency_bins,
        )

        assert callable(AudioChunkDataset)
        assert callable(COCODatasetBuilder)
        assert callable(SplitConfig)
        assert callable(collate_detections)
        assert callable(convert_audio_to_coco)
        assert callable(parse_frequency_bins)

    def test_import_trainer(self):
        """Test trainer module imports."""
        from rf_detr_finetuning.trainer import (
            CheckpointConfig,
            OptimizerConfig,
            SchedulerConfig,
            Trainer,
            TrainerConfig,
            TrainingState,
        )

        assert callable(Trainer)
        assert callable(TrainerConfig)
        assert callable(TrainingState)
        assert callable(OptimizerConfig)
        assert callable(SchedulerConfig)
        assert callable(CheckpointConfig)

    def test_import_predictor(self):
        """Test predictor module imports."""
        from rf_detr_finetuning.predictor import (
            AudioPredictor,
            Detection,
            PredictionResult,
            Predictor,
        )

        assert callable(Predictor)
        assert callable(Detection)
        assert callable(PredictionResult)
        assert callable(AudioPredictor)

    def test_import_eventprocessor(self):
        """Test eventprocessor module imports."""
        from rf_detr_finetuning.eventprocessor import (
            AudioEvent,
            EventList,
            EventMerger,
            EventPostProcessor,
            MergeConfig,
            nms_merge,
            temporal_merge,
        )

        assert callable(AudioEvent)
        assert callable(EventList)
        assert callable(EventMerger)
        assert callable(MergeConfig)
        assert callable(EventPostProcessor)
        assert callable(nms_merge)
        assert callable(temporal_merge)

    def test_import_main_package(self):
        """Test main package imports all modules."""
        import rf_detr_finetuning

        # Check version
        assert hasattr(rf_detr_finetuning, "__version__")
        assert rf_detr_finetuning.__version__ == "0.2.0"

        # Check new API
        assert hasattr(rf_detr_finetuning, "AudioChunker")
        assert hasattr(rf_detr_finetuning, "RFDETRTrainer")
        assert hasattr(rf_detr_finetuning, "EventPostProcessor")

        # Check legacy API still works
        assert hasattr(rf_detr_finetuning, "finetune_model")
        assert hasattr(rf_detr_finetuning, "convert_audio_to_coco")


class TestDataprocessor:
    """Test dataprocessor functionality."""

    def test_time_based_fft_config(self):
        """Test TimeBasedFFTConfig computation."""
        from rf_detr_finetuning.dataprocessor import TimeBasedFFTConfig

        config = TimeBasedFFTConfig(hop_ms=10.0, fft_ms=25.0, n_mels=128)

        # Test sample-rate independent properties
        assert config.hop_ms == 10.0
        assert config.fft_ms == 25.0
        assert config.n_mels == 128

        # Test sample-rate dependent computations
        sr = 16000
        assert config.get_hop_length(sr) == 160  # 10ms * 16000 / 1000
        assert config.get_n_fft(sr) == 400  # 25ms * 16000 / 1000

    def test_chunk_config(self):
        """Test ChunkConfig creation."""
        from rf_detr_finetuning.dataprocessor import ChunkConfig

        config = ChunkConfig(
            window_duration_ms=6400.0,
            target_width=640,
            target_height=640,
            min_chunk_content_ratio=0.5,
        )

        assert config.target_width == 640
        assert config.target_height == 640
        assert config.window_duration_ms == 6400.0

    def test_resize_spectrogram_no_aliasing(self):
        """Test scipy-based resize (no PIL aliasing)."""
        from rf_detr_finetuning.dataprocessor import resize_spectrogram

        # Create test spectrogram
        spec = np.random.rand(128, 500).astype(np.float32)

        # Resize to target
        resized = resize_spectrogram(spec, target_width=640, target_height=128)

        # Should pad width, not interpolate
        assert resized.shape == (128, 640)

        # Original content should be preserved (no aliasing from interpolation)
        # Width is padded, so first 500 columns should match
        np.testing.assert_array_almost_equal(resized[:, :500], spec)

    def test_grayscale_to_rgb(self):
        """Test grayscale to RGB conversion."""
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb

        gray = np.random.randint(0, 256, (128, 640), dtype=np.uint8)
        rgb = grayscale_to_rgb(gray)

        assert rgb.shape == (128, 640, 3)
        assert rgb.dtype == np.uint8

        # All channels should be equal for grayscale input
        np.testing.assert_array_equal(rgb[:, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, 1], gray)
        np.testing.assert_array_equal(rgb[:, :, 2], gray)


class TestEventProcessor:
    """Test event processing functionality."""

    def test_audio_event_creation(self):
        """Test AudioEvent dataclass."""
        from rf_detr_finetuning.eventprocessor import AudioEvent

        event = AudioEvent(
            start_ms=1000,
            end_ms=2000,
            class_id=0,
            class_name="whale",
            score=0.95,
        )

        assert event.duration_ms == 1000
        assert event.center_ms == 1500
        assert event.score == 0.95

    def test_temporal_iou(self):
        """Test temporal IoU computation."""
        from rf_detr_finetuning.eventprocessor import AudioEvent

        event1 = AudioEvent(start_ms=0, end_ms=1000, class_id=0)
        event2 = AudioEvent(start_ms=500, end_ms=1500, class_id=0)

        iou = event1.temporal_iou(event2)

        # Intersection: 500-1000 = 500ms
        # Union: 0-1500 = 1500ms
        # IoU = 500/1500 = 0.333...
        assert abs(iou - 0.333) < 0.01

    def test_event_merging(self):
        """Test event merging."""
        from rf_detr_finetuning.eventprocessor import (
            AudioEvent,
            EventList,
            EventMerger,
            MergeConfig,
        )

        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9),
                AudioEvent(start_ms=500, end_ms=1500, class_id=0, score=0.8),
                AudioEvent(start_ms=3000, end_ms=4000, class_id=0, score=0.7),
            ]
        )

        config = MergeConfig(iou_threshold=0.3, merge_strategy="max")
        merger = EventMerger(config)
        merged = merger.merge(events)

        # First two should merge, third stays separate
        assert len(merged) == 2

        # Merged event should span full range
        sorted_events = merged.sort_by_time()
        assert sorted_events[0].start_ms == 0
        assert sorted_events[0].end_ms == 1500


class TestTrainerConfig:
    """Test trainer configuration."""

    def test_trainer_config_creation(self):
        """Test TrainerConfig creation."""
        from rf_detr_finetuning.trainer import TrainerConfig

        config = TrainerConfig(
            epochs=20,
            batch_size=16,
            device="cpu",
        )

        assert config.epochs == 20
        assert config.batch_size == 16
        assert config.val_batch_size == 16  # Defaults to batch_size

    def test_trainer_config_from_dict(self):
        """Test TrainerConfig from dictionary."""
        from rf_detr_finetuning.trainer import TrainerConfig

        config_dict = {
            "epochs": 30,
            "batch_size": 8,
            "optimizer": {
                "name": "adamw",
                "lr": 5e-5,
            },
        }

        config = TrainerConfig.from_dict(config_dict)

        assert config.epochs == 30
        assert config.optimizer.lr == 5e-5


class TestPrediction:
    """Test prediction classes."""

    def test_detection_creation(self):
        """Test Detection dataclass."""
        from rf_detr_finetuning.predictor import Detection

        det = Detection(
            bbox=[10, 20, 100, 80],
            score=0.85,
            class_id=1,
            class_name="test",
        )

        assert det.x1 == 10
        assert det.y1 == 20
        assert det.x2 == 100
        assert det.y2 == 80
        assert det.width == 90
        assert det.height == 60
        assert det.area == 5400

    def test_prediction_result_filtering(self):
        """Test PredictionResult filtering."""
        from rf_detr_finetuning.predictor import Detection, PredictionResult

        result = PredictionResult(
            detections=[
                Detection(bbox=[0, 0, 10, 10], score=0.9, class_id=0),
                Detection(bbox=[0, 0, 10, 10], score=0.5, class_id=1),
                Detection(bbox=[0, 0, 10, 10], score=0.3, class_id=0),
            ]
        )

        # Filter by score
        filtered = result.filter_by_score(0.6)
        assert len(filtered) == 1
        assert filtered.detections[0].score == 0.9

        # Filter by class
        filtered = result.filter_by_class([1])
        assert len(filtered) == 1
        assert filtered.detections[0].class_id == 1


class TestDataloader:
    """Test dataloader components."""

    def test_split_config_validation(self):
        """Test SplitConfig validates ratios."""
        from rf_detr_finetuning.dataloader import SplitConfig

        # Valid config
        config = SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert abs(total - 1.0) < 1e-9  # Use tolerance for float comparison

        # Invalid config should raise
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_collate_detections(self):
        """Test collate function."""
        import torch

        from rf_detr_finetuning.dataloader import collate_detections
        from rf_detr_finetuning.dataloader.dataset import DetectionTarget

        # Create mock batch
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


class TestLegacyCompatibility:
    """Test backward compatibility with legacy API."""

    def test_legacy_chunking_imports(self):
        """Test that chunking classes are accessible from the main package."""
        import rf_detr_finetuning

        assert hasattr(rf_detr_finetuning, "AudioChunker")
        # Verify the class is the real one, not a stub
        from rf_detr_finetuning.dataprocessor import AudioChunker

        assert rf_detr_finetuning.AudioChunker is AudioChunker

    def test_legacy_coco_conversion(self):
        """Test that COCO conversion is accessible from the main package."""
        import rf_detr_finetuning

        assert hasattr(rf_detr_finetuning, "convert_audio_to_coco")
        from rf_detr_finetuning.dataloader import convert_audio_to_coco

        assert rf_detr_finetuning.convert_audio_to_coco is convert_audio_to_coco

    def test_legacy_finetune(self):
        """Test that finetune_model is accessible from the main package."""
        import rf_detr_finetuning

        assert hasattr(rf_detr_finetuning, "finetune_model")
        assert callable(rf_detr_finetuning.finetune_model)
