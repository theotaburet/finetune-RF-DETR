"""RF-DETR Fine-tuning Pipeline.

A modular pipeline for fine-tuning RF-DETR models on audio event detection.

Modules:
    dataprocessor: Audio I/O, spectrogram features, chunking, preprocessing
    dataloader: Dataset classes, collate functions, train/val/test splitting
    trainer: Training loop, checkpointing, logging
    predictor: Inference on images and audio files
    eventprocessor: Convert window detections to full-audio events

"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0"

if TYPE_CHECKING:
    from rf_detr_finetuning.data import convert_yolo_to_coco
    from rf_detr_finetuning.dataloader import (
        AudioChunkDataset,
        CategoryRegistry,
        COCOAudioDataset,
        COCODatasetBuilder,
        InMemoryChunkDataset,
        SplitConfig,
        collate_detections,
        convert_audio_to_coco,
        create_train_val_test_loaders,
        parse_frequency_bins,
        split_dataset,
        validate_coco_dataset,
    )
    from rf_detr_finetuning.dataprocessor import (
        AGCConfig,
        AudioChunk,
        AudioChunker,
        ChunkBbox,
        ChunkConfig,
        DynamicRangeConfig,
        PreprocessingConfig,
        TimeBasedFFTConfig,
        align_bbox_to_chunk,
        compute_chunk_boundaries,
        compute_mel_spectrogram,
        compute_mel_spectrogram_db,
        draw_bboxes_on_spectrogram,
        extract_audio_chunk,
        flip_spectrogram,
        grayscale_to_rgb,
        load_audio_file,
        load_chunking_config_from_yaml,
        normalize_spectrogram,
        preprocess_audio,
        resize_spectrogram,
    )
    from rf_detr_finetuning.eventprocessor import (
        AudioEvent,
        EventList,
        EventMerger,
        EventPostProcessor,
        MergeConfig,
        PostProcessorConfig,
        windows_to_events,
    )
    from rf_detr_finetuning.finetune import finetune_model
    from rf_detr_finetuning.predict import prediction
    from rf_detr_finetuning.predictor import (
        AudioPredictor,
        Detection,
        PredictionResult,
        Predictor,
        RFDETRPredictor,
        WindowPrediction,
        predict_directory,
    )
    from rf_detr_finetuning.trainer import (
        CheckpointManager,
        RFDETRTrainer,
        TrainerConfig,
        create_rfdetr_trainer,
        get_model_sizes,
    )

# Lazy import mapping: attribute name -> (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Data processing
    "load_audio_file": ("rf_detr_finetuning.dataprocessor", "load_audio_file"),
    "compute_mel_spectrogram": ("rf_detr_finetuning.dataprocessor", "compute_mel_spectrogram"),
    "compute_mel_spectrogram_db": ("rf_detr_finetuning.dataprocessor", "compute_mel_spectrogram_db"),
    "flip_spectrogram": ("rf_detr_finetuning.dataprocessor", "flip_spectrogram"),
    "AGCConfig": ("rf_detr_finetuning.dataprocessor", "AGCConfig"),
    "DynamicRangeConfig": ("rf_detr_finetuning.dataprocessor", "DynamicRangeConfig"),
    "PreprocessingConfig": ("rf_detr_finetuning.dataprocessor", "PreprocessingConfig"),
    "preprocess_audio": ("rf_detr_finetuning.dataprocessor", "preprocess_audio"),
    "normalize_spectrogram": ("rf_detr_finetuning.dataprocessor", "normalize_spectrogram"),
    "ChunkConfig": ("rf_detr_finetuning.dataprocessor", "ChunkConfig"),
    "TimeBasedFFTConfig": ("rf_detr_finetuning.dataprocessor", "TimeBasedFFTConfig"),
    "AudioChunk": ("rf_detr_finetuning.dataprocessor", "AudioChunk"),
    "ChunkBbox": ("rf_detr_finetuning.dataprocessor", "ChunkBbox"),
    "compute_chunk_boundaries": ("rf_detr_finetuning.dataprocessor", "compute_chunk_boundaries"),
    "extract_audio_chunk": ("rf_detr_finetuning.dataprocessor", "extract_audio_chunk"),
    "align_bbox_to_chunk": ("rf_detr_finetuning.dataprocessor", "align_bbox_to_chunk"),
    "resize_spectrogram": ("rf_detr_finetuning.dataprocessor", "resize_spectrogram"),
    "grayscale_to_rgb": ("rf_detr_finetuning.dataprocessor", "grayscale_to_rgb"),
    "AudioChunker": ("rf_detr_finetuning.dataprocessor", "AudioChunker"),
    "load_chunking_config_from_yaml": ("rf_detr_finetuning.dataprocessor", "load_chunking_config_from_yaml"),
    "draw_bboxes_on_spectrogram": ("rf_detr_finetuning.dataprocessor", "draw_bboxes_on_spectrogram"),
    # Data loading
    "AudioChunkDataset": ("rf_detr_finetuning.dataloader", "AudioChunkDataset"),
    "COCOAudioDataset": ("rf_detr_finetuning.dataloader", "COCOAudioDataset"),
    "InMemoryChunkDataset": ("rf_detr_finetuning.dataloader", "InMemoryChunkDataset"),
    "collate_detections": ("rf_detr_finetuning.dataloader", "collate_detections"),
    "SplitConfig": ("rf_detr_finetuning.dataloader", "SplitConfig"),
    "split_dataset": ("rf_detr_finetuning.dataloader", "split_dataset"),
    "create_train_val_test_loaders": ("rf_detr_finetuning.dataloader", "create_train_val_test_loaders"),
    "validate_coco_dataset": ("rf_detr_finetuning.dataloader", "validate_coco_dataset"),
    "parse_frequency_bins": ("rf_detr_finetuning.dataloader", "parse_frequency_bins"),
    "CategoryRegistry": ("rf_detr_finetuning.dataloader", "CategoryRegistry"),
    "COCODatasetBuilder": ("rf_detr_finetuning.dataloader", "COCODatasetBuilder"),
    "convert_audio_to_coco": ("rf_detr_finetuning.dataloader", "convert_audio_to_coco"),
    # Training
    "TrainerConfig": ("rf_detr_finetuning.trainer", "TrainerConfig"),
    "CheckpointManager": ("rf_detr_finetuning.trainer", "CheckpointManager"),
    "RFDETRTrainer": ("rf_detr_finetuning.trainer", "RFDETRTrainer"),
    "create_rfdetr_trainer": ("rf_detr_finetuning.trainer", "create_rfdetr_trainer"),
    "get_model_sizes": ("rf_detr_finetuning.trainer", "get_model_sizes"),
    # Prediction
    "Predictor": ("rf_detr_finetuning.predictor", "Predictor"),
    "RFDETRPredictor": ("rf_detr_finetuning.predictor", "RFDETRPredictor"),
    "Detection": ("rf_detr_finetuning.predictor", "Detection"),
    "PredictionResult": ("rf_detr_finetuning.predictor", "PredictionResult"),
    "AudioPredictor": ("rf_detr_finetuning.predictor", "AudioPredictor"),
    "WindowPrediction": ("rf_detr_finetuning.predictor", "WindowPrediction"),
    "predict_directory": ("rf_detr_finetuning.predictor", "predict_directory"),
    # Event processing
    "AudioEvent": ("rf_detr_finetuning.eventprocessor", "AudioEvent"),
    "EventList": ("rf_detr_finetuning.eventprocessor", "EventList"),
    "EventMerger": ("rf_detr_finetuning.eventprocessor", "EventMerger"),
    "MergeConfig": ("rf_detr_finetuning.eventprocessor", "MergeConfig"),
    "EventPostProcessor": ("rf_detr_finetuning.eventprocessor", "EventPostProcessor"),
    "PostProcessorConfig": ("rf_detr_finetuning.eventprocessor", "PostProcessorConfig"),
    "windows_to_events": ("rf_detr_finetuning.eventprocessor", "windows_to_events"),
    # Legacy (deprecated, will be removed in v0.3.0)
    "convert_yolo_to_coco": ("rf_detr_finetuning.data", "convert_yolo_to_coco"),
    "finetune_model": ("rf_detr_finetuning.finetune", "finetune_model"),
    "prediction": ("rf_detr_finetuning.predict", "prediction"),
}


def __getattr__(name: str) -> object:
    """Lazy import for package-level attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Data processing
    "load_audio_file",
    "compute_mel_spectrogram",
    "compute_mel_spectrogram_db",
    "flip_spectrogram",
    "AGCConfig",
    "DynamicRangeConfig",
    "PreprocessingConfig",
    "preprocess_audio",
    "normalize_spectrogram",
    "ChunkConfig",
    "TimeBasedFFTConfig",
    "AudioChunk",
    "ChunkBbox",
    "compute_chunk_boundaries",
    "extract_audio_chunk",
    "align_bbox_to_chunk",
    "resize_spectrogram",
    "grayscale_to_rgb",
    "AudioChunker",
    "load_chunking_config_from_yaml",
    # Data loading
    "AudioChunkDataset",
    "COCOAudioDataset",
    "InMemoryChunkDataset",
    "collate_detections",
    "SplitConfig",
    "split_dataset",
    "create_train_val_test_loaders",
    "validate_coco_dataset",
    "parse_frequency_bins",
    "CategoryRegistry",
    "COCODatasetBuilder",
    "convert_audio_to_coco",
    # Training
    "TrainerConfig",
    "CheckpointManager",
    "RFDETRTrainer",
    "create_rfdetr_trainer",
    "get_model_sizes",
    # Prediction
    "Predictor",
    "RFDETRPredictor",
    "Detection",
    "PredictionResult",
    "AudioPredictor",
    "WindowPrediction",
    "predict_directory",
    # Event processing
    "AudioEvent",
    "EventList",
    "EventMerger",
    "MergeConfig",
    "EventPostProcessor",
    "PostProcessorConfig",
    "windows_to_events",
    # Deprecated (will be removed in v0.3.0)
    "convert_yolo_to_coco",
    "finetune_model",
    "prediction",
    "draw_bboxes_on_spectrogram",
]
