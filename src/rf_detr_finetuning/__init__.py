"""RF-DETR Fine-tuning Pipeline.

A modular pipeline for fine-tuning RF-DETR models on audio event detection.

Modules:
    dataprocessor: Audio I/O, spectrogram features, chunking, preprocessing
    dataloader: Dataset classes, collate functions, train/val/test splitting
    trainer: Training loop, checkpointing, logging
    predictor: Inference on images and audio files
    eventprocessor: Convert window detections to full-audio events

Legacy modules (for backward compatibility):
    audio_chunking: Original chunking implementation
    audio_to_coco: Original COCO conversion
    finetune: Original training wrapper
    predict: Original prediction

"""

__version__ = "0.2.0"

# =============================================================================
# New modular API (preferred)
# =============================================================================

# Data processing
# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================
# Legacy modules have been removed. Use the new modular API instead:
# - from rf_detr_finetuning.dataprocessor import AudioChunker, ChunkConfig, TimeBasedFFTConfig
# - from rf_detr_finetuning.dataloader import COCODatasetBuilder, CategoryRegistry
from rf_detr_finetuning.data import convert_yolo_to_coco

# Data loading
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
    # Preprocessing
    AGCConfig,
    AudioChunk,
    # Chunker
    AudioChunker,
    ChunkBbox,
    # Chunking
    ChunkConfig,
    DynamicRangeConfig,
    PreprocessingConfig,
    TimeBasedFFTConfig,
    align_bbox_to_chunk,
    compute_chunk_boundaries,
    # Features
    compute_mel_spectrogram,
    compute_mel_spectrogram_db,
    # Visualization
    draw_bboxes_on_spectrogram,
    extract_audio_chunk,
    flip_spectrogram,
    grayscale_to_rgb,
    # I/O
    load_audio_file,
    load_chunking_config_from_yaml,
    normalize_spectrogram,
    preprocess_audio,
    # Normalization
    resize_spectrogram,
)

# Event processing
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

# Prediction
from rf_detr_finetuning.predictor import (
    AudioPredictionResult,
    AudioPredictor,
    Detection,
    PredictionResult,
    Predictor,
    RFDETRPredictor,
    WindowPrediction,
    predict_directory,
)

# Training
from rf_detr_finetuning.trainer import (
    CheckpointManager,
    RFDETRTrainer,
    Trainer,
    TrainerConfig,
    create_rfdetr_trainer,
)

__all__ = [
    # ==========================================================================
    # New modular API
    # ==========================================================================
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
    "Trainer",
    "CheckpointManager",
    "RFDETRTrainer",
    "create_rfdetr_trainer",
    # Prediction
    "Predictor",
    "RFDETRPredictor",
    "Detection",
    "PredictionResult",
    "AudioPredictor",
    "AudioPredictionResult",
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
    # ==========================================================================
    # Deprecated (will be removed in v0.3.0)
    # ==========================================================================
    "convert_yolo_to_coco",
    "finetune_model",
    "prediction",
    "draw_bboxes_on_spectrogram",
]
