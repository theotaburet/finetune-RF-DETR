"""RF-DETR Fine-tuning Pipeline.

This package provides a modular pipeline for fine-tuning RF-DETR models.

"""

__version__ = "0.1.0"


from rf_detr_finetuning.audio_chunking import (
    AudioChunk,
    AudioChunker,
    ChunkBbox,
    ChunkConfig,
    TimeBasedFFTConfig,
    draw_bboxes_on_spectrogram,
    load_chunking_config_from_yaml,
)
from rf_detr_finetuning.audio_to_coco import (
    AudioMetadata,
    BboxFrequencyInfo,
    BoundingBox,
    CategoryRegistry,
    COCODataset,
    FrequencyMapper,
    SpectrogramConfig,
    TimeMapper,
    convert_audio_to_coco,
    decode_bbox_to_frequency,
    infer_frequency_bins_from_data,
)
from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.finetune import finetune_model
from rf_detr_finetuning.predict import prediction

__all__ = [
    # Data conversion
    "convert_yolo_to_coco",
    "convert_audio_to_coco",
    # Audio processing
    "AudioMetadata",
    "BboxFrequencyInfo",
    "BoundingBox",
    "CategoryRegistry",
    "COCODataset",
    "FrequencyMapper",
    "SpectrogramConfig",
    "TimeMapper",
    "decode_bbox_to_frequency",
    "infer_frequency_bins_from_data",
    # Audio chunking
    "AudioChunk",
    "AudioChunker",
    "ChunkBbox",
    "ChunkConfig",
    "TimeBasedFFTConfig",
    "draw_bboxes_on_spectrogram",
    "load_chunking_config_from_yaml",
    # Training
    "finetune_model",
    "prediction",
]
