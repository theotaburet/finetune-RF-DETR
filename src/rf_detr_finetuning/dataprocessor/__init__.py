"""Data processor module for audio feature extraction and preprocessing.

This module handles all audio processing using ezakodio:
- Audio I/O (loading)
- Feature extraction (mel spectrograms)
- Preprocessing (AGC, detrend, preemphasis)
- Chunking/windowing
- Normalization
- Augmentation

"""

from rf_detr_finetuning.dataprocessor.augmentation import (
    add_noise,
    random_gain,
    random_pad_position,
    random_time_shift,
    spec_augment,
    time_shift,
)
from rf_detr_finetuning.dataprocessor.chunker import (
    AudioChunker,
    load_chunking_config_from_yaml,
)
from rf_detr_finetuning.dataprocessor.chunking import (
    AudioChunk,
    ChunkBbox,
    ChunkConfig,
    TimeBasedFFTConfig,
    align_bbox_to_chunk,
    compute_bbox_overlap,
    compute_chunk_boundaries,
    extract_audio_chunk,
    pad_audio,
)
from rf_detr_finetuning.dataprocessor.features import (
    compute_mel_spectrogram,
    compute_mel_spectrogram_db,
    flip_spectrogram,
)
from rf_detr_finetuning.dataprocessor.io import load_audio_file
from rf_detr_finetuning.dataprocessor.normalization import (
    grayscale_to_rgb,
    normalize_to_range,
    resize_spectrogram,
    resize_spectrogram_full,
    spectrogram_to_image_array,
)
from rf_detr_finetuning.dataprocessor.preprocessing import (
    AGCConfig,
    DynamicRangeConfig,
    PreprocessingConfig,
    apply_agc,
    apply_detrend,
    apply_preemphasis,
    normalize_spectrogram,
    preprocess_audio,
)
from rf_detr_finetuning.dataprocessor.visualization import (
    draw_bboxes_on_spectrogram,
)

__all__ = [
    # I/O
    "load_audio_file",
    # Features
    "compute_mel_spectrogram",
    "compute_mel_spectrogram_db",
    "flip_spectrogram",
    # Preprocessing
    "AGCConfig",
    "DynamicRangeConfig",
    "PreprocessingConfig",
    "preprocess_audio",
    "normalize_spectrogram",
    "apply_agc",
    "apply_detrend",
    "apply_preemphasis",
    # Chunking
    "ChunkConfig",
    "TimeBasedFFTConfig",
    "AudioChunk",
    "ChunkBbox",
    "compute_chunk_boundaries",
    "extract_audio_chunk",
    "align_bbox_to_chunk",
    "compute_bbox_overlap",
    "pad_audio",
    # Normalization
    "spectrogram_to_image_array",
    "resize_spectrogram",
    "resize_spectrogram_full",
    "normalize_to_range",
    "grayscale_to_rgb",
    # Chunker
    "AudioChunker",
    "load_chunking_config_from_yaml",
    # Augmentation
    "random_pad_position",
    "time_shift",
    "random_time_shift",
    "add_noise",
    "random_gain",
    "spec_augment",
    # Visualization
    "draw_bboxes_on_spectrogram",
]
