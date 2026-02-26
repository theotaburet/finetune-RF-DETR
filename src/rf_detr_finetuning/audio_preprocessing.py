"""Audio preprocessing utilities for spectrogram generation.

.. deprecated::
    This module is a backward-compatibility shim. All functionality has been
    consolidated into :mod:`rf_detr_finetuning.dataprocessor.preprocessing`.
    Import directly from there instead.

"""

from __future__ import annotations

from rf_detr_finetuning.dataprocessor.preprocessing import (
    AGCConfig,
    AudioPreprocessor,
    DynamicRangeConfig,
    PreprocessingConfig,
    apply_agc,
    apply_detrend,
    apply_dynamic_range_compression,
    apply_preemphasis,
    compute_percentile_rms_db,
    normalize_spectrogram,
    preprocess_audio,
    preprocess_for_inference,
)

__all__ = [
    "AGCConfig",
    "AudioPreprocessor",
    "DynamicRangeConfig",
    "PreprocessingConfig",
    "apply_agc",
    "apply_detrend",
    "apply_dynamic_range_compression",
    "apply_preemphasis",
    "compute_percentile_rms_db",
    "normalize_spectrogram",
    "preprocess_audio",
    "preprocess_for_inference",
]
