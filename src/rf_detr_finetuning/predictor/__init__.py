"""Predictor module for RF-DETR inference.

Provides inference capabilities for:
- Single images/spectrograms
- Batched inference
- Full audio file processing with window reconstruction

"""

from rf_detr_finetuning.predictor.audio import (
    AudioPredictionResult,
    AudioPredictor,
    WindowPrediction,
)
from rf_detr_finetuning.predictor.batch import (
    BatchPredictor,
    predict_dataset,
    predict_directory,
)
from rf_detr_finetuning.predictor.inference import (
    Detection,
    PredictionResult,
    Predictor,
    RFDETRPredictor,
)

__all__ = [
    # Core inference
    "Predictor",
    "RFDETRPredictor",
    "Detection",
    "PredictionResult",
    # Batch processing
    "BatchPredictor",
    "predict_directory",
    "predict_dataset",
    # Audio-specific
    "AudioPredictor",
    "AudioPredictionResult",
    "WindowPrediction",
]
