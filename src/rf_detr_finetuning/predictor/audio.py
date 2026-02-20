"""Audio-specific prediction utilities.

Handles running inference on audio files by:
1. Chunking audio into spectrogram windows
2. Running detection on each window
3. Returning window-level predictions

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rf_detr_finetuning.predictor.inference import (
    Detection,
    Predictor,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowPrediction:
    """Prediction result for a single audio window/chunk.

    Attributes:
        detections: List of detections in this window.
        start_ms: Window start time in milliseconds.
        end_ms: Window end time in milliseconds.
        window_index: Index of this window in the audio file.
        spectrogram_shape: Shape of the spectrogram (H, W).
        is_padded: Whether this window required padding.

    """

    detections: list[Detection] = field(default_factory=list)
    start_ms: float = 0.0
    end_ms: float = 0.0
    window_index: int = 0
    spectrogram_shape: tuple[int, int] = (0, 0)
    is_padded: bool = False

    def __len__(self) -> int:
        """Return number of detections in window."""
        return len(self.detections)

    def get_time_detections(
        self,
        time_per_pixel_ms: float,
    ) -> list[dict[str, Any]]:
        """Convert pixel coordinates to time coordinates.

        Args:
            time_per_pixel_ms: Milliseconds per pixel (from FFT config).

        Returns:
            List of detections with time coordinates.

        """
        time_dets = []
        for det in self.detections:
            # Convert X coordinates to time offset within window
            x1_ms = det.x1 * time_per_pixel_ms
            x2_ms = det.x2 * time_per_pixel_ms

            time_dets.append(
                {
                    "start_ms": self.start_ms + x1_ms,
                    "end_ms": self.start_ms + x2_ms,
                    "min_freq_idx": int(det.y1),
                    "max_freq_idx": int(det.y2),
                    "score": det.score,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                }
            )

        return time_dets


@dataclass
class AudioPredictionResult:
    """Full prediction result for an audio file.

    Attributes:
        window_predictions: List of per-window predictions.
        audio_path: Path to source audio file.
        duration_ms: Total audio duration in milliseconds.
        sample_rate: Audio sample rate.
        metadata: Additional metadata.

    """

    window_predictions: list[WindowPrediction] = field(default_factory=list)
    audio_path: str | Path | None = None
    duration_ms: float = 0.0
    sample_rate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of window predictions."""
        return len(self.window_predictions)

    @property
    def total_detections(self) -> int:
        """Total number of detections across all windows."""
        return sum(len(wp) for wp in self.window_predictions)

    def get_all_detections(self) -> list[Detection]:
        """Get flat list of all detections."""
        all_dets = []
        for wp in self.window_predictions:
            all_dets.extend(wp.detections)
        return all_dets


class AudioPredictor:
    """Predictor for full audio files.

    Chunks audio into spectrogram windows and runs detection on each.
    Returns window-level predictions without merging.

    Args:
        predictor: Base image predictor.
        chunker: AudioChunker instance for processing.

    """

    def __init__(
        self,
        predictor: Predictor,
        chunker: Any,  # AudioChunker from dataprocessor
    ) -> None:
        """Initialize audio inference pipeline.

        Args:
            predictor: Image predictor instance.
            chunker: AudioChunker instance for processing.

        """
        self.predictor = predictor
        self.chunker = chunker

    def predict(
        self,
        audio_path: str | Path,
        confidence_threshold: float = 0.5,
    ) -> AudioPredictionResult:
        """Run detection on audio file.

        Args:
            audio_path: Path to audio file.
            confidence_threshold: Minimum detection confidence.

        Returns:
            AudioPredictionResult with window predictions.

        """
        audio_path = Path(audio_path)

        # Load audio
        from rf_detr_finetuning.dataprocessor import load_audio_file

        audio, sr = load_audio_file(str(audio_path))
        duration_ms = len(audio) / sr * 1000

        # Process into chunks (no events for inference)
        chunks = self.chunker.chunk_audio(audio, sr, events=[], source_uuid=audio_path.stem)

        logger.info(f"Processing {audio_path.name}: {duration_ms:.0f}ms, {len(chunks)} windows")

        # Run inference on each chunk
        window_predictions = []
        for i, chunk in enumerate(chunks):
            # Convert spectrogram to image format
            image = self._prepare_spectrogram(chunk.spectrogram)

            # Run inference
            result = self.predictor.predict(image, confidence_threshold)

            # Create window prediction
            wp = WindowPrediction(
                detections=result.detections,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                window_index=i,
                spectrogram_shape=chunk.spectrogram.shape,
                is_padded=chunk.is_padded,
            )
            window_predictions.append(wp)

        return AudioPredictionResult(
            window_predictions=window_predictions,
            audio_path=audio_path,
            duration_ms=duration_ms,
            sample_rate=sr,
        )

    def _prepare_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Prepare spectrogram for inference.

        Args:
            spectrogram: 2D spectrogram array.

        Returns:
            RGB image array (H, W, 3).

        """
        from rf_detr_finetuning.dataprocessor import (
            grayscale_to_rgb,
            normalize_to_range,
        )

        # Normalize to 0-255
        normalized = normalize_to_range(spectrogram, 0, 255)

        # Convert to RGB
        rgb = grayscale_to_rgb(normalized)

        return rgb

    def predict_batch(
        self,
        audio_paths: list[str | Path],
        confidence_threshold: float = 0.5,
        show_progress: bool = True,
    ) -> list[AudioPredictionResult]:
        """Run detection on multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            confidence_threshold: Minimum confidence.
            show_progress: Show progress bar.

        Returns:
            List of AudioPredictionResults.

        """
        from rich.progress import Progress

        results = []

        if show_progress:
            with Progress() as progress:
                task = progress.add_task(
                    "Processing audio...",
                    total=len(audio_paths),
                )
                for path in audio_paths:
                    result = self.predict(path, confidence_threshold)
                    results.append(result)
                    progress.advance(task)
        else:
            for path in audio_paths:
                result = self.predict(path, confidence_threshold)
                results.append(result)

        return results

    @classmethod
    def from_config(
        cls,
        predictor: Predictor,
        chunking_config_path: str | Path,
    ) -> AudioPredictor:
        """Create AudioPredictor from config file.

        Args:
            predictor: Base predictor instance.
            chunking_config_path: Path to chunking YAML config.

        Returns:
            Configured AudioPredictor.

        """
        from rf_detr_finetuning.dataprocessor import (
            AudioChunker,
            load_chunking_config_from_yaml,
        )

        fft_config, chunk_config, preproc_config = load_chunking_config_from_yaml(chunking_config_path)

        chunker = AudioChunker(
            fft_config=fft_config,
            chunk_config=chunk_config,
            preprocessing_config=preproc_config,
        )

        return cls(predictor=predictor, chunker=chunker)
