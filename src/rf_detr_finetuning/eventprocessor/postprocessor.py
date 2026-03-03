"""Full post-processing pipeline for window predictions.

Converts window-level detections to full-audio events with:
- Timestamp reconstruction from window coordinates
- Overlapping window handling
- Event merging and filtering

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ezakodio import hz_to_mel, mel_to_hz

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    EventMerger,
    MergeConfig,
)
from rf_detr_finetuning.predictor.audio import (
    AudioPredictionResult,
    WindowPrediction,
)

logger = logging.getLogger(__name__)


@dataclass
class PostProcessorConfig:
    """Configuration for event post-processing.

    Attributes:
        time_per_pixel_ms: Milliseconds per pixel (from FFT hop).
        n_mels: Number of mel bins (for mel-to-Hz conversion).
        min_freq_hz: Minimum frequency in spectrogram.
        max_freq_hz: Maximum frequency in spectrogram.
        use_mel_scale: Whether spectrogram uses mel-scale frequency axis.
        confidence_threshold: Minimum detection confidence.
        min_event_duration_ms: Minimum event duration to keep.
        max_event_duration_ms: Maximum event duration (None = no limit).
        merge_config: Event merging configuration.
        class_names: Mapping of class IDs to names.

    """

    time_per_pixel_ms: float = 10.0
    n_mels: int = 128
    min_freq_hz: float = 0.0
    max_freq_hz: float = 8000.0
    use_mel_scale: bool = True
    confidence_threshold: float = 0.5
    min_event_duration_ms: float = 0.0
    max_event_duration_ms: float | None = None
    merge_config: MergeConfig = field(default_factory=MergeConfig)
    class_names: dict[int, str] = field(default_factory=dict)


class EventPostProcessor:
    """Converts window predictions to full-audio events.

    Handles the full pipeline:
    1. Convert pixel coordinates to time/frequency
    2. Handle overlapping windows
    3. Merge duplicate detections
    4. Filter by confidence and duration

    Args:
        config: Post-processor configuration.

    """

    def __init__(self, config: PostProcessorConfig | None = None) -> None:
        """Initialize post-processor.

        Args:
            config: Post-processor configuration. Uses defaults if None.

        """
        self.config = config or PostProcessorConfig()
        self.merger = EventMerger(self.config.merge_config)

    def process(
        self,
        prediction: AudioPredictionResult,
    ) -> EventList:
        """Process full audio prediction to events.

        Args:
            prediction: AudioPredictionResult from predictor.

        Returns:
            EventList with merged events.

        """
        # Step 1: Convert all window predictions to raw events
        raw_events = []
        for wp in prediction.window_predictions:
            events = self._window_to_events(wp)
            raw_events.extend(events)

        logger.debug(f"Extracted {len(raw_events)} raw events from windows")

        # Create initial event list
        event_list = EventList(
            events=raw_events,
            audio_path=prediction.audio_path,
            duration_ms=prediction.duration_ms,
            class_names=self.config.class_names,
        )

        # Step 2: Filter by confidence
        event_list = event_list.filter_by_score(self.config.confidence_threshold)
        logger.debug(f"After confidence filter: {len(event_list)} events")

        # Step 3: Merge overlapping events
        event_list = self.merger.merge(event_list)
        logger.debug(f"After merging: {len(event_list)} events")

        # Step 4: Filter by duration
        if self.config.min_event_duration_ms > 0 or self.config.max_event_duration_ms:
            event_list = event_list.filter_by_duration(
                min_duration_ms=self.config.min_event_duration_ms,
                max_duration_ms=self.config.max_event_duration_ms,
            )
            logger.debug(f"After duration filter: {len(event_list)} events")

        # Sort by time
        event_list = event_list.sort_by_time()

        logger.info(f"Final: {len(event_list)} events detected")
        return event_list

    def _pixel_to_freq_hz(self, pixel_y: float) -> float:
        """Convert a pixel Y-coordinate to frequency in Hz.

        After flipud, y=0 is max_freq and y=n_mels is min_freq.
        If use_mel_scale is True, converts through mel scale for accurate mapping.

        Args:
            pixel_y: Y pixel coordinate (0 = top = high freq).

        Returns:
            Frequency in Hz.

        """
        n_mels = self.config.n_mels
        min_freq = self.config.min_freq_hz
        max_freq = self.config.max_freq_hz

        if self.config.use_mel_scale:
            # Convert frequency bounds to mel
            min_mel = hz_to_mel(min_freq)
            max_mel = hz_to_mel(max_freq)
            # Interpolate in mel space (y=0 is top = max_freq = max_mel)
            mel = max_mel - (pixel_y / n_mels) * (max_mel - min_mel)
            # Convert mel back to Hz
            return mel_to_hz(mel)
        else:
            # Linear mapping
            return max_freq - (pixel_y / n_mels) * (max_freq - min_freq)

    def _window_to_events(
        self,
        window: WindowPrediction,
    ) -> list[AudioEvent]:
        """Convert window prediction to events.

        Args:
            window: Single window prediction.

        Returns:
            List of AudioEvents.

        """
        events = []

        for det in window.detections:
            # Convert X coordinates to absolute time
            # det.bbox is [x1, y1, x2, y2] in pixels
            x1_ms = window.start_ms + det.x1 * self.config.time_per_pixel_ms
            x2_ms = window.start_ms + det.x2 * self.config.time_per_pixel_ms

            # Convert Y coordinates to frequency (if n_mels is configured)
            # Note: Y is inverted in flipped spectrograms (y=0 is high freq)
            min_freq_hz = None
            max_freq_hz = None

            if self.config.n_mels > 0:
                # After flipud: y=0 is max_freq, y=height is min_freq
                # det.y1 is top (high freq), det.y2 is bottom (low freq)
                max_freq_hz = self._pixel_to_freq_hz(det.y1)
                min_freq_hz = self._pixel_to_freq_hz(det.y2)

            # Get class name
            class_name = det.class_name
            if not class_name and det.class_id in self.config.class_names:
                class_name = self.config.class_names[det.class_id]

            event = AudioEvent(
                start_ms=x1_ms,
                end_ms=x2_ms,
                class_id=det.class_id,
                class_name=class_name,
                score=det.score,
                min_freq_hz=min_freq_hz,
                max_freq_hz=max_freq_hz,
                source_windows=[window.window_index],
            )
            events.append(event)

        return events

    @classmethod
    def from_fft_config(
        cls,
        fft_config: Any,  # TimeBasedFFTConfig
        sample_rate: int,
        class_names: dict[int, str] | None = None,
        merge_config: MergeConfig | None = None,
    ) -> EventPostProcessor:
        """Create post-processor from FFT config.

        Args:
            fft_config: TimeBasedFFTConfig instance.
            sample_rate: Audio sample rate.
            class_names: Optional class name mapping.
            merge_config: Optional merge configuration.

        Returns:
            Configured EventPostProcessor.

        """
        # Compute time per pixel from FFT config
        time_per_pixel_ms = fft_config.hop_ms

        # Frequency mapping via mel scale
        n_mels = fft_config.n_mels
        max_freq_hz = sample_rate / 2  # Nyquist

        config = PostProcessorConfig(
            time_per_pixel_ms=time_per_pixel_ms,
            n_mels=n_mels,
            min_freq_hz=0.0,
            max_freq_hz=max_freq_hz,
            use_mel_scale=True,
            merge_config=merge_config or MergeConfig(),
            class_names=class_names or {},
        )

        return cls(config)


def windows_to_events(
    windows: list[WindowPrediction],
    time_per_pixel_ms: float,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> EventList:
    """Convenience function to convert windows to events.

    Args:
        windows: List of WindowPrediction objects.
        time_per_pixel_ms: Time resolution (ms per pixel).
        confidence_threshold: Minimum confidence.
        iou_threshold: IoU threshold for merging.
        class_names: Optional class names.

    Returns:
        Merged EventList.

    """
    config = PostProcessorConfig(
        time_per_pixel_ms=time_per_pixel_ms,
        confidence_threshold=confidence_threshold,
        merge_config=MergeConfig(iou_threshold=iou_threshold),
        class_names=class_names or {},
    )

    processor = EventPostProcessor(config)

    # Create a minimal AudioPredictionResult
    from rf_detr_finetuning.predictor.audio import AudioPredictionResult

    prediction = AudioPredictionResult(
        window_predictions=windows,
    )

    return processor.process(prediction)
