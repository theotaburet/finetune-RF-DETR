"""Audio chunking module for sliding window spectrogram generation.

This module handles:
- Time-based FFT configuration (sample-rate independent)
- Sliding window chunking with configurable overlap
- Bbox alignment to chunk coordinates
- Padding for short audio

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from ezakodio import hz_to_mel

logger = logging.getLogger(__name__)


@dataclass
class TimeBasedFFTConfig:
    """Time-based FFT configuration for consistent resolution across sample rates.

    Uses milliseconds instead of samples for sample-rate independence.

    Attributes:
        fft_ms: FFT window duration in milliseconds
        hop_ms: Hop duration in milliseconds (time per pixel)
        n_mels: Number of mel bands

    """

    fft_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 128

    def get_n_fft(self, sample_rate: int) -> int:
        """Compute n_fft for given sample rate."""
        n_fft = int(sample_rate * self.fft_ms / 1000)
        return max(n_fft, 1)

    def get_hop_length(self, sample_rate: int) -> int:
        """Compute hop_length for given sample rate."""
        hop = int(sample_rate * self.hop_ms / 1000)
        return max(hop, 1)

    def get_time_per_pixel(self) -> float:
        """Return time in ms per spectrogram pixel (width axis)."""
        return self.hop_ms

    def get_freq_resolution(self, sample_rate: int) -> float:
        """Return frequency resolution in Hz per FFT bin."""
        n_fft = self.get_n_fft(sample_rate)
        return sample_rate / n_fft


@dataclass
class ChunkConfig:
    """Configuration for audio chunking.

    Attributes:
        window_duration_ms: Duration of each chunk in milliseconds
        overlap_ratio: Overlap between chunks as ratio (0.0-1.0)
        min_overlap_with_event_ratio: Minimum overlap for event inclusion
        target_width: Target image width in pixels
        target_height: Target image height in pixels
        padding_mode: Padding mode ('zero', 'repeat', 'reflect')
        min_chunk_content_ratio: Minimum content ratio to keep chunk
        random_pad_position: Randomize padding position for augmentation

    """

    window_duration_ms: float = 5000.0
    overlap_ratio: float = 0.2
    min_overlap_with_event_ratio: float = 0.3
    target_width: int | None = 640
    target_height: int | None = 640
    padding_mode: str = "zero"
    min_chunk_content_ratio: float = 0.5
    random_pad_position: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.overlap_ratio < 0 or self.overlap_ratio >= 1:
            raise ValueError("overlap_ratio must be between 0 and 1 (exclusive)")
        if self.min_overlap_with_event_ratio < 0 or self.min_overlap_with_event_ratio > 1:
            raise ValueError("min_overlap_with_event_ratio must be between 0 and 1")
        if self.padding_mode not in ("zero", "repeat", "reflect"):
            raise ValueError(f"Invalid padding_mode: {self.padding_mode}")

    def get_overlap_ms(self) -> float:
        """Compute overlap in milliseconds from ratio."""
        return self.window_duration_ms * self.overlap_ratio


@dataclass
class ChunkBbox:
    """Bounding box within a chunk.

    Attributes:
        x: X coordinate (pixels) relative to chunk
        y: Y coordinate (pixels) relative to chunk
        width: Width in pixels
        height: Height in pixels
        category: Category name
        category_id: Category ID
        original_time_start_ms: Original event start time
        original_time_end_ms: Original event end time
        hz_min: Minimum frequency in Hz
        hz_max: Maximum frequency in Hz
        overlap_ratio: Event overlap with this chunk (0-1)

    """

    x: float
    y: float
    width: float
    height: float
    category: str
    category_id: int
    original_time_start_ms: float
    original_time_end_ms: float
    hz_min: float
    hz_max: float
    overlap_ratio: float = 1.0

    def to_coco_bbox(self) -> list[float]:
        """Return COCO format bbox [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class ChunkStats:
    """Per-chunk spectrogram statistics for AGC calibration.

    Attributes:
        spec_min: Minimum value in raw spectrogram (dB).
        spec_max: Maximum value in raw spectrogram (dB).
        spec_mean: Mean value in raw spectrogram (dB).
        spec_std: Standard deviation in raw spectrogram (dB).
        dynamic_range_db: Dynamic range (max - min) in dB.
        saturation_ratio: Fraction of pixels at 0 or 255 after normalization.
        padding_ratio: Fraction of chunk that is zero-padding.

    """

    spec_min: float = 0.0
    spec_max: float = 0.0
    spec_mean: float = 0.0
    spec_std: float = 0.0
    dynamic_range_db: float = 0.0
    saturation_ratio: float = 0.0
    padding_ratio: float = 0.0


@dataclass
class AudioChunk:
    """A chunk of audio with metadata and bboxes.

    Attributes:
        chunk_index: Index of this chunk
        start_ms: Start time in original audio (ms)
        end_ms: End time in original audio (ms)
        spectrogram: Spectrogram array for this chunk
        bboxes: List of bounding boxes
        source_uuid: Source file UUID
        sample_rate: Sample rate used
        is_padded: Whether chunk required padding
        padding_amount_ms: Amount of padding added (ms)
        stats: Per-chunk spectrogram statistics (optional)

    """

    chunk_index: int
    start_ms: float
    end_ms: float
    spectrogram: np.ndarray | None = None
    bboxes: list[ChunkBbox] = field(default_factory=list)
    source_uuid: str = ""
    sample_rate: int = 0
    is_padded: bool = False
    padding_amount_ms: float = 0.0
    stats: ChunkStats | None = None

    @property
    def duration_ms(self) -> float:
        """Return chunk duration in ms."""
        return self.end_ms - self.start_ms

    def get_chunk_id(self) -> str:
        """Return unique identifier for this chunk."""
        return f"{self.source_uuid}_chunk{self.chunk_index:04d}"


def compute_chunk_boundaries(
    total_duration_ms: float,
    config: ChunkConfig,
) -> list[tuple[float, float]]:
    """Compute start/end times for all chunks.

    Args:
        total_duration_ms: Total audio duration in milliseconds
        config: Chunk configuration

    Returns:
        List of (start_ms, end_ms) tuples for each chunk

    """
    if total_duration_ms <= 0:
        return []

    # Special case: audio shorter than window
    if total_duration_ms < config.window_duration_ms:
        return [(0.0, total_duration_ms)]

    boundaries = []
    stride_ms = config.window_duration_ms - config.get_overlap_ms()
    start_ms = 0.0

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + config.window_duration_ms, total_duration_ms)
        chunk_duration = end_ms - start_ms
        content_ratio = chunk_duration / config.window_duration_ms

        # Skip chunks with too little content
        if content_ratio >= config.min_chunk_content_ratio:
            boundaries.append((start_ms, end_ms))

        start_ms += stride_ms

        if stride_ms <= 0:
            break

    return boundaries


def compute_bbox_overlap(
    event_start_ms: float,
    event_end_ms: float,
    chunk_start_ms: float,
    chunk_end_ms: float,
) -> float:
    """Compute overlap ratio between event and chunk.

    Args:
        event_start_ms: Event start time (ms)
        event_end_ms: Event end time (ms)
        chunk_start_ms: Chunk start time (ms)
        chunk_end_ms: Chunk end time (ms)

    Returns:
        Overlap ratio (0-1) relative to event duration

    """
    event_duration = event_end_ms - event_start_ms
    if event_duration <= 0:
        return 0.0

    overlap_start = max(event_start_ms, chunk_start_ms)
    overlap_end = min(event_end_ms, chunk_end_ms)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    return overlap_duration / event_duration


def align_bbox_to_chunk(
    event_start_ms: float,
    event_end_ms: float,
    hz_min: float,
    hz_max: float,
    chunk_start_ms: float,
    chunk_end_ms: float,
    chunk_width_px: int,
    chunk_height_px: int,
    freq_min: float,
    freq_max: float,
    category: str,
    category_id: int,
    actual_width_px: int | None = None,
    use_mel_scale: bool = True,
) -> ChunkBbox | None:
    """Align event bbox to chunk pixel coordinates.

    Args:
        event_start_ms: Event start time (ms)
        event_end_ms: Event end time (ms)
        hz_min: Event minimum frequency (Hz)
        hz_max: Event maximum frequency (Hz)
        chunk_start_ms: Chunk start time (ms)
        chunk_end_ms: Chunk end time (ms)
        chunk_width_px: Chunk width in pixels
        chunk_height_px: Chunk height in pixels
        freq_min: Spectrogram minimum frequency (Hz)
        freq_max: Spectrogram maximum frequency (Hz)
        category: Category name
        category_id: Category ID
        actual_width_px: Actual content width before padding
        use_mel_scale: Use mel-scale frequency mapping

    Returns:
        ChunkBbox if event overlaps chunk, else None

    """
    chunk_duration = chunk_end_ms - chunk_start_ms
    if chunk_duration <= 0:
        return None

    overlap_start = max(event_start_ms, chunk_start_ms)
    overlap_end = min(event_end_ms, chunk_end_ms)

    if overlap_end <= overlap_start:
        return None

    overlap_ratio = compute_bbox_overlap(event_start_ms, event_end_ms, chunk_start_ms, chunk_end_ms)

    rel_start = overlap_start - chunk_start_ms
    rel_end = overlap_end - chunk_start_ms

    # Use actual content width if padding is present
    width_px = actual_width_px if actual_width_px is not None else chunk_width_px

    x = (rel_start / chunk_duration) * width_px
    x_end = (rel_end / chunk_duration) * width_px
    width = x_end - x

    # Frequency mapping
    if use_mel_scale:
        mel_min = hz_to_mel(freq_min)
        mel_max = hz_to_mel(freq_max)
        event_mel_min = hz_to_mel(hz_min)
        event_mel_max = hz_to_mel(hz_max)

        mel_range = mel_max - mel_min
        if mel_range <= 0:
            return None

        # ezakodio.viz.spectrogram_to_image flips vertically: high frequencies at TOP (y=0)
        # Compute normalized positions (0=low-freq end, 1=high-freq end), then flip.
        y_top = ((event_mel_max - mel_min) / mel_range) * chunk_height_px
        y_bottom = ((event_mel_min - mel_min) / mel_range) * chunk_height_px

        # Flip Y to match image convention where y=0 is at the top
        y_top = chunk_height_px - y_top
        y_bottom = chunk_height_px - y_bottom

        y = min(y_top, y_bottom)
        height = abs(y_bottom - y_top)
    else:
        freq_range = freq_max - freq_min
        if freq_range <= 0:
            return None

        y_bottom = ((hz_min - freq_min) / freq_range) * chunk_height_px
        y_top = ((hz_max - freq_min) / freq_range) * chunk_height_px
        y = min(y_top, y_bottom)
        height = abs(y_bottom - y_top)

    # Clamp to valid range
    x = max(0.0, min(x, chunk_width_px))
    y = max(0.0, min(y, chunk_height_px))
    width = max(0.0, min(width, chunk_width_px - x))
    height = max(0.0, min(height, chunk_height_px - y))

    if width <= 0 or height <= 0:
        return None

    return ChunkBbox(
        x=x,
        y=y,
        width=width,
        height=height,
        category=category,
        category_id=category_id,
        original_time_start_ms=event_start_ms,
        original_time_end_ms=event_end_ms,
        hz_min=hz_min,
        hz_max=hz_max,
        overlap_ratio=overlap_ratio,
    )


def pad_audio(
    audio: np.ndarray,
    target_samples: int,
    mode: str = "zero",
) -> tuple[np.ndarray, int]:
    """Pad audio to target length.

    Args:
        audio: Audio array (1D)
        target_samples: Target number of samples
        mode: Padding mode ('zero', 'repeat', 'reflect')

    Returns:
        Tuple of (padded audio, padding samples added)

    """
    current_samples = len(audio)
    if current_samples >= target_samples:
        return audio[:target_samples], 0

    padding_needed = target_samples - current_samples

    if mode == "zero":
        padded = np.pad(audio, (0, padding_needed), mode="constant", constant_values=0)
    elif mode == "repeat":
        repeats = (target_samples // current_samples) + 1
        padded = np.tile(audio, repeats)[:target_samples]
    elif mode == "reflect":
        padded = np.pad(audio, (0, padding_needed), mode="reflect")
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return padded, padding_needed


def extract_audio_chunk(
    audio: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    padding_mode: str = "zero",
    random_pad_position: bool = False,
) -> tuple[np.ndarray, bool, float]:
    """Extract a chunk from audio array.

    Args:
        audio: Full audio array (1D)
        sample_rate: Sample rate
        start_ms: Start time in ms
        end_ms: End time in ms
        padding_mode: How to pad if chunk extends beyond audio
        random_pad_position: Randomize audio position in padded chunk

    Returns:
        Tuple of (audio chunk, is_padded, padding_amount_ms)

    """
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)

    start_sample = max(0, start_sample)
    chunk = audio[start_sample:end_sample]

    target_samples = end_sample - start_sample
    is_padded = False
    padding_ms = 0.0

    if len(chunk) < target_samples:
        padding_needed = target_samples - len(chunk)

        if random_pad_position and padding_needed > 0:
            import random

            pad_before = random.randint(0, padding_needed)
            pad_after = padding_needed - pad_before
        else:
            pad_before = 0
            pad_after = padding_needed

        if padding_mode == "zero":
            chunk = np.pad(chunk, (pad_before, pad_after), mode="constant", constant_values=0)
        elif padding_mode == "repeat" and len(chunk) > 0:
            repeats = (target_samples // len(chunk)) + 2
            tiled = np.tile(chunk, repeats)
            chunk = tiled[pad_before : pad_before + target_samples]
        elif padding_mode == "reflect" and len(chunk) > 1:
            chunk = np.pad(chunk, (pad_before, pad_after), mode="reflect")
        else:
            chunk = np.pad(chunk, (pad_before, pad_after), mode="constant", constant_values=0)

        is_padded = True
        padding_ms = (padding_needed / sample_rate) * 1000

    return chunk, is_padded, padding_ms
