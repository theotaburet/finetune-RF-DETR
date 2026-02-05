"""Audio chunking module for fixed-size spectrogram generation.

This module provides sliding window chunking for audio files to produce
fixed-size spectrograms suitable for object detection models like RF-DETR.

Features:
- Time-based hop/n_fft for consistent resolution across sample rates
- Sliding windows with configurable overlap
- Padding for short audio files
- Bbox alignment to chunk coordinates
- Visual debugging with supervision library

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ezakodio import hz_to_mel
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TimeBasedFFTConfig:
    """Time-based FFT configuration for consistent resolution across sample rates.

    Instead of sample-based n_fft/hop_length, we use milliseconds to ensure
    consistent time/frequency resolution regardless of sample rate.

    Attributes:
        fft_ms: FFT window duration in milliseconds. Controls frequency resolution.
        hop_ms: Hop duration in milliseconds. Controls time resolution (ms per pixel).
        n_mels: Number of mel bands. Controls frequency axis height.

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
        window_duration_ms: Duration of each chunk in milliseconds.
        overlap_ratio: Overlap between consecutive chunks as a ratio (0.0-1.0).
            Example: 0.2 = 20% overlap. Converted to overlap_ms internally.
        min_overlap_with_event_ratio: Minimum overlap ratio for an event to be
            included in a chunk (0.0-1.0). Events with less overlap are excluded.
        target_width: Target image width in pixels. If None, computed from duration.
        target_height: Target image height in pixels. If None, uses n_mels.
        padding_mode: How to handle short audio ('zero', 'repeat', 'reflect').
        min_chunk_content_ratio: Minimum ratio of actual audio content in chunk (0.0-1.0).
            Chunks with less content are dropped to avoid useless padding.
            Default 0.5 means chunks need at least 50% real audio.
        random_pad_position: For very short audio (like 50ms gunshots), randomize where
            it sits in the padded chunk. Good for data augmentation/robustness.

    """

    window_duration_ms: float = 5000.0
    overlap_ratio: float = 0.2  # 20% overlap
    min_overlap_with_event_ratio: float = 0.3
    target_width: int | None = 640
    target_height: int | None = 640
    padding_mode: str = "zero"
    min_chunk_content_ratio: float = 0.5  # Drop chunks with <50% content
    random_pad_position: bool = True  # Randomize position for short audio

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
        x: X coordinate (pixels) relative to chunk.
        y: Y coordinate (pixels) relative to chunk.
        width: Width in pixels.
        height: Height in pixels.
        category: Category name.
        category_id: Category ID.
        original_time_start_ms: Original event start time in full audio.
        original_time_end_ms: Original event end time in full audio.
        hz_min: Minimum frequency in Hz.
        hz_max: Maximum frequency in Hz.
        overlap_ratio: How much of the original event is in this chunk (0-1).

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
class AudioChunk:
    """A chunk of audio with its metadata and bboxes.

    Attributes:
        chunk_index: Index of this chunk in the sequence.
        start_ms: Start time of chunk in original audio (ms).
        end_ms: End time of chunk in original audio (ms).
        spectrogram: Spectrogram tensor for this chunk.
        bboxes: List of bounding boxes in this chunk.
        source_uuid: UUID of the source audio file.
        sample_rate: Sample rate used.
        is_padded: Whether this chunk required padding.
        padding_amount_ms: Amount of padding added (ms).

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
        total_duration_ms: Total audio duration in milliseconds.
        config: Chunk configuration.

    Returns:
        List of (start_ms, end_ms) tuples for each chunk.

    """
    if total_duration_ms <= 0:
        return []

    # Special case: audio is shorter than window (e.g., 50ms gunshot in 5s window)
    # Create ONE chunk spanning the full audio - padding will center it with randomness
    if total_duration_ms < config.window_duration_ms:
        return [(0.0, total_duration_ms)]

    boundaries = []
    stride_ms = config.window_duration_ms - config.get_overlap_ms()
    start_ms = 0.0

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + config.window_duration_ms, total_duration_ms)
        chunk_duration = end_ms - start_ms
        content_ratio = chunk_duration / config.window_duration_ms

        # Skip chunks with too little content (avoid useless padding)
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
    """Compute overlap ratio between an event and a chunk.

    Args:
        event_start_ms: Event start time (ms).
        event_end_ms: Event end time (ms).
        chunk_start_ms: Chunk start time (ms).
        chunk_end_ms: Chunk end time (ms).

    Returns:
        Overlap ratio (0-1) relative to event duration.

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
    """Align an event bbox to chunk pixel coordinates.

    Args:
        event_start_ms: Event start time (ms) in original audio.
        event_end_ms: Event end time (ms) in original audio.
        hz_min: Event minimum frequency (Hz).
        hz_max: Event maximum frequency (Hz).
        chunk_start_ms: Chunk start time (ms).
        chunk_end_ms: Chunk end time (ms).
        chunk_width_px: Chunk width in pixels (target size, may include padding).
        chunk_height_px: Chunk height in pixels.
        freq_min: Spectrogram minimum frequency (Hz).
        freq_max: Spectrogram maximum frequency (Hz).
        category: Category name.
        category_id: Category ID.
        actual_width_px: Actual content width before padding (None = no padding).
        use_mel_scale: Use mel-scale frequency mapping (default True).

    Returns:
        ChunkBbox if event overlaps chunk sufficiently, else None.

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

    # Frequency mapping (mel or linear)
    if use_mel_scale:
        # Convert Hz to mel scale using ezakodio
        mel_min = hz_to_mel(freq_min)
        mel_max = hz_to_mel(freq_max)
        event_mel_min = hz_to_mel(hz_min)
        event_mel_max = hz_to_mel(hz_max)

        mel_range = mel_max - mel_min
        if mel_range <= 0:
            return None

        # After np.flipud, high frequencies are at TOP (y=0)
        # So we map from mel space with inversion
        y_top = ((event_mel_max - mel_min) / mel_range) * chunk_height_px
        y_bottom = ((event_mel_min - mel_min) / mel_range) * chunk_height_px

        # Flip Y axis because spectrogram is flipped
        y_top = chunk_height_px - y_top
        y_bottom = chunk_height_px - y_bottom

        y = min(y_top, y_bottom)
        height = abs(y_bottom - y_top)
    else:
        # Linear frequency mapping
        freq_range = freq_max - freq_min
        if freq_range <= 0:
            return None

        # After np.flipud, need to invert Y coordinates
        y_bottom = ((hz_min - freq_min) / freq_range) * chunk_height_px
        y_top = ((hz_max - freq_min) / freq_range) * chunk_height_px
        y = min(y_top, y_bottom)
        height = abs(y_bottom - y_top)

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
        audio: Audio array (1D).
        target_samples: Target number of samples.
        mode: Padding mode ('zero', 'repeat', 'reflect').

    Returns:
        Tuple of (padded audio, number of padding samples added).

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
        audio: Full audio array (1D).
        sample_rate: Sample rate.
        start_ms: Start time in ms.
        end_ms: End time in ms.
        padding_mode: How to pad if chunk extends beyond audio.
        random_pad_position: If True and padding is needed, randomize where
            audio sits in the padded chunk (data augmentation).

    Returns:
        Tuple of (audio chunk, is_padded, padding_amount_ms).

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
            # Random position for short audio (data augmentation)
            import random

            pad_before = random.randint(0, padding_needed)
            pad_after = padding_needed - pad_before
        else:
            # Default: pad only after (audio at start)
            pad_before = 0
            pad_after = padding_needed

        # Apply padding
        if padding_mode == "zero":
            chunk = np.pad(chunk, (pad_before, pad_after), mode="constant", constant_values=0)
        elif padding_mode == "repeat" and len(chunk) > 0:
            # Tile then position
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


def resize_spectrogram(
    spec: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize spectrogram to target dimensions.

    Strategy to preserve time resolution:
    - HEIGHT (frequency axis): Stretch/resize - this is OK, just visual scaling
    - WIDTH (time axis): Pad with zeros - preserves time resolution

    This ensures spectrograms from short audio aren't time-stretched, while
    keeping bboxes aligned properly.

    Args:
        spec: Spectrogram array (H, W) or (H, W, C).
        target_width: Target width.
        target_height: Target height.

    Returns:
        Resized/padded spectrogram as numpy array.

    """
    from PIL import Image

    orig_h, orig_w = spec.shape[:2]

    # If already correct size, return as-is
    if orig_h == target_height and orig_w == target_width:
        return spec

    # Step 1: Resize HEIGHT to target (stretch frequency axis - this is OK)
    if orig_h != target_height:
        if spec.ndim == 2:
            img = Image.fromarray(spec)
        else:
            img = Image.fromarray(spec.astype(np.uint8))

        # Resize only height, keep width
        resized = img.resize((orig_w, target_height), Image.Resampling.BILINEAR)
        spec = np.array(resized)

    # Step 2: PAD WIDTH to target (preserves time resolution)
    if orig_w < target_width:
        if spec.ndim == 2:
            padded = np.zeros((target_height, target_width), dtype=spec.dtype)
            padded[:, :orig_w] = spec
        else:
            padded = np.zeros((target_height, target_width, spec.shape[2]), dtype=spec.dtype)
            padded[:, :orig_w, :] = spec
        return padded
    elif orig_w > target_width:
        # If width is larger, crop (shouldn't happen with padding, but just in case)
        return spec[:, :target_width]
    else:
        return spec


def spectrogram_to_image_array(
    spec: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert spectrogram to image array (0-255, uint8).

    Args:
        spec: Spectrogram array (H, W).
        normalize: Whether to normalize to 0-255 range.

    Returns:
        Image array (H, W) as uint8.

    """
    if normalize:
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = np.zeros_like(spec)
        spec = (spec * 255).astype(np.uint8)
    else:
        spec = np.clip(spec, 0, 255).astype(np.uint8)

    return spec


class AudioChunker:
    """Main class for chunking audio files into fixed-size spectrograms.

    This class handles:
    - Loading and chunking audio files
    - Audio preprocessing (AGC, detrend)
    - Computing spectrograms for each chunk
    - Aligning bounding boxes to chunk coordinates
    - Resizing to target dimensions

    """

    def __init__(
        self,
        fft_config: TimeBasedFFTConfig | None = None,
        chunk_config: ChunkConfig | None = None,
        freq_scale: str = "mel",
        fmin: float = 0.0,
        fmax: float | None = None,
        preprocessing_config: Any | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            fft_config: Time-based FFT configuration.
            chunk_config: Chunk configuration.
            freq_scale: Frequency scale ('mel', 'linear', 'log').
            fmin: Minimum frequency for spectrogram.
            fmax: Maximum frequency (None = Nyquist).
            preprocessing_config: Audio preprocessing config (AGC, dynamic range).

        """
        self.fft_config = fft_config or TimeBasedFFTConfig()
        self.chunk_config = chunk_config or ChunkConfig()
        self.freq_scale = freq_scale
        self.fmin = fmin
        self.fmax = fmax
        self.preprocessing_config = preprocessing_config

    def _compute_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute spectrogram for audio chunk.

        Args:
            audio: Audio array (1D).
            sample_rate: Sample rate.

        Returns:
            Spectrogram as numpy array (H, W).

        """
        import torch
        from ezakodio.dsp import mel_spectrogram

        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        n_fft = self.fft_config.get_n_fft(sample_rate)
        hop_length = self.fft_config.get_hop_length(sample_rate)
        fmax = self.fmax if self.fmax else sample_rate / 2

        try:
            spec = mel_spectrogram(
                audio_tensor,
                sample_rate=sample_rate,
                n_mels=self.fft_config.n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=self.fmin,
                f_max=fmax,
                device="cpu",
            )

            if hasattr(spec, "cpu"):
                spec = spec.cpu().numpy()

            if spec.ndim == 3:
                spec = spec.squeeze(0)

            return spec

        except Exception as e:
            logger.error(f"Failed to compute spectrogram: {e}")
            raise

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        events: list[dict[str, Any]] | None = None,
        source_uuid: str = "",
    ) -> list[AudioChunk]:
        """Chunk audio into fixed-size segments with aligned bboxes.

        Args:
            audio: Audio array (1D).
            sample_rate: Sample rate.
            events: List of event dicts with keys:
                - time_start_ms, time_end_ms
                - hz_min, hz_max
                - category, category_id
            source_uuid: Source file UUID.

        Returns:
            List of AudioChunk objects.

        """
        events = events or []
        total_duration_ms = (len(audio) / sample_rate) * 1000
        fmax = self.fmax if self.fmax else sample_rate / 2

        boundaries = compute_chunk_boundaries(total_duration_ms, self.chunk_config)

        if not boundaries:
            logger.warning(f"No chunks generated for audio of {total_duration_ms:.1f}ms")
            return []

        chunks = []
        for idx, (start_ms, end_ms) in enumerate(boundaries):
            chunk_audio, is_padded, padding_ms = extract_audio_chunk(
                audio,
                sample_rate,
                start_ms,
                end_ms,
                self.chunk_config.padding_mode,
                random_pad_position=self.chunk_config.random_pad_position,
            )

            # Preprocessing already applied to full audio file (not per-chunk)
            # This ensures AGC/detrend has full file context
            spec = self._compute_spectrogram(chunk_audio, sample_rate)
            # Flip vertically so high frequencies are at top (standard visualization)
            spec = np.flipud(spec)

            # Apply dynamic range compression if configured
            if self.preprocessing_config is not None:
                from rf_detr_finetuning.audio_preprocessing import normalize_spectrogram

                spec_img = normalize_spectrogram(spec, self.preprocessing_config.dynamic_range)
            else:
                spec_img = spectrogram_to_image_array(spec)

            orig_height, orig_width = spec_img.shape[:2]
            target_w = self.chunk_config.target_width or orig_width
            target_h = self.chunk_config.target_height or orig_height

            # Track actual content width before padding
            actual_width = orig_width

            if (target_w != orig_width) or (target_h != orig_height):
                spec_img = resize_spectrogram(spec_img, target_w, target_h)

            chunk_bboxes = []
            for event in events:
                bbox = align_bbox_to_chunk(
                    event_start_ms=event.get("time_start_ms", 0),
                    event_end_ms=event.get("time_end_ms", total_duration_ms),
                    hz_min=event.get("hz_min", self.fmin),
                    hz_max=event.get("hz_max", fmax),
                    chunk_start_ms=start_ms,
                    chunk_end_ms=end_ms,
                    chunk_width_px=target_w,
                    chunk_height_px=target_h,
                    freq_min=self.fmin,
                    freq_max=fmax,
                    category=event.get("category", "unknown"),
                    category_id=event.get("category_id", 0),
                    actual_width_px=actual_width,  # Pass actual width before padding
                    use_mel_scale=(self.freq_scale == "mel"),  # Use mel scale if configured
                )

                # For file-level annotations, include bbox in all chunks
                # For event-level annotations, only include if overlap ratio meets threshold
                is_file_level = event.get("is_file_level", False)
                if bbox and (is_file_level or bbox.overlap_ratio >= self.chunk_config.min_overlap_with_event_ratio):
                    chunk_bboxes.append(bbox)

            chunk = AudioChunk(
                chunk_index=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                spectrogram=spec_img,
                bboxes=chunk_bboxes,
                source_uuid=source_uuid,
                sample_rate=sample_rate,
                is_padded=is_padded,
                padding_amount_ms=padding_ms,
            )
            chunks.append(chunk)

        return chunks

    def chunk_audio_file(
        self,
        audio_path: Path,
        metadata_path: Path | None = None,
    ) -> list[AudioChunk]:
        """Chunk an audio file with optional metadata.

        Args:
            audio_path: Path to audio file.
            metadata_path: Path to JSON metadata file (optional).

        Returns:
            List of AudioChunk objects.

        """
        from ezakodio.io import load_audio

        audio_tensor, sample_rate = load_audio(str(audio_path), mono=True, device="cpu")
        audio = audio_tensor.cpu().numpy().flatten()

        # Apply preprocessing to FULL audio file (not per-chunk)
        # This ensures AGC/detrend works on entire file context (e.g., 10min)
        # rather than small chunks (e.g., 6 seconds)
        if self.preprocessing_config is not None:
            from rf_detr_finetuning.audio_preprocessing import preprocess_audio

            audio, preprocess_metadata = preprocess_audio(audio, sample_rate, self.preprocessing_config)
            logger.debug(
                f"Preprocessed {audio_path.name}: "
                f"gain={preprocess_metadata.get('gain_applied_db', 0):.2f}dB, "
                f"RMS {preprocess_metadata['original_rms_db']:.2f}→{preprocess_metadata['final_rms_db']:.2f}dB"
            )

        events = []
        source_uuid = audio_path.stem

        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)

            source_uuid = meta.get("uuid", source_uuid)
            fmax = self.fmax if self.fmax else sample_rate / 2

            # Get actual audio duration (in case metadata is wrong)
            actual_duration_ms = (len(audio) / sample_rate) * 1000

            # Extract label from hierarchy (preferred) or fallback to annotation
            hierarchy = meta.get("label_hierarchy", "")
            if hierarchy and " > " in hierarchy:
                label = hierarchy.split(" > ")[-1].strip()
            elif hierarchy:
                label = hierarchy
            else:
                label = meta.get("annotation", "unknown")

            # Create file-level bbox spanning entire audio
            # Use actual duration, not metadata duration (which might include padding)
            # Mark as file-level so it bypasses overlap ratio check
            events.append(
                {
                    "time_start_ms": 0,
                    "time_end_ms": actual_duration_ms,
                    "hz_min": meta.get("hz_min", self.fmin),
                    "hz_max": meta.get("hz_max", fmax),
                    "category": label or "unknown",
                    "category_id": 0,
                    "is_file_level": True,  # Skip overlap check for file-level annotations
                }
            )

        return self.chunk_audio(audio, sample_rate, events, source_uuid)


def draw_bboxes_on_spectrogram(
    spectrogram: np.ndarray,
    bboxes: list[ChunkBbox],
    output_path: Path | None = None,
) -> np.ndarray:
    """Draw bounding boxes on spectrogram using supervision.

    Args:
        spectrogram: Spectrogram image array (H, W) or (H, W, 3).
        bboxes: List of ChunkBbox objects.
        output_path: Optional path to save annotated image.

    Returns:
        Annotated image as numpy array (H, W, 3).

    """
    try:
        import supervision as sv
    except ImportError:
        logger.warning("supervision not installed, drawing boxes manually")
        return _draw_bboxes_manual(spectrogram, bboxes, output_path)

    if spectrogram.ndim == 2:
        img = np.stack([spectrogram] * 3, axis=-1)
    else:
        img = spectrogram.copy()

    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    if not bboxes:
        if output_path:
            Image.fromarray(img).save(output_path)
        return img

    xyxy = np.array([bbox.to_xyxy() for bbox in bboxes])
    class_ids = np.array([bbox.category_id for bbox in bboxes])
    labels = [f"{bbox.category} ({bbox.overlap_ratio:.0%})" for bbox in bboxes]

    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_ids,
    )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    img = box_annotator.annotate(img, detections)
    img = label_annotator.annotate(img, detections, labels)

    if output_path:
        Image.fromarray(img).save(output_path)

    return img


def _draw_bboxes_manual(
    spectrogram: np.ndarray,
    bboxes: list[ChunkBbox],
    output_path: Path | None = None,
) -> np.ndarray:
    """Fallback bbox drawing without supervision."""
    if spectrogram.ndim == 2:
        img = np.stack([spectrogram] * 3, axis=-1)
    else:
        img = spectrogram.copy()

    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox.to_xyxy())
        x1 = max(0, min(x1, img.shape[1] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        color = (0, 255, 0)
        img[y1 : y1 + 2, x1:x2] = color
        img[y2 - 2 : y2, x1:x2] = color
        img[y1:y2, x1 : x1 + 2] = color
        img[y1:y2, x2 - 2 : x2] = color

    if output_path:
        Image.fromarray(img).save(output_path)

    return img


def load_chunking_config_from_yaml(
    yaml_path: Path,
) -> tuple[TimeBasedFFTConfig, ChunkConfig, Any | None]:
    """Load chunking configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Tuple of (TimeBasedFFTConfig, ChunkConfig, PreprocessingConfig or None).

    Example YAML::

        fft:
          fft_ms: 25.0
          hop_ms: 10.0
          n_mels: 128

        chunking:
          target_size: 640  # Auto-computes target_width, target_height, window_duration_ms
          overlap_ratio: 0.2  # 20% overlap (or use overlap_ms for backward compatibility)
          min_overlap_with_event_ratio: 0.3
          padding_mode: zero
          min_chunk_content_ratio: 0.5
          random_pad_position: true

        preprocessing:
          agc:
            enabled: true
            target_db: -25.0
          dynamic_range:
            top_db: 70.0
            clip_percentile: 99.0

    """
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    fft_section = config.get("fft", {})
    chunk_section = config.get("chunking", {})
    preprocessing_section = config.get("preprocessing", {})

    fft_config = TimeBasedFFTConfig(
        fft_ms=fft_section.get("fft_ms", 25.0),
        hop_ms=fft_section.get("hop_ms", 10.0),
        n_mels=fft_section.get("n_mels", 128),
    )

    # Support both old format (explicit dimensions) and new format (target_size)
    target_size = chunk_section.get("target_size")
    if target_size is not None:
        # New format: derive dimensions from target_size
        target_width = target_size
        target_height = target_size
        # Auto-calculate window_duration_ms = target_size * hop_ms
        window_duration_ms = target_size * fft_config.hop_ms
    else:
        # Old format: explicit dimensions
        target_width = chunk_section.get("target_width", 640)
        target_height = chunk_section.get("target_height", 640)
        window_duration_ms = chunk_section.get("window_duration_ms", 5000.0)

    # Support both overlap_ratio (new) and overlap_ms (old) for backward compatibility
    if "overlap_ratio" in chunk_section:
        overlap_ratio = chunk_section["overlap_ratio"]
    elif "overlap_ms" in chunk_section:
        # Convert old overlap_ms to ratio
        overlap_ratio = chunk_section["overlap_ms"] / window_duration_ms
    else:
        overlap_ratio = 0.2  # Default 20%

    chunk_config = ChunkConfig(
        window_duration_ms=window_duration_ms,
        overlap_ratio=overlap_ratio,
        min_overlap_with_event_ratio=chunk_section.get("min_overlap_with_event_ratio", 0.3),
        target_width=target_width,
        target_height=target_height,
        padding_mode=chunk_section.get("padding_mode", "zero"),
        min_chunk_content_ratio=chunk_section.get("min_chunk_content_ratio", 0.5),
        random_pad_position=chunk_section.get("random_pad_position", True),
    )

    # Load preprocessing config if present
    preprocessing_config = None
    if preprocessing_section:
        from rf_detr_finetuning.audio_preprocessing import PreprocessingConfig

        preprocessing_config = PreprocessingConfig.from_dict(preprocessing_section)

    return fft_config, chunk_config, preprocessing_config
