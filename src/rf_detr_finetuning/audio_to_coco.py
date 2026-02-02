"""Audio dataset to COCO format converter for RF-DETR training.

This module converts audio files with JSON metadata into spectrogram images
with COCO-format annotations for object detection training.

The key concept: Audio events become bounding boxes on spectrograms where:
- X-axis = Time (samples → pixels)
- Y-axis = Frequency (Hz → mel bins → pixels)

Frequency information is preserved in the annotations metadata to enable
frequency-aware classification (e.g., low-frequency horizontal = ship noise,
high-frequency = sonar).

Uses ezakodio for all audio processing (loading, spectrogram generation, etc.)

"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Import ezakodio components
from ezakodio import hz_to_mel, mel_to_hz
from ezakodio.dsp import mel_spectrogram
from ezakodio.io import load_audio
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram generation.

    Attributes:
        n_fft: FFT window size.
        hop_length: Number of samples between successive frames.
        n_mels: Number of mel filterbanks.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank (None = sr/2).
        power: Exponent for the magnitude spectrogram (1=energy, 2=power).
        target_sr: Target sample rate for resampling.
            - If set: All files resampled to this SR (simpler, recommended for beginners)
            - If None: Keep original SR, use normalize_frequency=True (advanced)
        normalize_frequency: Handle different sample rates gracefully.
            - True: Use frequency normalized to Nyquist (0-1), works with mixed SRs
            - False: Use absolute Hz values (requires consistent SR via target_sr)
            Recommended: True if target_sr=None, False if target_sr is set.
        normalize: Whether to normalize the spectrogram to 0-255 for image.
        log_scale: Apply log scaling to spectrogram (dB scale).
        ref_db: Reference dB level for normalization.
        min_db: Minimum dB level (values below are clipped).

    """

    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    power: float = 2.0
    target_sr: int | None = None
    normalize_frequency: bool = True  # Default to normalized mode for flexibility
    normalize: bool = True
    log_scale: bool = True
    ref_db: float = 0.0
    min_db: float = -80.0


@dataclass
class AudioMetadata:
    """Parsed metadata from audio JSON file.

    Attributes:
        uuid: Unique identifier for the audio file.
        label_hierarchy: Top-level label category.
        annotation: Specific annotation/class label.
        duration: Duration in milliseconds.
        hz_min: Minimum frequency of the event (Hz).
        hz_max: Maximum frequency of the event (Hz).
        sample_rate: Original sample rate.
        channels: Number of audio channels.
        source_start: Start time in source file (ms).
        source_end: End time in source file (ms).
        confidence: Labeling confidence score.
        other_labels: Additional labels for the sample.
        custom: Custom metadata dictionary.
        raw: Original raw JSON data.

    """

    uuid: str
    label_hierarchy: str
    annotation: str
    duration: float  # ms
    hz_min: float
    hz_max: float
    sample_rate: int
    channels: int = 1
    source_start: float = 0.0
    source_end: float = 0.0
    confidence: float = 1.0
    other_labels: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_path: Path | str) -> AudioMetadata:
        """Load metadata from a JSON file."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioMetadata:
        """Create metadata from a dictionary."""
        return cls(
            uuid=data.get("uuid", ""),
            label_hierarchy=data.get("label_hierarchy", ""),
            annotation=data.get("annotation", "unknown"),
            duration=float(data.get("duration", 0)),
            hz_min=float(data.get("hz_min", 0.0)),
            hz_max=float(data.get("hz_max", 22050.0)),
            sample_rate=int(data.get("sample_rate", 44100)),
            channels=int(data.get("channels", 1)),
            source_start=float(data.get("source_start", 0.0)),
            source_end=float(data.get("source_end", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
            other_labels=data.get("other_labels", []),
            custom=data.get("custom", {}),
            raw=data,
        )


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (COCO format: x, y, width, height).

    Attributes:
        x: Left edge x-coordinate.
        y: Top edge y-coordinate (0 = top of image = highest frequency).
        width: Box width in pixels.
        height: Box height in pixels.
        category_id: Category/class ID.
        category_name: Category/class name.
        hz_min: Original minimum frequency (Hz) - for metadata.
        hz_max: Original maximum frequency (Hz) - for metadata.
        time_start_ms: Original start time (ms) - for metadata.
        time_end_ms: Original end time (ms) - for metadata.

    """

    x: float
    y: float
    width: float
    height: float
    category_id: int
    category_name: str
    hz_min: float = 0.0
    hz_max: float = 0.0
    time_start_ms: float = 0.0
    time_end_ms: float = 0.0

    def to_coco_annotation(self, annotation_id: int, image_id: int) -> dict[str, Any]:
        """Convert to COCO annotation format."""
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": self.category_id,
            "bbox": [self.x, self.y, self.width, self.height],
            "area": self.width * self.height,
            "iscrowd": 0,
            # Extended metadata for frequency-aware analysis
            "attributes": {
                "hz_min": self.hz_min,
                "hz_max": self.hz_max,
                "time_start_ms": self.time_start_ms,
                "time_end_ms": self.time_end_ms,
            },
        }


# =============================================================================
# Frequency-to-Pixel Conversion
# =============================================================================


class FrequencyMapper:
    """Maps frequencies (Hz) to spectrogram pixel positions.

    For mel spectrograms, the y-axis is in mel scale, not linear Hz. This class handles the conversion properly using
    ezakodio's functions.

    Supports both absolute Hz and normalized frequency (0-1 relative to Nyquist).

    """

    def __init__(
        self,
        n_mels: int,
        fmin: float,
        fmax: float,
        sample_rate: int,
        normalize_frequency: bool = False,
    ):
        """Initialize the frequency mapper.

        Args:
            n_mels: Number of mel bins in the spectrogram.
            fmin: Minimum frequency of the mel filterbank.
            fmax: Maximum frequency of the mel filterbank.
            sample_rate: Audio sample rate.
            normalize_frequency: If True, work with normalized freq (0-1).

        """
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax else sample_rate / 2
        self.sample_rate = sample_rate
        self.normalize_frequency = normalize_frequency
        self.nyquist = sample_rate / 2

        # Pre-compute mel scale boundaries using ezakodio
        self.mel_min = hz_to_mel(self.fmin)
        self.mel_max = hz_to_mel(self.fmax)

    def hz_to_pixel(self, hz: float) -> float:
        """Convert frequency in Hz to pixel row (0 = top = highest freq).

        Spectrogram images are typically displayed with high frequencies at top.

        """
        # Clamp to valid range
        hz = max(self.fmin, min(hz, self.fmax))

        # Convert to mel using ezakodio
        mel = hz_to_mel(hz)

        # Normalize to 0-1 range in mel space
        mel_normalized = (mel - self.mel_min) / (self.mel_max - self.mel_min)

        # Convert to pixel (invert so high freq = top = row 0)
        pixel = (1.0 - mel_normalized) * self.n_mels

        return pixel

    def pixel_to_hz(self, pixel: float) -> float:
        """Convert pixel row to frequency in Hz."""
        # Invert: pixel 0 = top = highest freq
        mel_normalized = 1.0 - (pixel / self.n_mels)
        mel = mel_normalized * (self.mel_max - self.mel_min) + self.mel_min
        return mel_to_hz(mel)

    def hz_to_normalized(self, hz: float) -> float:
        """Convert Hz to normalized frequency (0-1 relative to Nyquist)."""
        return hz / self.nyquist

    def normalized_to_hz(self, norm_freq: float) -> float:
        """Convert normalized frequency to Hz."""
        return norm_freq * self.nyquist

    def get_frequency_value(self, hz: float) -> float:
        """Get frequency value in appropriate scale (Hz or normalized)."""
        if self.normalize_frequency:
            return self.hz_to_normalized(hz)
        return hz

    def to_hz(self, freq_value: float) -> float:
        """Convert frequency value (Hz or normalized) back to Hz."""
        if self.normalize_frequency:
            return self.normalized_to_hz(freq_value)
        return freq_value


class TimeMapper:
    """Maps time (ms) to spectrogram pixel positions."""

    def __init__(
        self,
        total_duration_ms: float,
        total_frames: int,
        sample_rate: int,
        hop_length: int,
    ):
        """Initialize the time mapper.

        Args:
            total_duration_ms: Total audio duration in milliseconds.
            total_frames: Total number of spectrogram frames (columns).
            sample_rate: Audio sample rate.
            hop_length: Hop length used for STFT.

        """
        self.total_duration_ms = total_duration_ms
        self.total_frames = total_frames
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.ms_per_frame = (hop_length / sample_rate) * 1000.0

    def ms_to_pixel(self, time_ms: float) -> float:
        """Convert time in ms to pixel column."""
        return time_ms / self.ms_per_frame

    def pixel_to_ms(self, pixel: float) -> float:
        """Convert pixel column to time in ms."""
        return pixel * self.ms_per_frame


# =============================================================================
# Spectrogram Generation
# =============================================================================


def spectrogram_to_image(
    spectrogram: np.ndarray,
    normalize: bool = True,
) -> Image.Image:
    """Convert spectrogram array to PIL Image.

    Args:
        spectrogram: 2D spectrogram array (n_mels, n_frames).
        normalize: Normalize to 0-255 range.

    Returns:
        PIL Image in RGB format.

    """
    spec = spectrogram.copy()

    if normalize:
        # Normalize to 0-255
        spec_min, spec_max = spec.min(), spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min) * 255.0
        else:
            spec = np.zeros_like(spec)

    # Convert to uint8
    spec = spec.astype(np.uint8)

    # Create RGB image (grayscale repeated across channels)
    img = Image.fromarray(spec, mode="L").convert("RGB")

    return img


# =============================================================================
# Bbox Coordinate Decoder
# =============================================================================


@dataclass
class BboxFrequencyInfo:
    """Frequency and time information extracted from bbox coordinates.

    Supports both absolute Hz and normalized frequency (0-1 relative to Nyquist).

    """

    hz_min: float
    hz_max: float
    hz_center: float
    time_start_ms: float
    time_end_ms: float
    duration_ms: float
    bandwidth_hz: float
    sample_rate: int = 0
    normalized_freq_min: float = 0.0
    normalized_freq_max: float = 1.0
    normalized_freq_center: float = 0.5

    def get_frequency_band(self) -> str:
        """Classify frequency into perceptual bands."""
        if self.hz_center < 100:
            return "infrasonic"
        elif self.hz_center < 500:
            return "very_low"
        elif self.hz_center < 2000:
            return "low"
        elif self.hz_center < 5000:
            return "mid"
        elif self.hz_center < 10000:
            return "high"
        else:
            return "very_high"

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "hz_min": self.hz_min,
            "hz_max": self.hz_max,
            "hz_center": self.hz_center,
            "normalized_freq_min": self.normalized_freq_min,
            "normalized_freq_max": self.normalized_freq_max,
            "normalized_freq_center": self.normalized_freq_center,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "duration_ms": self.duration_ms,
            "bandwidth_hz": self.bandwidth_hz,
            "sample_rate": self.sample_rate,
            "frequency_band": self.get_frequency_band(),
        }


def decode_bbox_to_frequency(
    bbox: BoundingBox,
    freq_mapper: FrequencyMapper,
    time_mapper: TimeMapper,
) -> BboxFrequencyInfo:
    """Extract Hz and time information from bbox pixel coordinates.

    This is the inverse operation - converting bbox pixel coordinates
    back to actual Hz and millisecond values.

    Args:
        bbox: Bounding box with pixel coordinates.
        freq_mapper: Frequency mapper for the spectrogram.
        time_mapper: Time mapper for the spectrogram.

    Returns:
        BboxFrequencyInfo with decoded Hz and time values.

    """
    # Decode Y coordinates (frequency) - remember Y is inverted (0 = top = high freq)
    y_top = bbox.y
    y_bottom = bbox.y + bbox.height

    hz_max = freq_mapper.pixel_to_hz(y_top)  # Top = high freq
    hz_min = freq_mapper.pixel_to_hz(y_bottom)  # Bottom = low freq
    hz_center = (hz_min + hz_max) / 2
    bandwidth = hz_max - hz_min

    # Decode X coordinates (time)
    x_left = bbox.x
    x_right = bbox.x + bbox.width

    time_start = time_mapper.pixel_to_ms(x_left)
    time_end = time_mapper.pixel_to_ms(x_right)
    duration = time_end - time_start

    nyquist = freq_mapper.sample_rate / 2
    return BboxFrequencyInfo(
        hz_min=hz_min,
        hz_max=hz_max,
        hz_center=hz_center,
        time_start_ms=time_start,
        time_end_ms=time_end,
        duration_ms=duration,
        bandwidth_hz=bandwidth,
        sample_rate=freq_mapper.sample_rate,
        normalized_freq_min=hz_min / nyquist,
        normalized_freq_max=hz_max / nyquist,
        normalized_freq_center=hz_center / nyquist,
    )


def infer_frequency_bins_from_data(
    audio_files: list[tuple[Path, Path]],
    n_bins: int = 3,
) -> list[tuple[float, float, str]]:
    """Automatically infer frequency bins from dataset by clustering.

    Analyzes the frequency distribution in the dataset and creates
    optimal frequency bins using percentile-based splitting.

    Args:
        audio_files: List of (audio_path, json_path) tuples.
        n_bins: Number of frequency bins to create.

    Returns:
        List of (hz_min, hz_max, name) tuples for frequency bins.

    """
    import numpy as np

    # Collect all frequency centers from metadata
    freq_centers = []
    for _, json_path in audio_files:
        try:
            meta = AudioMetadata.from_json(json_path)
            center = (meta.hz_min + meta.hz_max) / 2
            freq_centers.append(center)
        except Exception:
            continue

    if not freq_centers:
        # Fallback to default bins
        return [
            (0, 500, "low"),
            (500, 5000, "mid"),
            (5000, 22050, "high"),
        ]

    freq_centers = np.array(freq_centers)

    # Create bins using percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(freq_centers, percentiles)

    # Create bin names
    bin_names = []
    if n_bins == 2:
        bin_names = ["low", "high"]
    elif n_bins == 3:
        bin_names = ["low", "mid", "high"]
    elif n_bins == 4:
        bin_names = ["very_low", "low", "high", "very_high"]
    elif n_bins == 5:
        bin_names = ["very_low", "low", "mid", "high", "very_high"]
    else:
        bin_names = [f"band_{i}" for i in range(n_bins)]

    # Create frequency bins
    frequency_bins = []
    for i in range(n_bins):
        frequency_bins.append(
            (
                float(bin_edges[i]),
                float(bin_edges[i + 1]),
                bin_names[i],
            )
        )

    logger.info(f"Inferred {n_bins} frequency bins from {len(freq_centers)} samples:")
    for hz_min, hz_max, name in frequency_bins:
        logger.info(f"  {name}: {hz_min:.1f} - {hz_max:.1f} Hz")

    return frequency_bins


# =============================================================================
# Dataset Processing
# =============================================================================


@dataclass
class CategoryRegistry:
    """Registry for managing category mappings.

    Supports frequency-aware category generation where the same annotation can map to different categories based on
    frequency range.

    """

    categories: dict[str, int] = field(default_factory=dict)
    frequency_bins: list[tuple[float, float, str]] | None = None

    def __post_init__(self):
        """Post-initialization hook for dataclass."""
        self._next_id = 0

    def get_or_create(self, name: str) -> int:
        """Get category ID, creating if necessary."""
        if name not in self.categories:
            self.categories[name] = self._next_id
            self._next_id += 1
        return self.categories[name]

    def get_category_with_frequency(
        self,
        base_name: str,
        hz_min: float,
        hz_max: float,
    ) -> tuple[int, str]:
        """Get category considering frequency range.

        If frequency_bins is set, creates compound categories like:
        - "ship_noise_low_freq" for events below 500 Hz
        - "sonar_high_freq" for events above 5000 Hz

        Args:
            base_name: Base annotation name.
            hz_min: Minimum frequency of the event.
            hz_max: Maximum frequency of the event.

        Returns:
            Tuple of (category_id, category_name).

        """
        if self.frequency_bins is None:
            cat_id = self.get_or_create(base_name)
            return cat_id, base_name

        # Find matching frequency bin
        center_freq = (hz_min + hz_max) / 2
        suffix = ""

        for bin_min, bin_max, bin_name in self.frequency_bins:
            if bin_min <= center_freq < bin_max:
                suffix = f"_{bin_name}"
                break

        full_name = f"{base_name}{suffix}"
        cat_id = self.get_or_create(full_name)
        return cat_id, full_name

    def to_coco_categories(self) -> list[dict[str, Any]]:
        """Export categories in COCO format."""
        return [
            {"id": cat_id, "name": name, "supercategory": "audio_event"}
            for name, cat_id in sorted(self.categories.items(), key=lambda x: x[1])
        ]


def iterate_audio_dataset(
    input_dir: Path | str,
    audio_extensions: tuple[str, ...] = (".flac", ".wav", ".mp3", ".ogg"),
) -> Iterator[tuple[Path, Path]]:
    """Iterate over audio files with matching JSON metadata.

    Args:
        input_dir: Directory containing audio and JSON files.
        audio_extensions: Audio file extensions to look for.

    Yields:
        Tuples of (audio_path, json_path).

    """
    input_dir = Path(input_dir)

    for ext in audio_extensions:
        for audio_path in input_dir.glob(f"*{ext}"):
            json_path = audio_path.with_suffix(".json")
            if json_path.exists():
                yield audio_path, json_path
            else:
                logger.warning(f"No JSON metadata found for {audio_path}")


def process_audio_file(
    audio_path: Path,
    metadata: AudioMetadata,
    spec_config: SpectrogramConfig,
    category_registry: CategoryRegistry,
) -> tuple[Image.Image, BoundingBox, dict[str, Any]]:
    """Process a single audio file into spectrogram + bounding box.

    Uses ezakodio for audio loading and spectrogram computation.

    Supports two modes for handling different sample rates:
    1. Resampling mode (target_sr set): All files resampled to target_sr
    2. Normalized mode (target_sr=None, normalize_frequency=True): Keep original SRs,
       use frequency normalized to Nyquist (0-1) for SR-independent features

    Args:
        audio_path: Path to audio file.
        metadata: Audio metadata.
        spec_config: Spectrogram configuration.
        category_registry: Category registry for label management.

    Returns:
        Tuple of (spectrogram_image, bounding_box, extra_metadata).

    """
    # Load audio using ezakodio
    target_sr = spec_config.target_sr
    if target_sr is None:
        target_sr = metadata.sample_rate
        if not spec_config.normalize_frequency:
            logger.warning(
                f"target_sr=None and normalize_frequency=False! "
                f"Using original SR={metadata.sample_rate} Hz. "
                "Mixed SRs will cause issues. Set target_sr or use normalize_frequency=True."
            )

    audio_tensor, sr = load_audio(
        str(audio_path),
        sample_rate=target_sr,
        mono=True,
        device="cpu",
    )

    # Compute mel spectrogram using ezakodio
    fmax = spec_config.fmax if spec_config.fmax else sr / 2
    mel_spec_tensor = mel_spectrogram(
        audio_tensor,
        sample_rate=sr,
        n_mels=spec_config.n_mels,
        n_fft=spec_config.n_fft,
        hop_length=spec_config.hop_length,
        f_min=spec_config.fmin,
        f_max=fmax,
        power=spec_config.power,
        log_scale=spec_config.log_scale,
        device="cpu",
    )

    # Convert to numpy, remove batch dimension if present
    mel_spec = mel_spec_tensor.cpu().numpy()
    if mel_spec.ndim == 3:
        mel_spec = mel_spec[0]

    # Convert to image
    img = spectrogram_to_image(mel_spec, normalize=spec_config.normalize)

    # Get spectrogram dimensions
    n_mels, n_frames = mel_spec.shape

    # Create mappers
    freq_mapper = FrequencyMapper(
        n_mels, spec_config.fmin, fmax, sr, normalize_frequency=spec_config.normalize_frequency
    )
    time_mapper = TimeMapper(metadata.duration, n_frames, sr, spec_config.hop_length)

    # For whole-file annotations, the bbox covers the entire spectrogram
    # You can extend this for segment-level annotations
    if metadata.hz_min == 0 and metadata.hz_max >= fmax * 0.9:
        # Full-band event - cover entire height
        y_top = 0
        y_bottom = n_mels
    else:
        # Frequency-specific event
        y_top = freq_mapper.hz_to_pixel(metadata.hz_max)  # High freq = top
        y_bottom = freq_mapper.hz_to_pixel(metadata.hz_min)  # Low freq = bottom

    # Time bounds (full duration for single-file annotation)
    x_left = 0
    x_right = n_frames

    # Get category with frequency awareness
    cat_id, cat_name = category_registry.get_category_with_frequency(
        metadata.annotation,
        metadata.hz_min,
        metadata.hz_max,
    )

    bbox = BoundingBox(
        x=float(x_left),
        y=float(y_top),
        width=float(x_right - x_left),
        height=float(y_bottom - y_top),
        category_id=cat_id,
        category_name=cat_name,
        hz_min=metadata.hz_min,
        hz_max=metadata.hz_max,
        time_start_ms=0.0,
        time_end_ms=metadata.duration,
    )

    # Decode bbox coordinates back to Hz/time for verification and extra features
    freq_info = decode_bbox_to_frequency(bbox, freq_mapper, time_mapper)

    extra_metadata = {
        "original_sr": metadata.sample_rate,
        "used_sr": sr,
        "sample_rate": sr,  # Current SR for this spectrogram
        "nyquist_hz": sr / 2,  # Maximum frequency
        "n_frames": n_frames,
        "n_mels": n_mels,
        "label_hierarchy": metadata.label_hierarchy,
        "confidence": metadata.confidence,
        "uuid": metadata.uuid,
        "normalize_frequency": spec_config.normalize_frequency,
        # Add decoded frequency info for use in model training
        "frequency_info": freq_info.to_dict(),
    }

    return img, bbox, extra_metadata


# =============================================================================
# COCO Dataset Builder
# =============================================================================


@dataclass
class COCODataset:
    """Builder for COCO-format dataset."""

    info: dict[str, Any] = field(
        default_factory=lambda: {
            "description": "Audio spectrogram dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat(),
        }
    )
    licenses: list[dict[str, Any]] = field(default_factory=lambda: [{"id": 1, "name": "Unknown", "url": ""}])
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)

    _image_id: int = field(default=1, init=False)
    _annotation_id: int = field(default=1, init=False)

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Add an image to the dataset.

        Returns:
            Image ID.

        """
        image_entry = {
            "id": self._image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            image_entry.update(extra)

        self.images.append(image_entry)
        image_id = self._image_id
        self._image_id += 1
        return image_id

    def add_annotation(
        self,
        bbox: BoundingBox,
        image_id: int,
    ) -> int:
        """Add an annotation to the dataset.

        Returns:
            Annotation ID.

        """
        annotation = bbox.to_coco_annotation(self._annotation_id, image_id)
        self.annotations.append(annotation)
        annotation_id = self._annotation_id
        self._annotation_id += 1
        return annotation_id

    def set_categories(self, category_registry: CategoryRegistry) -> None:
        """Set categories from registry."""
        self.categories = category_registry.to_coco_categories()

    def to_dict(self) -> dict[str, Any]:
        """Export dataset to dictionary."""
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, output_path: Path | str) -> None:
        """Save dataset to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved COCO dataset to {output_path}")


# =============================================================================
# Main Conversion Pipeline
# =============================================================================


def convert_audio_to_coco(
    input_dir: str | Path,
    output_dir: str | Path,
    spec_config: SpectrogramConfig | None = None,
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    frequency_bins: list[tuple[float, float, str]] | None = None,
    auto_infer_bins: bool | int = False,
    audio_extensions: tuple[str, ...] = (".flac", ".wav", ".mp3", ".ogg"),
    random_seed: int | None = 42,
) -> Path:
    """Convert an audio dataset to COCO format for RF-DETR training.

    Handles datasets with mixed sample rates in two ways:
    1. Resampling mode (target_sr set, normalize_frequency=False):
       - All files resampled to target_sr
       - Uses absolute Hz values
       - Simpler, recommended for beginners

    2. Normalized mode (target_sr=None, normalize_frequency=True):
       - Keep original sample rates
       - Uses frequency normalized to Nyquist (0-1)
       - More flexible, handles any SR mix

    Args:
        input_dir: Input directory containing audio files and JSON metadata.
        output_dir: Output directory for the COCO dataset.
        spec_config: Spectrogram configuration. Defaults to normalized frequency mode.
            Set target_sr for resampling mode, or use normalize_frequency=True.
        split_ratios: Train/valid/test split ratios.
        frequency_bins: Optional frequency bins for frequency-aware categories.
            Example: [(0, 500, "low"), (500, 5000, "mid"), (5000, 22050, "high")]
            If None and auto_infer_bins=False, no frequency-aware categories.
        auto_infer_bins: HYPERPARAMETER - Automatically infer frequency bins from data.
            - False: Use manual frequency_bins or no bins
            - True: Infer 3 bins automatically
            - int: Infer N bins (tune this based on dataset size and diversity)
            Optimal n_bins depends on:
            - Dataset size (more bins = more categories = need more samples)
            - Frequency diversity (narrow vs broad-band events)
            - Task complexity (fine-grained vs coarse classification)
            Recommended starting values: 2-5 bins for most datasets.
        audio_extensions: Audio file extensions to process.
        random_seed: Random seed for reproducible splits.

    Returns:
        Path to output directory.

    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if spec_config is None:
        spec_config = SpectrogramConfig()

    # Collect all audio files first
    audio_files = list(iterate_audio_dataset(input_dir, audio_extensions))

    if not audio_files:
        raise ValueError(f"No audio files with JSON metadata found in {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files to process")

    # Auto-infer frequency bins if requested
    if auto_infer_bins:
        n_bins = 3 if auto_infer_bins is True else int(auto_infer_bins)
        frequency_bins = infer_frequency_bins_from_data(audio_files, n_bins=n_bins)

    # Initialize category registry
    category_registry = CategoryRegistry(frequency_bins=frequency_bins)

    # Shuffle for random split
    if random_seed is not None:
        import random

        random.seed(random_seed)
        random.shuffle(audio_files)

    # Split into train/valid/test
    n_total = len(audio_files)
    n_train = int(n_total * split_ratios[0])
    n_valid = int(n_total * split_ratios[1])

    splits = {
        "train": audio_files[:n_train],
        "valid": audio_files[n_train : n_train + n_valid],
        "test": audio_files[n_train + n_valid :],
    }

    # Process each split
    for split_name, split_files in splits.items():
        if not split_files:
            continue

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_dataset = COCODataset()

        for audio_path, json_path in split_files:
            try:
                # Load metadata
                metadata = AudioMetadata.from_json(json_path)

                # Process audio to spectrogram
                img, bbox, extra = process_audio_file(audio_path, metadata, spec_config, category_registry)

                # Save spectrogram image
                img_filename = f"{metadata.uuid}.png"
                img_path = split_dir / img_filename
                img.save(img_path)

                # Add to COCO dataset
                image_id = coco_dataset.add_image(
                    file_name=img_filename,
                    width=img.width,
                    height=img.height,
                    extra={"audio_metadata": extra},
                )
                coco_dataset.add_annotation(bbox, image_id)

                logger.debug(f"Processed {audio_path.name} -> {img_filename}")

            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue

        # Finalize and save
        coco_dataset.set_categories(category_registry)
        coco_dataset.save(split_dir / "_annotations.coco.json")

        logger.info(f"{split_name}: {len(split_files)} samples -> {split_dir}")

    return output_dir


# =============================================================================
# CLI Integration
# =============================================================================


def add_audio_cli_args(parser) -> None:
    """Add audio conversion arguments to an argparse parser."""
    audio_group = parser.add_argument_group("Audio Processing")

    audio_group.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT window size (default: 2048)",
    )
    audio_group.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for STFT (default: 512)",
    )
    audio_group.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel filterbanks (default: 128)",
    )
    audio_group.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="Minimum frequency for mel filterbank (default: 0.0)",
    )
    audio_group.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency for mel filterbank (default: sr/2)",
    )
    audio_group.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Target sample rate for resampling (default: keep original)",
    )
    audio_group.add_argument(
        "--frequency-bins",
        type=str,
        default=None,
        help="Frequency bins for category splitting, format: 'min1,max1,name1;min2,max2,name2'",
    )


def parse_frequency_bins(bins_str: str | None) -> list[tuple[float, float, str]] | None:
    """Parse frequency bins from CLI string."""
    if not bins_str:
        return None

    bins = []
    for bin_spec in bins_str.split(";"):
        parts = bin_spec.split(",")
        if len(parts) == 3:
            bins.append((float(parts[0]), float(parts[1]), parts[2].strip()))

    return bins if bins else None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Convert audio dataset to COCO format for RF-DETR")
    parser.add_argument("input_dir", help="Input directory with audio + JSON files")
    parser.add_argument("output_dir", help="Output directory for COCO dataset")
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.7,0.2,0.1",
        help="Train,valid,test split ratios (default: 0.7,0.2,0.1)",
    )
    add_audio_cli_args(parser)

    args = parser.parse_args()

    # Parse split ratios
    split_ratios = tuple(float(x) for x in args.split_ratios.split(","))

    # Build spectrogram config
    spec_config = SpectrogramConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        target_sr=args.target_sr,
    )

    # Parse frequency bins
    frequency_bins = parse_frequency_bins(args.frequency_bins)

    # Run conversion
    output_path = convert_audio_to_coco(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        spec_config=spec_config,
        split_ratios=split_ratios,
        frequency_bins=frequency_bins,
    )

    print(f"Dataset created at: {output_path}")
