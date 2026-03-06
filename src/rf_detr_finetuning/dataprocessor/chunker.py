"""High-level audio chunker that integrates all dataprocessor components.

This module provides the main AudioChunker class that orchestrates:
- Audio loading
- Preprocessing
- Feature extraction
- Chunking
- Bbox alignment

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rf_detr_finetuning.dataprocessor.chunking import (
    AudioChunk,
    ChunkConfig,
    TimeBasedFFTConfig,
    align_bbox_to_chunk,
    compute_chunk_boundaries,
    extract_audio_chunk,
)
from rf_detr_finetuning.dataprocessor.features import compute_mel_spectrogram_db, flip_spectrogram
from rf_detr_finetuning.dataprocessor.io import load_audio_file
from rf_detr_finetuning.dataprocessor.normalization import resize_spectrogram, spectrogram_to_image_array
from rf_detr_finetuning.dataprocessor.preprocessing import (
    PreprocessingConfig,
    normalize_spectrogram,
    preprocess_audio,
)

logger = logging.getLogger(__name__)


class AudioChunker:
    """Main class for chunking audio files into fixed-size spectrograms.

    Integrates all dataprocessor components:
    - Audio loading (io)
    - Preprocessing (AGC, detrend)
    - Feature extraction (mel spectrogram)
    - Chunking/windowing
    - Bbox alignment
    - Normalization

    Example:
        >>> chunker = AudioChunker(fft_config, chunk_config)
        >>> chunks = chunker.chunk_audio_file(audio_path, metadata_path)
        >>> for chunk in chunks:
        ...     save_image(chunk.spectrogram, f"{chunk.get_chunk_id()}.png")

    """

    def __init__(
        self,
        fft_config: TimeBasedFFTConfig | None = None,
        chunk_config: ChunkConfig | None = None,
        freq_scale: str = "mel",
        fmin: float = 0.0,
        fmax: float | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            fft_config: Time-based FFT configuration
            chunk_config: Chunk configuration
            freq_scale: Frequency scale ('mel', 'linear', 'log')
            fmin: Minimum frequency for spectrogram
            fmax: Maximum frequency (None = Nyquist)
            preprocessing_config: Audio preprocessing config

        """
        self.fft_config = fft_config or TimeBasedFFTConfig()
        self.chunk_config = chunk_config or ChunkConfig()
        self.freq_scale = freq_scale
        self.fmin = fmin
        self.fmax = fmax
        self.preprocessing_config = preprocessing_config

    def _resolve_fmax(self, sample_rate: int) -> float:
        """Resolve maximum frequency, defaulting to Nyquist if not set.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Maximum frequency in Hz.

        """
        return self.fmax if self.fmax else sample_rate / 2

    def _compute_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute mel spectrogram in dB scale with optional spectral normalization.

        Pipeline:
        1. Mel spectrogram (power=2.0) -> dB conversion (10*log10)
        2. Optional: per-frequency median subtraction (spectral whitening)
        3. Optional: robust std (MAD) normalization

        Args:
            audio: Audio array (1D)
            sample_rate: Sample rate

        Returns:
            Spectrogram as numpy array (H, W) in dB scale

        """
        n_fft = self.fft_config.get_n_fft(sample_rate)
        hop_length = self.fft_config.get_hop_length(sample_rate)
        fmax = self._resolve_fmax(sample_rate)

        spec = compute_mel_spectrogram_db(
            audio=audio,
            sample_rate=sample_rate,
            n_mels=self.fft_config.n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=self.fmin,
            f_max=fmax,
            min_db=-80.0,
        )

        # Per-frequency median subtraction (spectral whitening)
        if self.preprocessing_config is not None and self.preprocessing_config.spectral_whitening:
            median_profile = np.median(spec, axis=1, keepdims=True)
            spec = spec - median_profile

        # Robust std (MAD) normalization
        if self.preprocessing_config is not None and self.preprocessing_config.mad_normalization:
            mad = np.median(np.abs(spec - np.median(spec, axis=1, keepdims=True)), axis=1, keepdims=True)
            mad = np.maximum(mad, 1e-6)
            spec = spec / mad

        return spec

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        events: list[dict[str, Any]] | None = None,
        source_uuid: str = "",
    ) -> list[AudioChunk]:
        """Chunk audio into fixed-size segments with aligned bboxes.

        Args:
            audio: Audio array (1D)
            sample_rate: Sample rate
            events: List of event dicts with keys:
                - time_start_ms, time_end_ms
                - hz_min, hz_max
                - category, category_id
            source_uuid: Source file UUID

        Returns:
            List of AudioChunk objects

        """
        events = events or []
        total_duration_ms = (len(audio) / sample_rate) * 1000
        fmax = self._resolve_fmax(sample_rate)

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

            # Compute spectrogram
            spec = self._compute_spectrogram(chunk_audio, sample_rate)

            # Flip vertically (high frequencies at top)
            spec = flip_spectrogram(spec)

            # Apply dynamic range normalization if configured
            if self.preprocessing_config is not None:
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

            # Align bboxes to chunk
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
                    actual_width_px=actual_width,
                    use_mel_scale=(self.freq_scale == "mel"),
                )

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
            audio_path: Path to audio file
            metadata_path: Path to JSON metadata file (optional)

        Returns:
            List of AudioChunk objects

        """
        audio, sample_rate = load_audio_file(audio_path)

        # Pad short audio files to meet minimum FFT requirements
        n_fft = self.fft_config.get_n_fft(sample_rate)
        if len(audio) < n_fft:
            original_length = len(audio)
            padding_needed = n_fft - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode="constant", constant_values=0)
            logger.debug(
                f"Padded {audio_path.name}: {original_length} -> {len(audio)} samples "
                f"({original_length / sample_rate * 1000:.1f}ms -> {len(audio) / sample_rate * 1000:.1f}ms)"
            )

        # Apply preprocessing to FULL audio file (not per-chunk)
        if self.preprocessing_config is not None:
            audio, preprocess_metadata = preprocess_audio(audio, sample_rate, self.preprocessing_config)
            logger.debug(f"Preprocessed {audio_path.name}: gain={preprocess_metadata.get('gain_applied_db', 0):.2f}dB")

        events = []
        source_uuid = audio_path.stem
        fmax = self._resolve_fmax(sample_rate)
        actual_duration_ms = (len(audio) / sample_rate) * 1000

        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)

            source_uuid = meta.get("uuid", source_uuid)

            # Extract label from hierarchy or annotation
            hierarchy = meta.get("label_hierarchy", "")
            if isinstance(hierarchy, str) and hierarchy:
                if " + " in hierarchy:
                    label = hierarchy.split(" + ")[-1].strip()
                elif " > " in hierarchy:
                    label = hierarchy.split(" > ")[-1].strip()
                else:
                    label = hierarchy.strip()
            else:
                label = meta.get("annotation", "unknown")

            # Create file-level bbox spanning entire audio
            events.append(
                {
                    "time_start_ms": 0,
                    "time_end_ms": actual_duration_ms,
                    "hz_min": meta.get("hz_min", self.fmin),
                    "hz_max": meta.get("hz_max", fmax),
                    "category": label or "unknown",
                    "category_id": 0,
                    "is_file_level": True,
                }
            )

        return self.chunk_audio(audio, sample_rate, events, source_uuid)

    def process_file(
        self,
        audio_path: Path | str,
        events: list[dict[str, Any]] | None = None,
    ) -> list[AudioChunk]:
        """Process an audio file into chunks (inference-oriented convenience method).

        Unlike chunk_audio_file, this method does not read metadata sidecar
        files.  It accepts an optional list of events directly.

        Args:
            audio_path: Path to audio file.
            events: Optional pre-built event list. Defaults to empty (no annotations).

        Returns:
            List of AudioChunk objects.

        """
        audio_path = Path(audio_path)
        audio, sample_rate = load_audio_file(audio_path)

        # Pad short audio (same logic as chunk_audio_file)
        n_fft = self.fft_config.get_n_fft(sample_rate)
        if len(audio) < n_fft:
            padding_needed = n_fft - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode="constant", constant_values=0)

        # Apply preprocessing to full audio
        if self.preprocessing_config is not None:
            audio, _meta = preprocess_audio(audio, sample_rate, self.preprocessing_config)

        return self.chunk_audio(audio, sample_rate, events or [], source_uuid=audio_path.stem)


def load_chunking_config_from_yaml(
    yaml_path: Path | str,
) -> tuple[TimeBasedFFTConfig, ChunkConfig, PreprocessingConfig | None, dict]:
    """Load chunking configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Tuple of (TimeBasedFFTConfig, ChunkConfig, PreprocessingConfig or None,
        spectrogram_config dict with keys 'freq_scale', 'fmin', 'fmax')

    """
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    fft_section = config.get("fft", {})
    chunk_section = config.get("chunking", {})
    preprocessing_section = config.get("preprocessing", {})
    spectrogram_section = config.get("spectrogram", {})

    fft_config = TimeBasedFFTConfig(
        fft_ms=fft_section.get("fft_ms", 25.0),
        hop_ms=fft_section.get("hop_ms", 10.0),
        n_mels=fft_section.get("n_mels", 128),
    )

    # Support both target_size and explicit dimensions
    target_size = chunk_section.get("target_size")
    if target_size is not None:
        target_width = target_size
        target_height = target_size
        window_duration_ms = target_size * fft_config.hop_ms
    else:
        target_width = chunk_section.get("target_width", 640)
        target_height = chunk_section.get("target_height", 640)
        window_duration_ms = chunk_section.get("window_duration_ms", 5000.0)

    # Support both overlap_ratio and overlap_ms
    if "overlap_ratio" in chunk_section:
        overlap_ratio = chunk_section["overlap_ratio"]
    elif "overlap_ms" in chunk_section:
        overlap_ratio = chunk_section["overlap_ms"] / window_duration_ms
    else:
        overlap_ratio = 0.2

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
        preprocessing_config = PreprocessingConfig.from_dict(preprocessing_section)

    # Load spectrogram config (freq_scale, fmin, fmax)
    spectrogram_config = {
        "freq_scale": spectrogram_section.get("freq_scale", "mel"),
        "fmin": spectrogram_section.get("fmin", 0.0),
        "fmax": spectrogram_section.get("fmax", None),
    }

    return fft_config, chunk_config, preprocessing_config, spectrogram_config
