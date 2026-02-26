"""Audio I/O module using ezakodio.

All audio loading MUST use ezakodio.io.load_audio.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch
from ezakodio.io import load_audio

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_audio_file(
    audio_path: Path | str,
    mono: bool = True,
    target_sr: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, int]:
    """Load audio file using ezakodio.

    Args:
        audio_path: Path to audio file (.flac, .wav, .mp3, etc.)
        mono: Convert to mono if True
        target_sr: Target sample rate for resampling (None = keep original)
        device: Device for processing ("cpu" or "cuda")

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is 1D numpy array

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails

    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load using ezakodio
    audio_tensor, sample_rate = load_audio(
        str(audio_path),
        mono=mono,
        device=device,
    )

    # Resample if target_sr specified
    if target_sr is not None and target_sr != sample_rate:
        # TODO(ezakodio-missing): Check if ezakodio has resampling
        # For now, use torchaudio functional resample
        try:
            import torchaudio.functional as F

            audio_tensor = F.resample(audio_tensor, sample_rate, target_sr)
            sample_rate = target_sr
        except ImportError:
            raise NotImplementedError("Resampling requires torchaudio. Install with: pip install torchaudio")

    # Convert to numpy
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.cpu().numpy().flatten()
    else:
        audio = np.asarray(audio_tensor).flatten()

    return audio, sample_rate


def load_audio_segment(
    audio_path: Path | str,
    start_ms: float,
    end_ms: float,
    mono: bool = True,
    device: str = "cpu",
) -> tuple[np.ndarray, int]:
    """Load a segment of an audio file.

    Args:
        audio_path: Path to audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        mono: Convert to mono if True
        device: Device for processing

    Returns:
        Tuple of (audio_segment, sample_rate)

    """
    audio, sample_rate = load_audio_file(audio_path, mono=mono, device=device)

    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    return audio[start_sample:end_sample], sample_rate


def get_audio_duration_ms(audio_path: Path | str) -> float:
    """Get audio file duration in milliseconds from file header.

    Uses soundfile to read metadata without loading the full audio data,
    which is significantly faster for large files.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in milliseconds

    """
    info = sf.info(str(audio_path))
    return (info.frames / info.samplerate) * 1000


def filter_audio_by_duration(
    audio_files: list[Path],
    max_duration_s: float | None,
) -> list[Path]:
    """Filter audio files by maximum duration.

    Reads file headers (no full load) to determine duration and excludes
    files exceeding the threshold. Useful for fast iteration during testing.

    Args:
        audio_files: List of audio file paths.
        max_duration_s: Maximum allowed duration in seconds. None disables filtering.

    Returns:
        Filtered list of audio files that are within the duration limit.

    """
    if max_duration_s is None:
        return audio_files

    max_duration_ms = max_duration_s * 1000
    filtered = []
    skipped = 0

    for path in audio_files:
        try:
            duration_ms = get_audio_duration_ms(path)
            if duration_ms <= max_duration_ms:
                filtered.append(path)
            else:
                skipped += 1
                logger.debug(
                    "Skipping %s (%.1fs > %.1fs max)",
                    path.name,
                    duration_ms / 1000,
                    max_duration_s,
                )
        except Exception:
            logger.warning("Could not read duration for %s, keeping file", path.name)
            filtered.append(path)

    if skipped > 0:
        logger.info(
            "Duration filter: kept %d / %d files (skipped %d > %.0fs)",
            len(filtered),
            len(audio_files),
            skipped,
            max_duration_s,
        )

    return filtered
