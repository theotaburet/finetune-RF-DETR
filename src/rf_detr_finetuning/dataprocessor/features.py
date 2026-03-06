"""Feature extraction module using ezakodio.

All spectrogram computation MUST use ezakodio.dsp.mel_spectrogram.

"""

from __future__ import annotations

import logging

import numpy as np
import torch
from ezakodio.dsp import mel_spectrogram

logger = logging.getLogger(__name__)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int | None = None,
    hop_length: int | None = None,
    f_min: float = 0.0,
    f_max: float | None = None,
    power: float = 2.0,
    device: str = "cpu",
) -> np.ndarray:
    """Compute mel spectrogram using ezakodio.

    Args:
        audio: Audio waveform as 1D numpy array
        sample_rate: Sample rate in Hz
        n_mels: Number of mel filterbanks
        n_fft: FFT window size (None = auto from sample_rate)
        hop_length: Hop length in samples (None = auto)
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency (None = sr/2)
        power: Exponent for magnitude spectrogram (1=amplitude, 2=power)
        device: Device for computation ("cpu" or "cuda")

    Returns:
        Mel spectrogram as 2D numpy array (n_mels, n_frames)

    """
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

    # Set defaults if not provided
    if n_fft is None:
        n_fft = int(sample_rate * 0.025)  # 25ms default
    if hop_length is None:
        hop_length = int(sample_rate * 0.010)  # 10ms default
    if f_max is None:
        f_max = sample_rate / 2

    # Ensure minimum values
    n_fft = max(n_fft, 1)
    hop_length = max(hop_length, 1)

    # Pad audio if too short for FFT window
    audio_length = len(audio)
    if audio_length < n_fft:
        padding_needed = n_fft - audio_length
        audio = np.pad(audio, (0, padding_needed), mode="constant", constant_values=0.0)
        logger.debug(f"Padded audio: {audio_length} -> {len(audio)} samples for n_fft={n_fft}")
        # Recreate tensor from padded audio
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

    try:
        spec = mel_spectrogram(
            audio_tensor,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=power,
            device=device,
            log_scale=False,  # Return raw power spectrum; dB conversion is handled by compute_mel_spectrogram_db()
        )

        # Convert to numpy
        if hasattr(spec, "cpu"):
            spec = spec.cpu().numpy()

        # Remove batch dimension if present
        if spec.ndim == 3:
            spec = spec.squeeze(0)

        return spec

    except Exception as e:
        logger.error(f"Failed to compute mel spectrogram: {e}")
        raise


def compute_mel_spectrogram_db(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int | None = None,
    hop_length: int | None = None,
    f_min: float = 0.0,
    f_max: float | None = None,
    ref_db: float = 0.0,
    min_db: float = -80.0,
    device: str = "cpu",
) -> np.ndarray:
    """Compute mel spectrogram in dB scale.

    Args:
        audio: Audio waveform as 1D numpy array
        sample_rate: Sample rate in Hz
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length in samples
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency
        ref_db: Reference dB level
        min_db: Minimum dB level (values below are clipped)
        device: Device for computation

    Returns:
        Mel spectrogram in dB scale as 2D numpy array

    """
    spec = compute_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        device=device,
    )

    # Convert to dB scale
    eps = 1e-10
    spec_db = 10.0 * np.log10(np.maximum(spec, eps))

    # Apply reference and floor
    spec_db = spec_db - ref_db
    spec_db = np.maximum(spec_db, min_db)

    return spec_db


def flip_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Flip spectrogram vertically so high frequencies are at top.

    Standard visualization convention: high frequencies at y=0 (top).

    Args:
        spec: Spectrogram array (H, W)

    Returns:
        Vertically flipped spectrogram

    """
    return np.flipud(spec)
