"""Audio and spectrogram augmentation module.

Provides data augmentation for training robustness.

"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def random_pad_position(
    audio: np.ndarray,
    target_length: int,
    padding_mode: str = "zero",
) -> np.ndarray:
    """Pad audio with random position (augmentation for short audio).

    Instead of always padding at the end, randomize where the audio
    sits in the padded output. Useful for very short sounds like
    gunshots (50ms) to improve model robustness.

    Args:
        audio: Audio array (1D)
        target_length: Target length in samples
        padding_mode: Padding mode ('zero', 'reflect')

    Returns:
        Padded audio with random position

    """
    current_length = len(audio)

    if current_length >= target_length:
        return audio[:target_length]

    padding_needed = target_length - current_length
    pad_before = random.randint(0, padding_needed)
    pad_after = padding_needed - pad_before

    if padding_mode == "zero":
        return np.pad(audio, (pad_before, pad_after), mode="constant", constant_values=0)
    elif padding_mode == "reflect" and len(audio) > 1:
        return np.pad(audio, (pad_before, pad_after), mode="reflect")
    else:
        return np.pad(audio, (pad_before, pad_after), mode="constant", constant_values=0)


def time_shift(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift audio in time (circular shift).

    Args:
        audio: Audio array (1D)
        shift_samples: Number of samples to shift (positive = right)

    Returns:
        Shifted audio

    """
    return np.roll(audio, shift_samples)


def random_time_shift(audio: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
    """Apply random time shift to audio.

    Args:
        audio: Audio array (1D)
        max_shift_ratio: Maximum shift as ratio of length

    Returns:
        Shifted audio

    """
    max_shift = int(len(audio) * max_shift_ratio)
    if max_shift == 0:
        return audio
    shift = random.randint(-max_shift, max_shift)
    return time_shift(audio, shift)


def add_noise(
    audio: np.ndarray,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Add Gaussian noise to audio.

    Args:
        audio: Audio array (1D)
        noise_level: Standard deviation of noise (relative to audio RMS)

    Returns:
        Noisy audio

    """
    rms = np.sqrt(np.mean(audio**2))
    noise = np.random.normal(0, noise_level * rms, len(audio))
    return audio + noise


def random_gain(
    audio: np.ndarray,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
) -> np.ndarray:
    """Apply random gain to audio.

    Args:
        audio: Audio array (1D)
        min_gain_db: Minimum gain in dB
        max_gain_db: Maximum gain in dB

    Returns:
        Audio with random gain applied

    """
    gain_db = random.uniform(min_gain_db, max_gain_db)
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def spec_augment(
    spec: np.ndarray,
    freq_mask_param: int = 10,
    time_mask_param: int = 10,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
) -> np.ndarray:
    """Apply SpecAugment to spectrogram.

    SpecAugment: A Simple Data Augmentation Method for ASR
    https://arxiv.org/abs/1904.08779

    Args:
        spec: Spectrogram (H, W)
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        n_freq_masks: Number of frequency masks
        n_time_masks: Number of time masks

    Returns:
        Augmented spectrogram

    """
    spec = spec.copy()
    n_freq, n_time = spec.shape

    # Frequency masks
    for _ in range(n_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_freq))
        f0 = random.randint(0, n_freq - f)
        spec[f0 : f0 + f, :] = 0

    # Time masks
    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_param, n_time))
        t0 = random.randint(0, n_time - t)
        spec[:, t0 : t0 + t] = 0

    return spec


def mixup_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Mix two audio samples (mixup augmentation).

    Args:
        audio1: First audio array
        audio2: Second audio array
        alpha: Mixing coefficient (0.5 = equal mix)

    Returns:
        Mixed audio

    """
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    return alpha * audio1 + (1 - alpha) * audio2
