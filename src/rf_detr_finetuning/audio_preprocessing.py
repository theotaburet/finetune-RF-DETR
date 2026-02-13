"""Audio preprocessing utilities for spectrogram generation.

This module provides preprocessing functions for audio signals before
spectrogram computation. It can be used both during training (chunking)
and inference (prediction).

Key features:
- AGC (Automatic Gain Control) for consistent spectrogram brightness
- Configurable via YAML or PreprocessingConfig dataclass
- Designed to prevent "burned" spectrograms from overly loud signals

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ezakodio.transforms import (
    apply_agc as ezakodio_apply_agc,
)
from ezakodio.transforms import (
    compute_rms_db,
    detrend,
    preemphasis,
)

logger = logging.getLogger(__name__)


@dataclass
class AGCConfig:
    """Automatic Gain Control configuration.

    AGC normalizes audio loudness to ensure consistent spectrogram appearance.
    Uses percentile-based RMS for robustness against transients.

    Attributes:
        enabled: Whether to apply AGC.
        target_db: Target RMS level in dB. Lower = quieter spectrogram.
            - -30 to -25 dB: Quiet, good for very loud sources
            - -25 to -20 dB: Moderate (recommended default)
            - -20 to -15 dB: Loud, good for quiet sources
        max_gain_db: Maximum gain boost (prevents extreme amplification of quiet signals).
        min_gain_db: Maximum gain reduction (prevents over-attenuation of loud signals).
        frame_size_s: Frame size in seconds for percentile RMS computation.
        percentile: Target percentile for RMS (0.5 = median, robust to outliers).

    """

    enabled: bool = True
    target_db: float = -25.0  # Conservative default to prevent burning
    max_gain_db: float = 30.0
    min_gain_db: float = -20.0
    frame_size_s: float = 0.1
    percentile: float = 0.5  # Median - robust to transients

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AGCConfig:
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            target_db=float(data.get("target_db", -25.0)),
            max_gain_db=float(data.get("max_gain_db", 30.0)),
            min_gain_db=float(data.get("min_gain_db", -20.0)),
            frame_size_s=float(data.get("frame_size_s", 0.1)),
            percentile=float(data.get("percentile", 0.5)),
        )


@dataclass
class DynamicRangeConfig:
    """Dynamic range compression configuration.

    Controls how the spectrogram values are mapped to the display range.
    Helps prevent burned (clipped) or too dark spectrograms.

    Attributes:
        top_db: Maximum dynamic range in dB. Values below max - top_db are clipped.
            - 60-80 dB: Standard for most audio
            - 40-60 dB: More compressed, shows quieter details
            - 80-100 dB: Full dynamic range, may appear dark
        ref_db: Reference level in dB for normalization.
        clip_percentile: Percentile for adaptive clipping (0-100).
            If set, clips values above this percentile to prevent burning.
            None = no percentile clipping.

    """

    top_db: float = 80.0
    ref_db: float = 0.0
    clip_percentile: float | None = 99.0  # Clip top 1% to prevent burning

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicRangeConfig:
        """Create config from dictionary."""
        return cls(
            top_db=float(data.get("top_db", 80.0)),
            ref_db=float(data.get("ref_db", 0.0)),
            clip_percentile=data.get("clip_percentile", 99.0),
        )


@dataclass
class PreprocessingConfig:
    """Complete audio preprocessing configuration.

    Combines AGC and dynamic range settings for consistent spectrogram generation.

    Attributes:
        agc: AGC configuration.
        dynamic_range: Dynamic range configuration.
        detrend: Whether to remove DC offset before processing.
        preemphasis: Pre-emphasis coefficient (0 = disabled, 0.97 = typical).

    """

    agc: AGCConfig = field(default_factory=AGCConfig)
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    detrend: bool = True
    preemphasis: float = 0.0  # Disabled by default, use 0.97 for speech

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingConfig:
        """Create config from dictionary."""
        agc_data = data.get("agc", {})
        dr_data = data.get("dynamic_range", {})

        return cls(
            agc=AGCConfig.from_dict(agc_data) if agc_data else AGCConfig(),
            dynamic_range=DynamicRangeConfig.from_dict(dr_data) if dr_data else DynamicRangeConfig(),
            detrend=data.get("detrend", True),
            preemphasis=float(data.get("preemphasis", 0.0)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> PreprocessingConfig:
        """Load config from YAML file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            PreprocessingConfig instance.

        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        preprocessing_data = data.get("preprocessing", {})
        return cls.from_dict(preprocessing_data)

    def to_dict(self) -> dict[str, Any]:
        """Export config to dictionary."""
        return {
            "agc": {
                "enabled": self.agc.enabled,
                "target_db": self.agc.target_db,
                "max_gain_db": self.agc.max_gain_db,
                "min_gain_db": self.agc.min_gain_db,
                "frame_size_s": self.agc.frame_size_s,
                "percentile": self.agc.percentile,
            },
            "dynamic_range": {
                "top_db": self.dynamic_range.top_db,
                "ref_db": self.dynamic_range.ref_db,
                "clip_percentile": self.dynamic_range.clip_percentile,
            },
            "detrend": self.detrend,
            "preemphasis": self.preemphasis,
        }


# ---------------------------------------------------------------------------
# AGC Functions (using ezakodio.transforms)
# ---------------------------------------------------------------------------


def _compute_rms_db_wrapper(audio: np.ndarray | torch.Tensor, eps: float = 1e-8) -> float:
    """Compute RMS level in dB (wrapper for ezakodio.transforms.compute_rms_db).

    Args:
        audio: Audio waveform (1D array or tensor).
        eps: Small value to avoid log(0).

    Returns:
        RMS level in dB.

    """
    # Convert to torch if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    return compute_rms_db(audio, eps=eps)


def compute_percentile_rms_db(
    audio: np.ndarray | torch.Tensor,
    sample_rate: int,
    frame_size_s: float = 0.1,
    percentile: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute percentile-based RMS level in dB.

    More robust than global RMS for audio with loud transients or silence.

    Args:
        audio: Audio waveform (1D).
        sample_rate: Sample rate in Hz.
        frame_size_s: Frame size in seconds.
        percentile: Target percentile (0.5 = median).
        eps: Small value to avoid log(0).

    Returns:
        Percentile RMS level in dB.

    """
    frame_size = int(frame_size_s * sample_rate)

    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy().flatten()
    else:
        audio_np = audio.flatten()

    if len(audio_np) < frame_size:
        # Audio too short, use global RMS
        return _compute_rms_db_wrapper(audio_np, eps)

    # Compute frame-level RMS with 50% overlap
    hop = frame_size // 2
    n_frames = (len(audio_np) - frame_size) // hop + 1

    if n_frames <= 0:
        return _compute_rms_db_wrapper(audio_np, eps)

    rms_values = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_size
        frame = audio_np[start:end]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    percentile_rms = np.percentile(rms_values, percentile * 100)

    return 20 * np.log10(percentile_rms + eps)


def apply_agc(
    audio: np.ndarray,
    sample_rate: int,
    config: AGCConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Apply Automatic Gain Control to audio (uses ezakodio.transforms.apply_agc).

    Normalizes audio to a target RMS level using percentile-based estimation.
    Returns the processed audio and the gain applied.

    Args:
        audio: Audio waveform (1D numpy array).
        sample_rate: Sample rate in Hz.
        config: AGC configuration. If None, uses defaults.

    Returns:
        Tuple of (processed_audio, gain_applied_db).

    """
    if config is None:
        config = AGCConfig()

    if not config.enabled:
        return audio, 0.0

    # Compute original RMS for gain calculation
    original_db = _compute_rms_db_wrapper(audio)

    # Convert to torch tensor
    audio_t = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio

    # Apply ezakodio's AGC with percentile mode
    # Good parameters for audio event detection:
    # - mode='percentile': Robust to transients and silence
    # - target_db=-25.0: Conservative to prevent burning
    # - max_gain_db=30.0: Reasonable boost limit
    # - soft_clip=True: Prevents hard clipping artifacts
    # - apply_if_below_db=-35.0: Skip very quiet signals (likely noise)
    # - skip_if_above_db=-15.0: Skip already loud signals
    audio_out_t = ezakodio_apply_agc(
        audio_t,
        sample_rate=sample_rate,
        target_db=config.target_db,
        max_gain_db=config.max_gain_db,
        soft_clip=True,  # Prevent hard clipping
        mode="percentile",  # Use percentile-based RMS (robust to transients)
        apply_if_below_db=config.target_db - 10.0,  # Skip if already close to target
        skip_if_above_db=config.target_db + 10.0,  # Skip if too loud
    )

    # Convert back to numpy
    if hasattr(audio_out_t, "cpu"):
        audio_out_t = audio_out_t.cpu()
    audio_out = audio_out_t.numpy()

    # Compute gain applied
    final_db = _compute_rms_db_wrapper(audio_out)
    gain_db = final_db - original_db

    # Clamp reported gain to config limits for consistency
    gain_db = np.clip(gain_db, config.min_gain_db, config.max_gain_db)

    return audio_out, gain_db


def apply_detrend(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio (uses ezakodio.transforms.detrend).

    Args:
        audio: Audio waveform (1D).

    Returns:
        Audio with DC offset removed.

    """
    audio_t = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    out = detrend(audio_t, mode="constant")
    if hasattr(out, "cpu"):
        out = out.cpu()
    return out.numpy()


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio (uses ezakodio.transforms.preemphasis).

    Emphasizes high frequencies. Useful for speech processing.
    Formula: y[n] = x[n] - coef * x[n-1]

    Args:
        audio: Audio waveform (1D).
        coef: Pre-emphasis coefficient (0.95-0.97 typical).

    Returns:
        Pre-emphasized audio.

    """
    if coef <= 0:
        return audio

    audio_t = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    out = preemphasis(audio_t, coef=coef)
    if hasattr(out, "cpu"):
        out = out.cpu()
    return out.numpy()


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply full preprocessing pipeline to audio.

    Applies in order:
    1. Detrend (DC offset removal)
    2. Pre-emphasis (if enabled)
    3. AGC (Automatic Gain Control)

    Args:
        audio: Audio waveform (1D numpy array).
        sample_rate: Sample rate in Hz.
        config: Preprocessing configuration. If None, uses defaults.

    Returns:
        Tuple of (processed_audio, metadata_dict).
        Metadata includes original_rms_db, gain_applied_db, etc.

    """
    if config is None:
        config = PreprocessingConfig()

    metadata = {
        "original_rms_db": _compute_rms_db_wrapper(audio),
        "detrend_applied": False,
        "preemphasis_applied": False,
        "agc_applied": False,
        "gain_applied_db": 0.0,
    }

    # 1. Detrend
    if config.detrend:
        audio = apply_detrend(audio)
        metadata["detrend_applied"] = True

    # 2. Pre-emphasis
    if config.preemphasis > 0:
        audio = apply_preemphasis(audio, config.preemphasis)
        metadata["preemphasis_applied"] = True

    # 3. AGC
    if config.agc.enabled:
        audio, gain_db = apply_agc(audio, sample_rate, config.agc)
        metadata["agc_applied"] = True
        metadata["gain_applied_db"] = gain_db

    metadata["final_rms_db"] = _compute_rms_db_wrapper(audio)

    return audio, metadata


# ---------------------------------------------------------------------------
# Spectrogram Post-processing
# ---------------------------------------------------------------------------


def apply_dynamic_range_compression(
    spectrogram: np.ndarray,
    config: DynamicRangeConfig | None = None,
) -> np.ndarray:
    """Apply dynamic range compression to spectrogram.

    Prevents burning by limiting dynamic range and optionally clipping
    extreme values based on percentile.

    Args:
        spectrogram: Spectrogram array (H, W), expected in dB scale.
        config: Dynamic range configuration.

    Returns:
        Compressed spectrogram.

    """
    if config is None:
        config = DynamicRangeConfig()

    spec = spectrogram.copy()

    # Apply percentile clipping first (prevents burning from outliers)
    if config.clip_percentile is not None and config.clip_percentile < 100:
        clip_value = np.percentile(spec, config.clip_percentile)
        spec = np.clip(spec, None, clip_value)

    # Apply top_db limiting
    if config.top_db is not None:
        spec_max = spec.max()
        spec = np.clip(spec, spec_max - config.top_db, None)

    return spec


def normalize_spectrogram(
    spectrogram: np.ndarray,
    config: DynamicRangeConfig | None = None,
) -> np.ndarray:
    """Normalize spectrogram to 0-255 range with dynamic range control.

    Combines dynamic range compression with min-max normalization.

    Args:
        spectrogram: Spectrogram array (H, W).
        config: Dynamic range configuration.

    Returns:
        Normalized spectrogram as uint8 (0-255).

    """
    spec = apply_dynamic_range_compression(spectrogram, config)

    # Min-max normalization to 0-255
    spec_min = spec.min()
    spec_max = spec.max()

    if spec_max > spec_min:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)

    return (spec * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


class AudioPreprocessor:
    """High-level audio preprocessing interface.

    Use this class for consistent preprocessing across training and inference.

    Example:
        >>> preprocessor = AudioPreprocessor.from_yaml("config/chunking.yaml")
        >>> audio_processed, metadata = preprocessor.process(audio, sample_rate)
        >>> spec_normalized = preprocessor.normalize_spectrogram(spectrogram)

    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration. If None, uses defaults.

        """
        self.config = config or PreprocessingConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> AudioPreprocessor:
        """Create preprocessor from YAML config file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            AudioPreprocessor instance.

        """
        config = PreprocessingConfig.from_yaml(yaml_path)
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AudioPreprocessor:
        """Create preprocessor from config dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            AudioPreprocessor instance.

        """
        config = PreprocessingConfig.from_dict(config_dict)
        return cls(config)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process audio with full preprocessing pipeline.

        Args:
            audio: Audio waveform (1D numpy array).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (processed_audio, metadata_dict).

        """
        return preprocess_audio(audio, sample_rate, self.config)

    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize spectrogram with dynamic range control.

        Args:
            spectrogram: Spectrogram array (H, W).

        Returns:
            Normalized spectrogram as uint8 (0-255).

        """
        return normalize_spectrogram(spectrogram, self.config.dynamic_range)

    def __repr__(self) -> str:
        """Return a concise string representation of the instance."""  # FIXME
        agc_status = "enabled" if self.config.agc.enabled else "disabled"
        return (
            f"AudioPreprocessor(agc={agc_status}, "
            f"target_db={self.config.agc.target_db:.1f}, "
            f"top_db={self.config.dynamic_range.top_db:.0f})"
        )


# ---------------------------------------------------------------------------
# Convenience functions for inference
# ---------------------------------------------------------------------------


def preprocess_for_inference(
    audio: np.ndarray,
    sample_rate: int,
    config_path: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess audio for inference using config file.

    Convenience function for inference pipelines.

    Args:
        audio: Audio waveform (1D numpy array).
        sample_rate: Sample rate in Hz.
        config_path: Path to config YAML. If None, uses defaults.

    Returns:
        Tuple of (processed_audio, metadata_dict).

    """
    if config_path:
        preprocessor = AudioPreprocessor.from_yaml(config_path)
    else:
        preprocessor = AudioPreprocessor()

    return preprocessor.process(audio, sample_rate)
