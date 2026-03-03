"""Audio preprocessing module using ezakodio.transforms.

All preprocessing MUST use ezakodio.transforms functions:
- apply_agc
- compute_rms_db
- detrend
- preemphasis

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

    Attributes:
        enabled: Whether to apply AGC
        target_db: Target RMS level in dB (-30 to -15 typical)
        max_gain_db: Maximum gain boost
        min_gain_db: Maximum gain reduction (negative)
        frame_size_s: Frame size in seconds for percentile RMS
        percentile: Target percentile (0.5 = median)

    """

    enabled: bool = True
    target_db: float = -25.0
    max_gain_db: float = 30.0
    min_gain_db: float = -20.0
    frame_size_s: float = 0.1
    percentile: float = 0.5

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
    """Dynamic range compression configuration for spectrograms.

    Attributes:
        top_db: Maximum dynamic range in dB
        ref_db: Reference level in dB
        clip_percentile: Percentile for adaptive clipping (0-100)

    """

    top_db: float = 80.0
    ref_db: float = 0.0
    clip_percentile: float | None = 99.0

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

    Attributes:
        agc: AGC configuration
        dynamic_range: Dynamic range configuration
        detrend: Whether to remove DC offset
        preemphasis: Pre-emphasis coefficient (0 = disabled)

    """

    agc: AGCConfig = field(default_factory=AGCConfig)
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    detrend: bool = True
    preemphasis: float = 0.0

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
        """Load config from YAML file."""
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


def _to_tensor(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert audio to torch tensor if needed."""
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio).float()
    return audio.float()


def _to_numpy(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert audio to numpy array if needed."""
    if isinstance(audio, torch.Tensor):
        return audio.cpu().numpy()
    return audio


def apply_detrend(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio using ezakodio.transforms.detrend.

    Args:
        audio: Audio waveform (1D)

    Returns:
        Audio with DC offset removed

    """
    audio_t = _to_tensor(audio)
    out = detrend(audio_t, mode="constant")
    return _to_numpy(out)


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter using ezakodio.transforms.preemphasis.

    Args:
        audio: Audio waveform (1D)
        coef: Pre-emphasis coefficient (0.95-0.97 typical)

    Returns:
        Pre-emphasized audio

    """
    if coef <= 0:
        return audio

    audio_t = _to_tensor(audio)
    out = preemphasis(audio_t, coef=coef)
    return _to_numpy(out)


def apply_agc(
    audio: np.ndarray,
    sample_rate: int,
    config: AGCConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Apply Automatic Gain Control using ezakodio.transforms.apply_agc.

    Args:
        audio: Audio waveform (1D numpy array)
        sample_rate: Sample rate in Hz
        config: AGC configuration

    Returns:
        Tuple of (processed_audio, gain_applied_db)

    """
    if config is None:
        config = AGCConfig()

    if not config.enabled:
        return audio, 0.0

    # Compute original RMS
    audio_t = _to_tensor(audio)
    original_db = compute_rms_db(audio_t)

    # Apply ezakodio's AGC
    audio_out_t = ezakodio_apply_agc(
        audio_t,
        sample_rate=sample_rate,
        target_db=config.target_db,
        max_gain_db=config.max_gain_db,
        soft_clip=True,
        mode="percentile",
        apply_if_below_db=config.target_db - 10.0,
        skip_if_above_db=config.target_db + 10.0,
    )

    audio_out = _to_numpy(audio_out_t)

    # Compute gain applied
    final_db = compute_rms_db(_to_tensor(audio_out))
    gain_db = float(final_db - original_db)
    gain_db = np.clip(gain_db, config.min_gain_db, config.max_gain_db)

    return audio_out, gain_db


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply full preprocessing pipeline to audio.

    Pipeline order:
    1. Detrend (DC offset removal)
    2. Pre-emphasis (if enabled)
    3. AGC (Automatic Gain Control)

    Args:
        audio: Audio waveform (1D numpy array)
        sample_rate: Sample rate in Hz
        config: Preprocessing configuration

    Returns:
        Tuple of (processed_audio, metadata_dict)

    """
    if config is None:
        config = PreprocessingConfig()

    metadata = {
        "original_rms_db": float(compute_rms_db(_to_tensor(audio))),
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

    metadata["final_rms_db"] = float(compute_rms_db(_to_tensor(audio)))

    return audio, metadata


def apply_dynamic_range_compression(
    spectrogram: np.ndarray,
    config: DynamicRangeConfig | None = None,
) -> np.ndarray:
    """Apply dynamic range compression to spectrogram.

    Args:
        spectrogram: Spectrogram array (H, W), expected in dB scale
        config: Dynamic range configuration

    Returns:
        Compressed spectrogram

    """
    if config is None:
        config = DynamicRangeConfig()

    spec = spectrogram.copy()

    # Percentile clipping (prevents burning from outliers)
    if config.clip_percentile is not None and config.clip_percentile < 100:
        clip_value = np.percentile(spec, config.clip_percentile)
        spec = np.clip(spec, None, clip_value)

    # Top_db limiting
    if config.top_db is not None:
        spec_max = spec.max()
        spec = np.clip(spec, spec_max - config.top_db, None)

    return spec


def normalize_spectrogram(
    spectrogram: np.ndarray,
    config: DynamicRangeConfig | None = None,
) -> np.ndarray:
    """Normalize spectrogram to 0-255 range with dynamic range control.

    Args:
        spectrogram: Spectrogram array (H, W)
        config: Dynamic range configuration

    Returns:
        Normalized spectrogram as uint8 (0-255)

    """
    from rf_detr_finetuning.dataprocessor.normalization import normalize_to_range

    spec = apply_dynamic_range_compression(spectrogram, config)

    # Min-max normalization to 0-255
    return normalize_to_range(spec, 0.0, 255.0).astype(np.uint8)


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
        return float(compute_rms_db(_to_tensor(audio_np), eps=eps))

    # Compute frame-level RMS with 50% overlap
    hop = frame_size // 2
    n_frames = (len(audio_np) - frame_size) // hop + 1

    if n_frames <= 0:
        return float(compute_rms_db(_to_tensor(audio_np), eps=eps))

    rms_values = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_size
        frame = audio_np[start:end]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    rms_arr = np.array(rms_values)
    percentile_rms = np.percentile(rms_arr, percentile * 100)

    return float(20 * np.log10(percentile_rms + eps))
