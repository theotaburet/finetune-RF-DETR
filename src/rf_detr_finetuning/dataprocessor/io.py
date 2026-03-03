"""Audio I/O module using ezakodio.

All audio loading MUST use ezakodio.io.load_audio.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ezakodio.io import load_audio


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
