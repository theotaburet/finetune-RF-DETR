#!/usr/bin/env python3
"""Visual test script for audio preprocessing parameters.

Run this to experiment with AGC, dynamic range, and colormap settings.
Generates side-by-side comparisons of spectrograms with different parameters.

Usage:
    python scripts/test_preprocessing_visual.py --audio data/audio/example.wav --output output/preprocessing_test/

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ezakodio import load_audio
from ezakodio.dsp import mel_spectrogram

from rf_detr_finetuning.audio_preprocessing import (
    AGCConfig,
    AudioPreprocessor,
    DynamicRangeConfig,
    PreprocessingConfig,
)


def apply_colormap(spec_uint8: np.ndarray, colormap_name: str = "magma") -> np.ndarray:
    """Apply matplotlib colormap to grayscale spectrogram.

    Args:
        spec_uint8: Grayscale spectrogram (H, W) in 0-255 range.
        colormap_name: Colormap name (magma, viridis, inferno, plasma, etc.).

    Returns:
        RGB spectrogram (H, W, 3) in 0-255 range.

    """
    cmap = plt.get_cmap(colormap_name)
    # Normalize to 0-1 for colormap
    spec_norm = spec_uint8.astype(np.float32) / 255.0
    # Apply colormap (returns RGBA)
    spec_rgba = cmap(spec_norm)
    # Convert to RGB uint8
    spec_rgb = (spec_rgba[..., :3] * 255).astype(np.uint8)
    return spec_rgb


def generate_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    fft_ms: float = 25.0,
    hop_ms: float = 10.0,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Generate mel spectrogram in dB scale.

    Args:
        audio: Audio waveform.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel bands.
        fft_ms: FFT window size in ms.
        hop_ms: Hop size in ms.

    Returns:
        Mel spectrogram in dB scale (H, W).

    """
    n_fft = int(fft_ms * sample_rate / 1000)
    hop_length = int(hop_ms * sample_rate / 1000)

    # Generate mel spectrogram (returns power spectrogram)
    import torch

    audio_t = torch.from_numpy(audio).float()
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)

    if f_max is None:
        f_max = sample_rate // 2

    spec = mel_spectrogram(
        audio_t,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    # Convert to numpy (mel_spectrogram may return a torch tensor on GPU)
    if hasattr(spec, "cpu"):
        spec = spec.cpu().numpy()

    if spec.ndim == 3 and spec.shape[0] == 1:
        spec = spec.squeeze(0)

    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.maximum(spec, 0.0)

    # Convert to dB scale
    spec_db = 10 * np.log10(spec + 1e-10)

    # Flip vertically for standard visualization (high freq at top)
    spec_db = np.flipud(spec_db)

    return spec_db


def test_preprocessing_params(
    audio_path: Path,
    output_dir: Path,
    duration_s: float,
    n_mels: int,
    fft_ms: float,
    hop_ms: float,
    f_min: float,
    f_max: float | None,
):
    """Test different preprocessing parameters and save visual comparisons.

    Args:
        audio_path: Path to audio file.
        output_dir: Output directory for images.
        duration_s: Duration to load (seconds).
        n_mels: Number of mel bands.
        fft_ms: FFT window size in milliseconds.
        hop_ms: Hop size in milliseconds.
        f_min: Minimum frequency.
        f_max: Maximum frequency (or None).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sample_rate = load_audio(str(audio_path), duration=duration_s)
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio).flatten()
    print(f"  Sample rate: {sample_rate} Hz, Duration: {len(audio) / sample_rate:.2f}s")

    # Test configurations
    configs = {
        "no_preprocessing": PreprocessingConfig(
            agc=AGCConfig(enabled=False),
            dynamic_range=DynamicRangeConfig(top_db=80.0, clip_percentile=None),
            detrend=False,
        ),
        "agc_-30db": PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=-30.0),
            dynamic_range=DynamicRangeConfig(top_db=80.0, clip_percentile=99.0),
        ),
        "agc_-25db": PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=-25.0),
            dynamic_range=DynamicRangeConfig(top_db=80.0, clip_percentile=99.0),
        ),
        "agc_-20db": PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=-20.0),
            dynamic_range=DynamicRangeConfig(top_db=80.0, clip_percentile=99.0),
        ),
        "agc_-25db_top60": PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=-25.0),
            dynamic_range=DynamicRangeConfig(top_db=60.0, clip_percentile=99.0),
        ),
        "agc_-25db_clip95": PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=-25.0),
            dynamic_range=DynamicRangeConfig(top_db=80.0, clip_percentile=95.0),
        ),
    }

    # Colormap options
    colormaps = ["grayscale", "magma", "viridis", "inferno", "plasma"]

    # Process each configuration
    results = {}
    for config_name, config in configs.items():
        print(f"\nProcessing: {config_name}")
        preprocessor = AudioPreprocessor(config)

        # Preprocess audio
        audio_processed, metadata = preprocessor.process(audio, sample_rate)
        print(f"  Gain applied: {metadata.get('gain_applied_db', 0):.2f} dB")
        print(f"  Original RMS: {metadata['original_rms_db']:.2f} dB")
        print(f"  Final RMS: {metadata['final_rms_db']:.2f} dB")

        # Generate spectrogram
        spec_db = generate_spectrogram(
            audio_processed,
            sample_rate,
            n_mels=n_mels,
            fft_ms=fft_ms,
            hop_ms=hop_ms,
            f_min=f_min,
            f_max=f_max,
        )

        # Normalize
        spec_normalized = preprocessor.normalize_spectrogram(spec_db)

        results[config_name] = {
            "spec": spec_normalized,
            "metadata": metadata,
        }

    # Create comparison grid
    n_configs = len(configs)
    n_colormaps = len(colormaps)

    fig, axes = plt.subplots(n_configs, n_colormaps, figsize=(4 * n_colormaps, 3 * n_configs))
    fig.suptitle(f"Preprocessing Comparison: {audio_path.name}", fontsize=16, y=0.995)

    for row, (config_name, result) in enumerate(results.items()):
        spec = result["spec"]
        metadata = result["metadata"]

        for col, cmap_name in enumerate(colormaps):
            ax = axes[row, col] if n_configs > 1 else axes[col]

            # Apply colormap
            if cmap_name == "grayscale":
                # Grayscale - duplicate to RGB
                spec_rgb = np.stack([spec, spec, spec], axis=-1)
                # cmap_display = "gray"
            else:
                # Apply colormap
                spec_rgb = apply_colormap(spec, cmap_name)
                # cmap_display = cmap_name

            # Display
            ax.imshow(spec_rgb, aspect="auto", origin="upper", interpolation="bilinear")

            # Title
            if row == 0:
                ax.set_title(f"{cmap_name.capitalize()}", fontsize=12, fontweight="bold")

            # Y-axis label
            if col == 0:
                gain_str = (
                    f"+{metadata.get('gain_applied_db', 0):.1f}dB"
                    if metadata.get("gain_applied_db", 0) > 0
                    else f"{metadata.get('gain_applied_db', 0):.1f}dB"
                )
                ax.set_ylabel(f"{config_name}\n({gain_str})", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_path = output_dir / f"preprocessing_comparison_{audio_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved comparison: {output_path}")
    plt.close()

    # Create individual high-res versions
    for config_name, result in results.items():
        for cmap_name in colormaps:
            spec = result["spec"]

            if cmap_name == "grayscale":
                spec_rgb = np.stack([spec, spec, spec], axis=-1)
            else:
                spec_rgb = apply_colormap(spec, cmap_name)

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(spec_rgb, aspect="auto", origin="upper", interpolation="bilinear")
            ax.set_title(f"{config_name} | {cmap_name}", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Frequency (Mel)", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            output_path = output_dir / f"{config_name}_{cmap_name}_{audio_path.stem}.png"
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()

    print(f"\n✓ Saved {len(configs) * len(colormaps)} individual images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test audio preprocessing parameters visually")
    parser.add_argument(
        "--config",
        type=str,
        default="config/audio_chunking.yaml",
        help="Chunking config YAML to pull n_mels/FFT params from",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or directory of audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/preprocessing_test",
        help="Output directory for test images",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to load from each file (seconds)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.output)

    n_mels = 128
    fft_ms = 25.0
    hop_ms = 10.0
    f_min = 0.0
    f_max = None

    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        fft_cfg = cfg.get("fft", {})
        spec_cfg = cfg.get("spectrogram", {})
        n_mels = int(fft_cfg.get("n_mels", n_mels))
        fft_ms = float(fft_cfg.get("fft_ms", fft_ms))
        hop_ms = float(fft_cfg.get("hop_ms", hop_ms))
        f_min = float(spec_cfg.get("fmin", f_min)) if spec_cfg.get("fmin") is not None else f_min
        f_max = spec_cfg.get("fmax", f_max)

    if audio_path.is_file():
        test_preprocessing_params(
            audio_path,
            output_dir,
            args.duration,
            n_mels=n_mels,
            fft_ms=fft_ms,
            hop_ms=hop_ms,
            f_min=f_min,
            f_max=f_max,
        )
    elif audio_path.is_dir():
        # Process all audio files in directory
        audio_files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.flac"))
        print(f"Found {len(audio_files)} audio files in {audio_path}")

        for audio_file in audio_files:
            test_preprocessing_params(
                audio_file,
                output_dir / audio_file.stem,
                args.duration,
                n_mels=n_mels,
                fft_ms=fft_ms,
                hop_ms=hop_ms,
                f_min=f_min,
                f_max=f_max,
            )
    else:
        print(f"Error: {audio_path} is not a file or directory")
        return

    print("\n✓ All tests complete!")


if __name__ == "__main__":
    main()
