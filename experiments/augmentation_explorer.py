"""Audio augmentation parameter experimentation toolkit.

This script allows interactive exploration of augmentation parameters
(noise, time shift, gain, spec augment) with visual feedback.

Usage:
    python experiments/augmentation_explorer.py --audio data/sample.flac --output output/aug_explore/

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rf_detr_finetuning.dataprocessor import (
    compute_mel_spectrogram,
    flip_spectrogram,
    load_audio_file,
    spectrogram_to_image_array,
)
from rf_detr_finetuning.dataprocessor.augmentation import (
    add_noise,
    random_gain,
    random_time_shift,
    spec_augment,
)
from rf_detr_finetuning.dataprocessor.chunking import TimeBasedFFTConfig

logger = logging.getLogger(__name__)


def create_augmentation_variations(
    audio: np.ndarray,
    sample_rate: int,
    n_variations: int = 3,
) -> dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]]:
    """Create multiple augmentation variations for comparison.

    Args:
        audio: Input audio array
        sample_rate: Sample rate
        n_variations: Number of random variations per augmentation type

    Returns:
        Dictionary mapping augmentation category to list of (name, audio, params)

    """
    variations = {}

    # 1. Original (no augmentation)
    variations["original"] = [("original", audio.copy(), {})]

    # 2. Noise variations
    variations["noise"] = []
    for noise_level in [0.001, 0.005, 0.01, 0.02]:
        aug_audio = add_noise(audio.copy(), noise_level=noise_level)
        snr_db = 20 * np.log10(np.std(audio) / (noise_level * np.std(audio) + 1e-10))
        variations["noise"].append(
            (f"noise_{noise_level:.3f}", aug_audio, {"noise_level": noise_level, "snr_db": snr_db})
        )

    # 3. Time shift variations
    variations["time_shift"] = []
    for max_shift in [0.05, 0.1, 0.2]:
        for i in range(n_variations):
            aug_audio = random_time_shift(audio.copy(), max_shift_ratio=max_shift)
            variations["time_shift"].append(
                (f"shift_{max_shift:.2f}_v{i + 1}", aug_audio, {"max_shift_ratio": max_shift})
            )

    # 4. Gain variations
    variations["gain"] = []
    for min_gain_db in [-6, -3, 0]:
        for max_gain_db in [0, 3, 6]:
            for i in range(2):
                aug_audio = random_gain(audio.copy(), min_gain_db=min_gain_db, max_gain_db=max_gain_db)
                gain_factor = np.std(aug_audio) / (np.std(audio) + 1e-10)
                variations["gain"].append(
                    (
                        f"gain_{min_gain_db}dB_{max_gain_db}dB_v{i + 1}",
                        aug_audio,
                        {"min_gain_db": min_gain_db, "max_gain_db": max_gain_db, "applied_gain": gain_factor},
                    )
                )

    return variations


def create_spec_augment_variations(
    spec: np.ndarray,
    n_variations: int = 3,
) -> dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]]:
    """Create spectrogram augmentation variations.

    Args:
        spec: Input spectrogram
        n_variations: Number of random variations

    Returns:
        Dictionary mapping augmentation category to list of (name, spec, params)

    """
    variations = {}

    # Original
    variations["original"] = [("original", spec.copy(), {})]

    # Time masking variations
    variations["time_mask"] = []
    for time_mask_param in [10, 20, 40]:
        for i in range(n_variations):
            aug_spec = spec_augment(
                spec.copy(),
                time_mask_param=time_mask_param,
                freq_mask_param=0,
                n_time_masks=1,
                n_freq_masks=0,
            )
            variations["time_mask"].append(
                (f"time_mask_{time_mask_param}_v{i + 1}", aug_spec, {"time_mask_param": time_mask_param})
            )

    # Frequency masking variations
    variations["freq_mask"] = []
    for freq_mask_param in [5, 10, 20]:
        for i in range(n_variations):
            aug_spec = spec_augment(
                spec.copy(),
                time_mask_param=0,
                freq_mask_param=freq_mask_param,
                n_time_masks=0,
                n_freq_masks=1,
            )
            variations["freq_mask"].append(
                (f"freq_mask_{freq_mask_param}_v{i + 1}", aug_spec, {"freq_mask_param": freq_mask_param})
            )

    # Combined variations
    variations["combined"] = []
    for time_mask_param in [10, 20]:
        for freq_mask_param in [5, 10]:
            for i in range(n_variations):
                aug_spec = spec_augment(
                    spec.copy(),
                    time_mask_param=time_mask_param,
                    freq_mask_param=freq_mask_param,
                    n_time_masks=1,
                    n_freq_masks=1,
                )
                variations["combined"].append(
                    (
                        f"both_t{time_mask_param}_f{freq_mask_param}_v{i + 1}",
                        aug_spec,
                        {"time_mask_param": time_mask_param, "freq_mask_param": freq_mask_param},
                    )
                )

    return variations


def _compute_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute mel spectrogram."""
    fft_config = TimeBasedFFTConfig()
    spec = compute_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=fft_config.n_mels,
        n_fft=fft_config.get_n_fft(sample_rate),
        hop_length=fft_config.get_hop_length(sample_rate),
    )
    spec = flip_spectrogram(spec)
    return spec


def plot_augmentation_comparison(
    variations: dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]],
    output_path: Path,
    augmentation_type: str,
    sample_rate: int | None = None,
) -> None:
    """Create comparison plot for audio augmentations.

    Args:
        variations: Dictionary of augmentation variations
        output_path: Path to save plot
        augmentation_type: Type of augmentation ('audio' or 'spec')
        sample_rate: Sample rate for time axis

    """
    # Get max number of variations in any category
    max_variations = max(len(v) for v in variations.values())
    n_categories = len(variations)

    fig, axes = plt.subplots(n_categories, max_variations, figsize=(4 * max_variations, 3 * n_categories))
    if n_categories == 1:
        axes = axes.reshape(1, -1)

    for cat_idx, (category, var_list) in enumerate(variations.items()):
        for var_idx, (name, data, params) in enumerate(var_list):
            ax = axes[cat_idx, var_idx]

            if augmentation_type == "audio":
                # Plot waveform
                if sample_rate:
                    time = np.arange(len(data)) / sample_rate
                    ax.plot(time, data, linewidth=0.5)
                    ax.set_xlabel("Time (s)")
                else:
                    ax.plot(data, linewidth=0.5)
                    ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.3)

                # Add stats
                rms = np.sqrt(np.mean(data**2))
                ax.text(0.02, 0.98, f"RMS: {rms:.3f}", transform=ax.transAxes, verticalalignment="top", fontsize=8)

            else:  # spectrogram
                # Plot spectrogram with consistent viridis colormap
                spec_img = spectrogram_to_image_array(data, normalize=True)
                ax.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")
                ax.set_xlabel("Time")
                ax.set_ylabel("Freq")

            ax.set_title(name, fontsize=10)

        # Hide unused subplots
        for var_idx in range(len(var_list), max_variations):
            axes[cat_idx, var_idx].axis("off")

        # Add row label
        axes[cat_idx, 0].set_ylabel(category, fontsize=12, fontweight="bold", rotation=0, ha="right", va="center")

    plt.suptitle(f"{augmentation_type.capitalize()} Augmentation Variations", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved {augmentation_type} augmentation comparison to {output_path}")


def save_augmentation_config(
    variations: dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]],
    output_path: Path,
    augmentation_type: str,
) -> None:
    """Save augmentation configurations to YAML.

    Args:
        variations: Dictionary of augmentation variations
        output_path: Path to save YAML
        augmentation_type: Type of augmentation

    """
    config_data = {
        "augmentation_type": augmentation_type,
        "variations": {},
    }

    for category, var_list in variations.items():
        config_data["variations"][category] = []
        for name, _, params in var_list:
            config_data["variations"][category].append(
                {
                    "name": name,
                    "parameters": params,
                }
            )

    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    logger.info(f"Saved augmentation config to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore augmentation parameters with visual feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore all augmentation variations
  python experiments/augmentation_explorer.py --audio data/sample.flac

  # Explore only specific augmentation types
  python experiments/augmentation_explorer.py --audio data/sample.flac --types noise,gain

  # Generate more variations
  python experiments/augmentation_explorer.py --audio data/sample.flac --variations 5
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/augmentation_experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="all",
        help="Comma-separated list of augmentation types (default: all)",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="Number of random variations per parameter (default: 3)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio_path = Path(args.audio)
    logger.info(f"Loading audio from {audio_path}")
    audio, sample_rate = load_audio_file(audio_path)
    logger.info(f"Loaded audio: {len(audio)} samples @ {sample_rate} Hz")

    # Compute base spectrogram
    logger.info("Computing base spectrogram...")
    base_spec = _compute_spectrogram(audio, sample_rate)

    # Parse augmentation types
    aug_types = [t.strip() for t in args.types.split(",")]

    # Create audio augmentations
    if "all" in aug_types or "audio" in aug_types:
        logger.info("Creating audio augmentation variations...")
        audio_variations = create_augmentation_variations(audio, sample_rate, args.variations)

        # Filter by requested types
        if "all" not in aug_types:
            audio_variations = {k: v for k, v in audio_variations.items() if k in aug_types}

        if audio_variations:
            plot_path = output_dir / f"{audio_path.stem}_audio_aug.png"
            plot_augmentation_comparison(audio_variations, plot_path, "audio", sample_rate)

            config_path = output_dir / f"{audio_path.stem}_audio_aug.yaml"
            save_augmentation_config(audio_variations, config_path, "audio")

    # Create spectrogram augmentations
    if "all" in aug_types or "spec" in aug_types:
        logger.info("Creating spectrogram augmentation variations...")
        spec_variations = create_spec_augment_variations(base_spec, args.variations)

        plot_path = output_dir / f"{audio_path.stem}_spec_aug.png"
        plot_augmentation_comparison(spec_variations, plot_path, "spec")

        config_path = output_dir / f"{audio_path.stem}_spec_aug.yaml"
        save_augmentation_config(spec_variations, config_path, "spectrogram")

    # Save example spectrograms
    specs_dir = output_dir / "spectrograms"
    specs_dir.mkdir(exist_ok=True)

    for aug_type, var_list in create_augmentation_variations(audio, sample_rate, 1).items():
        for name, aug_audio, _ in var_list[:1]:  # Save only first variation
            spec = _compute_spectrogram(aug_audio, sample_rate)
            spec_img = spectrogram_to_image_array(spec, normalize=True)

            plt.figure(figsize=(10, 6))
            plt.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")
            plt.title(f"Augmentation: {aug_type} - {name}")
            plt.xlabel("Time (frames)")
            plt.ylabel("Frequency (mel bins)")
            plt.colorbar(label="Intensity")
            plt.tight_layout()
            plt.savefig(specs_dir / f"{audio_path.stem}_{aug_type}.png", dpi=300, bbox_inches="tight")
            plt.close()

    logger.info(f"\n{'=' * 70}")
    logger.info("Augmentation Exploration Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
