"""Audio preprocessing parameter experimentation toolkit.

This script allows interactive exploration of preprocessing parameters
(AGC, dynamic range, preemphasis, detrend) with visual feedback.

Usage:
    python experiments/preprocessing_explorer.py --audio data/sample.flac --output output/preproc_explore/

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec

from rf_detr_finetuning.dataprocessor import (
    AGCConfig,
    PreprocessingConfig,
    compute_mel_spectrogram,
    flip_spectrogram,
    load_audio_file,
    preprocess_audio,
    spectrogram_to_image_array,
)
from rf_detr_finetuning.dataprocessor.chunking import TimeBasedFFTConfig

logger = logging.getLogger(__name__)


def create_preprocessing_variations(
    audio: np.ndarray,
    sample_rate: int,
    base_config: PreprocessingConfig | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, PreprocessingConfig]]:
    """Create multiple preprocessing variations for comparison.

    Args:
        audio: Input audio array
        sample_rate: Sample rate
        base_config: Base preprocessing config (uses default if None)

    Returns:
        Dictionary mapping variation name to (audio, spectrogram, config)

    """
    if base_config is None:
        base_config = PreprocessingConfig()

    variations = {}

    # 1. No preprocessing
    config_none = PreprocessingConfig(
        agc=AGCConfig(enabled=False),
        detrend=False,
        preemphasis=0.0,
    )
    audio_none = audio.copy()
    spec_none = _compute_spectrogram(audio_none, sample_rate)
    variations["none"] = (audio_none, spec_none, config_none)

    # 2. AGC only
    for target_db in [-30.0, -25.0, -20.0, -15.0]:
        config_agc = PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=target_db),
            detrend=False,
            preemphasis=0.0,
        )
        audio_agc, _ = preprocess_audio(audio.copy(), sample_rate, config_agc)
        spec_agc = _compute_spectrogram(audio_agc, sample_rate)
        variations[f"agc_{target_db:.0f}db"] = (audio_agc, spec_agc, config_agc)

    # 3. Detrend only
    config_detrend = PreprocessingConfig(
        agc=AGCConfig(enabled=False),
        detrend=True,
        preemphasis=0.0,
    )
    audio_detrend, _ = preprocess_audio(audio.copy(), sample_rate, config_detrend)
    spec_detrend = _compute_spectrogram(audio_detrend, sample_rate)
    variations["detrend"] = (audio_detrend, spec_detrend, config_detrend)

    # 4. Preemphasis only
    config_preemp = PreprocessingConfig(
        agc=AGCConfig(enabled=False),
        detrend=False,
        preemphasis=0.97,
    )
    audio_preemp, _ = preprocess_audio(audio.copy(), sample_rate, config_preemp)
    spec_preemp = _compute_spectrogram(audio_preemp, sample_rate)
    variations["preemphasis"] = (audio_preemp, spec_preemp, config_preemp)

    # 5. Full preprocessing with different AGC targets
    for target_db in [-30.0, -25.0, -20.0]:
        config_full = PreprocessingConfig(
            agc=AGCConfig(enabled=True, target_db=target_db),
            detrend=True,
            preemphasis=0.97,
        )
        audio_full, _ = preprocess_audio(audio.copy(), sample_rate, config_full)
        spec_full = _compute_spectrogram(audio_full, sample_rate)
        variations[f"full_agc{target_db:.0f}db"] = (audio_full, spec_full, config_full)

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


def plot_preprocessing_comparison(
    variations: dict[str, tuple[np.ndarray, np.ndarray, PreprocessingConfig]],
    output_path: Path,
    original_audio: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> None:
    """Create comprehensive comparison plot.

    Args:
        variations: Dictionary of preprocessing variations
        output_path: Path to save plot
        original_audio: Original audio for RMS comparison
        sample_rate: Sample rate for time axis

    """
    n_variations = len(variations)
    fig = plt.figure(figsize=(16, 4 * n_variations))
    gs = GridSpec(n_variations, 3, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (name, (audio, spec, config)) in enumerate(variations.items()):
        # Column 1: Config info
        ax_config = fig.add_subplot(gs[idx, 0])
        ax_config.axis("off")

        config_text = f"Variation: {name}\n\n"
        config_text += f"AGC: {config.agc.enabled}\n"
        if config.agc.enabled:
            config_text += f"  Target: {config.agc.target_db:.1f} dB\n"
        config_text += f"Detrend: {config.detrend}\n"
        config_text += f"Preemphasis: {config.preemphasis}\n"

        # Compute RMS
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        config_text += f"\nOutput RMS: {rms_db:.1f} dB"

        ax_config.text(
            0.1,
            0.5,
            config_text,
            fontsize=10,
            verticalalignment="center",
            fontfamily="monospace",
            transform=ax_config.transAxes,
        )
        ax_config.set_title(f"Configuration: {name}", fontsize=12, fontweight="bold")

        # Column 2: Waveform
        ax_wave = fig.add_subplot(gs[idx, 1])
        if sample_rate:
            time = np.arange(len(audio)) / sample_rate
            ax_wave.plot(time, audio, linewidth=0.5)
            ax_wave.set_xlabel("Time (s)")
        else:
            ax_wave.plot(audio, linewidth=0.5)
            ax_wave.set_xlabel("Sample")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title("Waveform")
        ax_wave.grid(True, alpha=0.3)
        ax_wave.set_ylim(-1.1, 1.1)

        # Column 3: Spectrogram
        ax_spec = fig.add_subplot(gs[idx, 2])
        spec_img = spectrogram_to_image_array(spec, normalize=True)
        ax_spec.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")
        ax_spec.set_xlabel("Time (frames)")
        ax_spec.set_ylabel("Frequency (mel bins)")
        ax_spec.set_title("Mel Spectrogram")

    plt.suptitle("Preprocessing Parameter Comparison", fontsize=16, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {output_path}")


def save_experiment_config(
    variations: dict[str, tuple[np.ndarray, np.ndarray, PreprocessingConfig]],
    output_path: Path,
    audio_file: Path,
) -> None:
    """Save experiment configurations to YAML.

    Args:
        variations: Dictionary of preprocessing variations
        output_path: Path to save YAML
        audio_file: Source audio file path

    """
    experiment_data = {
        "source_audio": str(audio_file),
        "variations": {},
    }

    for name, (_, _, config) in variations.items():
        experiment_data["variations"][name] = {
            "agc": {
                "enabled": config.agc.enabled,
                "target_db": config.agc.target_db,
                "max_gain_db": config.agc.max_gain_db,
                "min_gain_db": config.agc.min_gain_db,
            },
            "detrend": config.detrend,
            "preemphasis": config.preemphasis,
        }

    with open(output_path, "w") as f:
        yaml.dump(experiment_data, f, default_flow_style=False)

    logger.info(f"Saved experiment config to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore preprocessing parameters with visual feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore all preprocessing variations
  python experiments/preprocessing_explorer.py --audio data/sample.flac

  # Save to specific directory
  python experiments/preprocessing_explorer.py --audio data/sample.flac --output experiments/results/

  # Use custom config file
  python experiments/preprocessing_explorer.py --audio data/sample.flac --config config/preprocessing.yaml
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
        default="output/preprocessing_experiments",
        help="Output directory for results (default: output/preprocessing_experiments)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional preprocessing config file to use as base",
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

    # Load base config if provided
    base_config = None
    if args.config:
        logger.info(f"Loading base config from {args.config}")
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            if "preprocessing" in config_data:
                base_config = PreprocessingConfig.from_dict(config_data["preprocessing"])

    # Create variations
    logger.info("Creating preprocessing variations...")
    variations = create_preprocessing_variations(audio, sample_rate, base_config)
    logger.info(f"Created {len(variations)} variations")

    # Generate comparison plot
    plot_path = output_dir / f"{audio_path.stem}_comparison.png"
    plot_preprocessing_comparison(variations, plot_path, audio, sample_rate)

    # Save configurations
    config_path = output_dir / f"{audio_path.stem}_configs.yaml"
    save_experiment_config(variations, config_path, audio_path)

    # Save individual spectrograms
    specs_dir = output_dir / "spectrograms"
    specs_dir.mkdir(exist_ok=True)

    for name, (_, spec, _) in variations.items():
        spec_img = spectrogram_to_image_array(spec, normalize=True)
        plt.figure(figsize=(12, 8))
        plt.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")
        plt.title(f"Preprocessing: {name}")
        plt.xlabel("Time (frames)")
        plt.ylabel("Frequency (mel bins)")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        plt.savefig(specs_dir / f"{audio_path.stem}_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"\n{'=' * 70}")
    logger.info("Preprocessing Exploration Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Comparison plot: {plot_path}")
    logger.info(f"Configs saved: {config_path}")
    logger.info(f"Individual spectrograms: {specs_dir}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
