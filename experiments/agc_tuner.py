"""AGC (Automatic Gain Control) parameter tuning toolkit.

This script provides interactive tuning of AGC parameters with
real-time visual feedback to find optimal settings for your dataset.

Usage:
    python experiments/agc_tuner.py --audio data/sample.flac --interactive

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

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


def compute_rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB."""
    rms = np.sqrt(np.mean(audio**2))
    return 20 * np.log10(rms + 1e-10)


def create_agc_parameter_grid(
    audio: np.ndarray,
    sample_rate: int,
    target_dbs: list[float] | None = None,
    max_gains: list[float] | None = None,
    frame_sizes: list[float] | None = None,
) -> dict[str, list[tuple[str, np.ndarray, AGCConfig, dict[str, float]]]]:
    """Create AGC parameter grid for exploration.

    Args:
        audio: Input audio array
        sample_rate: Sample rate
        target_dbs: List of target dB values to test
        max_gains: List of max gain values to test
        frame_sizes: List of frame sizes to test

    Returns:
        Dictionary mapping parameter name to list of variations

    """
    if target_dbs is None:
        target_dbs = [-35, -30, -25, -20, -15, -10]

    if max_gains is None:
        max_gains = [20, 30, 40, 50]

    if frame_sizes is None:
        frame_sizes = [0.05, 0.1, 0.2, 0.5]

    variations = {}

    # 1. Original (no AGC)
    variations["no_agc"] = [("original", audio.copy(), AGCConfig(enabled=False), {"rms_db": compute_rms_db(audio)})]

    # 2. Target dB variations
    variations["target_db"] = []
    for target_db in target_dbs:
        config = PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=target_db))
        processed, _ = preprocess_audio(audio.copy(), sample_rate, config)
        stats = {
            "rms_db": compute_rms_db(processed),
            "target_db": target_db,
            "gain_change_db": compute_rms_db(processed) - compute_rms_db(audio),
        }
        variations["target_db"].append((f"target_{target_db:.0f}db", processed, config.agc, stats))

    # 3. Max gain variations (with fixed target)
    variations["max_gain"] = []
    base_target = -25.0
    for max_gain in max_gains:
        config = PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=base_target, max_gain_db=max_gain))
        processed, _ = preprocess_audio(audio.copy(), sample_rate, config)
        stats = {
            "rms_db": compute_rms_db(processed),
            "max_gain": max_gain,
            "gain_change_db": compute_rms_db(processed) - compute_rms_db(audio),
        }
        variations["max_gain"].append((f"maxgain_{max_gain:.0f}db", processed, config.agc, stats))

    # 4. Frame size variations
    variations["frame_size"] = []
    for frame_size in frame_sizes:
        config = PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=base_target, frame_size_s=frame_size))
        processed, _ = preprocess_audio(audio.copy(), sample_rate, config)
        stats = {
            "rms_db": compute_rms_db(processed),
            "frame_size": frame_size,
            "gain_change_db": compute_rms_db(processed) - compute_rms_db(audio),
        }
        variations["frame_size"].append((f"framesize_{frame_size:.2f}s", processed, config.agc, stats))

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


def plot_agc_comparison(
    variations: dict[str, list[tuple[str, np.ndarray, AGCConfig, dict[str, float]]]],
    output_path: Path,
    original_audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Create comprehensive AGC comparison plot.

    Args:
        variations: Dictionary of AGC variations
        output_path: Path to save plot
        original_audio: Original audio for reference
        sample_rate: Sample rate

    """
    n_categories = len(variations)
    max_variations = max(len(v) for v in variations.values())

    fig = plt.figure(figsize=(5 * max_variations, 4 * n_categories))
    gs = GridSpec(n_categories, max_variations, figure=fig, hspace=0.4, wspace=0.3)

    original_rms = compute_rms_db(original_audio)

    for cat_idx, (category, var_list) in enumerate(variations.items()):
        for var_idx, (name, audio, config, stats) in enumerate(var_list):
            ax = fig.add_subplot(gs[cat_idx, var_idx])

            # Compute spectrogram
            spec = _compute_spectrogram(audio, sample_rate)
            spec_img = spectrogram_to_image_array(spec, normalize=True)

            # Plot with consistent viridis colormap
            im = ax.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")

            # Add info text
            info_text = f"RMS: {stats['rms_db']:.1f} dB\n"
            if "gain_change_db" in stats:
                info_text += f"ΔGain: {stats['gain_change_db']:+.1f} dB\n"
            if config.enabled:
                info_text += f"Target: {config.target_db:.0f} dB"

            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("Freq")

            # Add colorbar for first subplot in each row
            if var_idx == len(var_list) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for var_idx in range(len(var_list), max_variations):
            fig.add_subplot(gs[cat_idx, var_idx]).axis("off")

    plt.suptitle(f"AGC Parameter Comparison (Original RMS: {original_rms:.1f} dB)", fontsize=16, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved AGC comparison to {output_path}")


def plot_agc_waveforms(
    variations: dict[str, list[tuple[str, np.ndarray, AGCConfig, dict[str, float]]]],
    output_path: Path,
    sample_rate: int,
) -> None:
    """Plot waveforms for AGC variations.

    Args:
        variations: Dictionary of AGC variations
        output_path: Path to save plot
        sample_rate: Sample rate

    """
    # Select a subset for waveform visualization
    waveform_variations = []
    if "no_agc" in variations:
        waveform_variations.append(variations["no_agc"][0])
    if "target_db" in variations:
        waveform_variations.extend(variations["target_db"][::2])  # Every other

    n_plots = len(waveform_variations)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for idx, (name, audio, config, stats) in enumerate(waveform_variations):
        ax = axes[idx]
        time = np.arange(len(audio)) / sample_rate

        ax.plot(time, audio, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f"{name}\n({stats['rms_db']:.1f} dB)")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

        # Add RMS indicator
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("AGC Waveform Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved waveform comparison to {output_path}")


def recommend_agc_settings(
    variations: dict[str, list[tuple[str, np.ndarray, AGCConfig, dict[str, float]]]],
    original_audio: np.ndarray,
) -> dict[str, Any]:
    """Analyze variations and recommend AGC settings.

    Args:
        variations: Dictionary of AGC variations
        original_audio: Original audio

    Returns:
        Dictionary with recommendations

    """
    original_rms = compute_rms_db(original_audio)

    # Find best target dB
    best_target = None
    target_results = variations.get("target_db", [])
    if target_results:
        # Find target that gets closest to -25 dB without clipping
        best_target = None
        best_score = float("inf")

        for name, audio, config, stats in target_results:
            rms = stats["rms_db"]
            max_amp = np.max(np.abs(audio))

            # Score: penalize if far from -25 dB or if clipping
            score = abs(rms - (-25.0))
            if max_amp > 0.99:  # Clipping detected
                score += 10

            if score < best_score:
                best_score = score
                best_target = config.target_db

    # Find appropriate max gain
    recommended_max_gain = None
    max_gain_results = variations.get("max_gain", [])
    if max_gain_results:
        for name, audio, config, stats in max_gain_results:
            max_amp = np.max(np.abs(audio))
            if max_amp < 0.99:  # No clipping
                recommended_max_gain = config.max_gain_db
                break

        if recommended_max_gain is None:
            recommended_max_gain = 30.0  # Conservative default

    recommendations = {
        "original_rms_db": original_rms,
        "recommended_target_db": best_target if best_target else -25.0,
        "recommended_max_gain_db": recommended_max_gain if recommended_max_gain else 30.0,
        "recommended_frame_size_s": 0.1,
        "note": "Target -25 dB provides good balance. Adjust based on dataset.",
    }

    return recommendations


def save_agc_analysis(
    variations: dict[str, list[tuple[str, np.ndarray, AGCConfig, dict[str, float]]]],
    recommendations: dict[str, Any],
    output_path: Path,
) -> None:
    """Save AGC analysis and recommendations.

    Args:
        variations: Dictionary of AGC variations
        recommendations: Recommended settings
        output_path: Path to save YAML

    """
    analysis = {
        "recommendations": recommendations,
        "tested_parameters": {},
    }

    for category, var_list in variations.items():
        analysis["tested_parameters"][category] = []
        for name, audio, config, stats in var_list:
            entry = {
                "name": name,
                "rms_db": float(stats["rms_db"]),
                "max_amplitude": float(np.max(np.abs(audio))),
            }
            if config.enabled:
                entry["config"] = {
                    "target_db": config.target_db,
                    "max_gain_db": config.max_gain_db,
                    "min_gain_db": config.min_gain_db,
                    "frame_size_s": config.frame_size_s,
                }
            analysis["tested_parameters"][category].append(entry)

    with open(output_path, "w") as f:
        yaml.dump(analysis, f, default_flow_style=False)

    logger.info(f"Saved AGC analysis to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tune AGC parameters with visual feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameter grid
  python experiments/agc_tuner.py --audio data/sample.flac

  # Custom parameter ranges
  python experiments/agc_tuner.py --audio data/sample.flac \\
    --target-dbs -30,-25,-20 --max-gains 20,30,40

  # Save recommended config for training
  python experiments/agc_tuner.py --audio data/sample.flac --save-config config/agc.yaml
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/agc_tuning",
        help="Output directory for results",
    )
    parser.add_argument(
        "--target-dbs",
        type=str,
        default="-35,-30,-25,-20,-15,-10",
        help="Comma-separated list of target dB values to test",
    )
    parser.add_argument(
        "--max-gains",
        type=str,
        default="20,30,40,50",
        help="Comma-separated list of max gain values to test",
    )
    parser.add_argument(
        "--frame-sizes",
        type=str,
        default="0.05,0.1,0.2,0.5",
        help="Comma-separated list of frame sizes to test",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save recommended config to YAML file for training",
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

    # Parse parameter ranges
    target_dbs = [float(x) for x in args.target_dbs.split(",")]
    max_gains = [float(x) for x in args.max_gains.split(",")]
    frame_sizes = [float(x) for x in args.frame_sizes.split(",")]

    # Create parameter grid
    logger.info("Creating AGC parameter grid...")
    variations = create_agc_parameter_grid(audio, sample_rate, target_dbs, max_gains, frame_sizes)

    # Generate comparison plots
    logger.info("Generating comparison plots...")
    spec_path = output_dir / f"{audio_path.stem}_agc_spectrograms.png"
    plot_agc_comparison(variations, spec_path, audio, sample_rate)

    wave_path = output_dir / f"{audio_path.stem}_agc_waveforms.png"
    plot_agc_waveforms(variations, wave_path, sample_rate)

    # Generate recommendations
    logger.info("Analyzing results...")
    recommendations = recommend_agc_settings(variations, audio)

    # Print recommendations
    logger.info(f"\n{'=' * 70}")
    logger.info("AGC TUNING RECOMMENDATIONS")
    logger.info(f"{'=' * 70}")
    logger.info(f"Original RMS: {recommendations['original_rms_db']:.1f} dB")
    logger.info(f"Recommended Target: {recommendations['recommended_target_db']:.1f} dB")
    logger.info(f"Recommended Max Gain: {recommendations['recommended_max_gain_db']:.0f} dB")
    logger.info(f"Recommended Frame Size: {recommendations['recommended_frame_size_s']:.2f} s")
    logger.info(f"{'=' * 70}")

    # Save analysis
    analysis_path = output_dir / f"{audio_path.stem}_agc_analysis.yaml"
    save_agc_analysis(variations, recommendations, analysis_path)

    # Save config if requested
    if args.save_config:
        config_path = Path(args.save_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        agc_config = {
            "preprocessing": {
                "agc": {
                    "enabled": True,
                    "target_db": recommendations["recommended_target_db"],
                    "max_gain_db": recommendations["recommended_max_gain_db"],
                    "min_gain_db": -20.0,
                    "frame_size_s": recommendations["recommended_frame_size_s"],
                }
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(agc_config, f, default_flow_style=False)

        logger.info(f"Saved recommended config to {config_path}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
