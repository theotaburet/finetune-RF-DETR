"""Batch experiment runner for parameter exploration.

This script runs experiments across multiple audio files with
various parameter configurations and aggregates results.

Usage:
    python experiments/batch_experiments.py --config experiments/configs/preprocessing_grid.yaml

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rf_detr_finetuning.dataprocessor import (
    AGCConfig,
    PreprocessingConfig,
    load_audio_file,
    preprocess_audio,
)
from rf_detr_finetuning.dataprocessor.augmentation import (
    add_noise,
    random_gain,
    random_time_shift,
)

logger = logging.getLogger(__name__)
console = Console()


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    """Load experiment configuration from YAML.

    Args:
        config_path: Path to config file

    Returns:
        Experiment configuration dictionary

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_audio_files(input_dir: Path, extensions: tuple[str, ...] = (".flac", ".wav", ".mp3")) -> list[Path]:
    """Get list of audio files in directory.

    Args:
        input_dir: Input directory
        extensions: File extensions to include

    Returns:
        List of audio file paths

    """
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
    return sorted(files)


def run_preprocessing_experiment(
    audio_files: list[Path],
    configs: list[PreprocessingConfig],
    config_names: list[str],
) -> dict[str, Any]:
    """Run preprocessing experiment across multiple files.

    Args:
        audio_files: List of audio files to process
        configs: List of preprocessing configurations
        config_names: Names for each config

    Returns:
        Dictionary with experiment results

    """
    results = {
        "experiment_type": "preprocessing",
        "files_processed": len(audio_files),
        "configurations": {},
    }

    for config, config_name in zip(configs, config_names):
        config_results = {
            "rms_db": [],
            "max_amplitude": [],
            "files": [],
        }

        for audio_path in audio_files:
            try:
                audio, sample_rate = load_audio_file(audio_path)
                processed = preprocess_audio(audio, sample_rate, config)

                rms_db = 20 * np.log10(np.sqrt(np.mean(processed**2)) + 1e-10)
                max_amp = np.max(np.abs(processed))

                config_results["rms_db"].append(float(rms_db))
                config_results["max_amplitude"].append(float(max_amp))
                config_results["files"].append(str(audio_path.name))

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")

        # Compute statistics
        if config_results["rms_db"]:
            results["configurations"][config_name] = {
                "mean_rms_db": float(np.mean(config_results["rms_db"])),
                "std_rms_db": float(np.std(config_results["rms_db"])),
                "mean_max_amp": float(np.mean(config_results["max_amplitude"])),
                "clipping_pct": float(np.mean(np.array(config_results["max_amplitude"]) > 0.99) * 100),
                "n_files": len(config_results["rms_db"]),
            }

    return results


def run_augmentation_experiment(
    audio_files: list[Path],
    augmentation_params: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Run augmentation experiment across multiple files.

    Args:
        audio_files: List of audio files to process
        augmentation_params: Dictionary mapping augmentation name to list of parameter dicts

    Returns:
        Dictionary with experiment results

    """
    results = {
        "experiment_type": "augmentation",
        "files_processed": 0,
        "augmentations": {},
    }

    for aug_name, param_list in augmentation_params.items():
        aug_results = []

        for params in param_list:
            variation_results = {
                "params": params,
                "files_processed": 0,
                "success_rate": 0.0,
            }

            success_count = 0
            for audio_path in audio_files[:10]:  # Sample 10 files
                try:
                    audio, _ = load_audio_file(audio_path)

                    # Apply augmentation based on name
                    if aug_name == "noise":
                        add_noise(audio, **params)
                    elif aug_name == "time_shift":
                        random_time_shift(audio, **params)
                    elif aug_name == "gain":
                        random_gain(audio, **params)

                    success_count += 1
                except Exception as e:
                    logger.debug(f"Augmentation failed for {audio_path}: {e}")

            variation_results["files_processed"] = min(len(audio_files), 10)
            variation_results["success_rate"] = success_count / min(len(audio_files), 10)
            aug_results.append(variation_results)

        results["augmentations"][aug_name] = aug_results

    results["files_processed"] = len(audio_files)
    return results


def display_results(results: dict[str, Any]) -> None:
    """Display experiment results in a nice table.

    Args:
        results: Experiment results dictionary

    """
    console.print(f"\n[bold blue]{'=' * 70}[/bold blue]")
    console.print(f"[bold]Experiment Results: {results['experiment_type']}[/bold]")
    console.print(f"[bold blue]{'=' * 70}[/bold blue]\n")

    if results["experiment_type"] == "preprocessing":
        table = Table(title="Preprocessing Configuration Comparison")
        table.add_column("Config", style="cyan")
        table.add_column("Mean RMS (dB)", justify="right")
        table.add_column("Std RMS (dB)", justify="right")
        table.add_column("Clipping %", justify="right")
        table.add_column("Files", justify="right")

        for config_name, stats in results["configurations"].items():
            table.add_row(
                config_name,
                f"{stats['mean_rms_db']:.1f}",
                f"{stats['std_rms_db']:.1f}",
                f"{stats['clipping_pct']:.1f}%",
                str(stats["n_files"]),
            )

        console.print(table)

    elif results["experiment_type"] == "augmentation":
        for aug_name, aug_results in results["augmentations"].items():
            console.print(f"\n[bold]{aug_name.upper()}[/bold]")
            table = Table()
            table.add_column("Parameters")
            table.add_column("Success Rate", justify="right")

            for result in aug_results:
                params_str = ", ".join(f"{k}={v}" for k, v in result["params"].items())
                table.add_row(
                    params_str,
                    f"{result['success_rate'] * 100:.0f}%",
                )

            console.print(table)


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save experiment results to JSON.

    Args:
        results: Experiment results
        output_path: Path to save JSON

    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def create_default_preprocessing_configs() -> tuple[list[PreprocessingConfig], list[str]]:
    """Create default preprocessing configurations for comparison.

    Returns:
        Tuple of (configs, config_names)

    """
    configs = [
        PreprocessingConfig(agc=AGCConfig(enabled=False)),  # No preprocessing
        PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=-30.0)),
        PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=-25.0)),
        PreprocessingConfig(agc=AGCConfig(enabled=True, target_db=-20.0)),
        PreprocessingConfig(  # Full preprocessing
            agc=AGCConfig(enabled=True, target_db=-25.0),
            apply_detrend=True,
            apply_preemphasis=True,
        ),
    ]

    names = [
        "none",
        "agc_-30db",
        "agc_-25db",
        "agc_-20db",
        "full",
    ]

    return configs, names


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments across multiple audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python experiments/batch_experiments.py --config experiments/configs/test.yaml

  # Run preprocessing experiments on directory
  python experiments/batch_experiments.py --input data/audio/ --type preprocessing

  # Run augmentation experiments
  python experiments/batch_experiments.py --input data/audio/ --type augmentation
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["preprocessing", "augmentation"],
        default="preprocessing",
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/batch_experiments",
        help="Output directory for results",
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

    # Load or create experiment config
    if args.config:
        config = load_experiment_config(Path(args.config))
        input_dir = Path(config.get("input_dir", args.input))
    else:
        if not args.input:
            parser.error("--input is required when not using --config")
        input_dir = Path(args.input)
        config = {"experiment_type": args.type}

    # Get audio files
    audio_files = get_audio_files(input_dir)
    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    # Run experiment
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Running {args.type} experiments...", total=None)

        if args.type == "preprocessing":
            configs, names = create_default_preprocessing_configs()
            results = run_preprocessing_experiment(audio_files, configs, names)
        elif args.type == "augmentation":
            # Default augmentation params
            aug_params = {
                "noise": [
                    {"noise_level": 0.001},
                    {"noise_level": 0.005},
                    {"noise_level": 0.01},
                ],
                "gain": [
                    {"min_gain": 0.8, "max_gain": 1.2},
                    {"min_gain": 0.7, "max_gain": 1.3},
                ],
                "time_shift": [
                    {"max_shift_ratio": 0.1},
                    {"max_shift_ratio": 0.2},
                ],
            }
            results = run_augmentation_experiment(audio_files, aug_params)

    # Display and save results
    display_results(results)

    results_path = output_dir / f"{args.type}_results.json"
    save_results(results, results_path)

    console.print(f"\n[green]Results saved to: {output_dir}[/green]")


if __name__ == "__main__":
    main()
