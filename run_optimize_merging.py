#!/usr/bin/env python3
"""Optimize merge parameters for audio event detection.

Runs inference on test audio files, then grid-searches over merge parameters
(delta_time_ms, delta_freq_hz, score_threshold, min_duration_ms) to maximize
F1 score against ground truth annotations.

The optimized parameters are saved to a YAML config file for reuse during inference.

Usage:
    # Optimize using download metadata (test split)
    python run_optimize_merging.py \
        --metadata data/downloaded/metadata/download_metadata.json \
        --weights output/checkpoint_best.pth \
        --chunking-config config/chunking.yaml \
        --output config/merging.yaml

    # Optimize with custom search space
    python run_optimize_merging.py \
        --metadata data/downloaded/metadata/download_metadata.json \
        --weights output/checkpoint_best.pth \
        --chunking-config config/chunking.yaml \
        --output config/merging.yaml \
        --delta-time-values 200 500 1000 2000 4000 8000 \
        --delta-freq-values 50 100 200 500 \
        --score-threshold-values 0.1 0.2 0.3 0.5

    # Per-class optimization (slower, more accurate)
    python run_optimize_merging.py \
        --metadata data/downloaded/metadata/download_metadata.json \
        --weights output/checkpoint_best.pth \
        --chunking-config config/chunking.yaml \
        --output config/merging.yaml \
        --per-class

    # Evaluate current config without optimization
    python run_optimize_merging.py \
        --metadata data/downloaded/metadata/download_metadata.json \
        --weights output/checkpoint_best.pth \
        --chunking-config config/chunking.yaml \
        --merge-config config/merging.yaml \
        --evaluate-only

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize merge parameters for audio event detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to download_metadata.json with test split annotations",
    )
    input_group.add_argument(
        "--audio-dir",
        type=Path,
        help="Override audio directory (default: inferred from metadata output_path)",
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model weights/checkpoint",
    )
    model_group.add_argument(
        "--model-size",
        choices=["nano", "small", "base", "medium", "large"],
        default="base",
        help="RF-DETR model size",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device",
    )

    # Config
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--chunking-config",
        type=Path,
        required=True,
        help="Path to chunking config YAML",
    )
    config_group.add_argument(
        "--merge-config",
        type=Path,
        help="Existing merge config (for --evaluate-only mode)",
    )
    config_group.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (applied before merging)",
    )

    # Search space
    search_group = parser.add_argument_group("Search Space")
    search_group.add_argument(
        "--delta-time-values",
        type=float,
        nargs="+",
        default=[200, 500, 1000, 2000, 4000, 8000],
        help="Delta time values to search (ms)",
    )
    search_group.add_argument(
        "--delta-freq-values",
        type=float,
        nargs="+",
        default=[50, 100, 200, 500],
        help="Delta frequency values to search (Hz)",
    )
    search_group.add_argument(
        "--score-threshold-values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5],
        help="Score threshold values to search",
    )
    search_group.add_argument(
        "--min-duration-values",
        type=float,
        nargs="+",
        default=[0, 100, 200],
        help="Minimum duration values to search (ms)",
    )
    search_group.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for evaluation matching",
    )

    # Mode
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--per-class",
        action="store_true",
        help="Optimize per-class parameters (slower but more accurate)",
    )
    mode_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing merge config without optimization",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        type=Path,
        default=Path("config/merging.yaml"),
        help="Output YAML file for optimized parameters",
    )
    output_group.add_argument(
        "--results-json",
        type=Path,
        help="Save detailed results to JSON",
    )

    # Debug
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    debug_group.add_argument(
        "--max-files",
        type=int,
        help="Maximum test files to process",
    )
    debug_group.add_argument(
        "--max-duration",
        type=float,
        help="Skip audio files longer than this duration in seconds (for testing)",
    )

    return parser.parse_args()


def load_test_data(
    metadata_path: Path,
    audio_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load test split audio files and their ground truth events from download metadata.

    Args:
        metadata_path: Path to download_metadata.json.
        audio_dir: Override directory for audio files.

    Returns:
        List of dicts with keys: audio_path, events, source_file, sound_id.

    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    test_sounds = []
    for sound in metadata.get("sounds", []):
        if sound.get("split") != "test":
            continue
        if not sound.get("success", False):
            continue

        output_path = Path(sound["output_path"])
        if audio_dir:
            output_path = audio_dir / output_path.name

        if not output_path.exists():
            logger.warning(f"Audio file not found: {output_path}")
            continue

        # Convert events from seconds to milliseconds
        events = []
        for event in sound.get("events", []):
            events.append(
                {
                    "start_ms": event.get("start_s", 0) * 1000,
                    "end_ms": event.get("end_s", 0) * 1000,
                    "hz_min": event.get("hz_min"),
                    "hz_max": event.get("hz_max"),
                    "label_hierarchy": event.get("label_hierarchy", ""),
                    "confidence": event.get("confidence", 1.0),
                }
            )

        test_sounds.append(
            {
                "audio_path": output_path,
                "events": events,
                "source_file": sound.get("source_file", ""),
                "sound_id": sound.get("sound_id", ""),
            }
        )

    return test_sounds


def build_class_mapping(test_sounds: list[dict], class_names: list[str] | None = None) -> dict[str, int]:
    """Build label-to-class-ID mapping from test data or model class names.

    Args:
        test_sounds: Test sound data with events.
        class_names: Model class names (ordered by class ID).

    Returns:
        Dict mapping label string to class ID.

    """
    if class_names:
        return {name: i for i, name in enumerate(class_names)}

    # Auto-discover from ground truth labels
    labels = set()
    for sound in test_sounds:
        for event in sound["events"]:
            hierarchy = event.get("label_hierarchy", "")
            if hierarchy:
                label = hierarchy.split(" + ")[-1]
                labels.add(label)

    return {label: i for i, label in enumerate(sorted(labels))}


def events_to_audio_events(
    events: list[dict],
    label_to_class: dict[str, int],
) -> list:
    """Convert raw event dicts to AudioEvent objects.

    Args:
        events: Raw event dictionaries.
        label_to_class: Label-to-class-ID mapping.

    Returns:
        List of AudioEvent objects.

    """
    from rf_detr_finetuning.eventprocessor.event import AudioEvent

    result = []
    for event in events:
        hierarchy = event.get("label_hierarchy", "")
        label = hierarchy.split(" + ")[-1] if hierarchy else "unknown"
        class_id = label_to_class.get(label, 0)

        result.append(
            AudioEvent(
                start_ms=event["start_ms"],
                end_ms=event["end_ms"],
                class_id=class_id,
                class_name=label,
                score=event.get("confidence", 1.0),
                min_freq_hz=event.get("hz_min"),
                max_freq_hz=event.get("hz_max"),
            )
        )

    return result


def run_inference_on_test_files(
    test_sounds: list[dict],
    weights_path: Path,
    chunking_config_path: Path,
    model_size: str,
    device: str,
    class_names: list[str] | None,
    confidence: float,
) -> list:
    """Run inference on test audio files, returning raw (unmerged) events.

    Uses the AudioInferencePipeline with IoU-based merging disabled (very
    large merge gap = no merging effectively). The postprocessor converts
    pixel coordinates to absolute time but does NOT merge overlapping windows.

    Args:
        test_sounds: Test sound metadata.
        weights_path: Model weights path.
        chunking_config_path: Chunking config path.
        model_size: RF-DETR model size.
        device: Inference device.
        class_names: Model class names.
        confidence: Detection confidence threshold.

    Returns:
        List of EventList (one per file, unmerged).

    """
    from rf_detr_finetuning.eventprocessor import EventPostProcessor, MergeConfig, PostProcessorConfig
    from run_inference_audio import AudioInferencePipeline

    # Initialize pipeline with no merging (just coordinate conversion)
    pipeline = AudioInferencePipeline(
        weights_path=weights_path,
        config_path=chunking_config_path,
        model_size=model_size,
        device=device,
        class_names=class_names,
        confidence_threshold=confidence,
        iou_threshold=1.0,
        merge_gap_ms=0.0,
        min_event_duration_ms=0.0,
    )

    # Override postprocessor to use very permissive settings (no merging)
    no_merge_config = PostProcessorConfig(
        time_per_pixel_ms=pipeline.fft_config.hop_ms,
        confidence_threshold=confidence,
        min_event_duration_ms=0.0,
        merge_config=MergeConfig(
            iou_threshold=1.0,
            score_threshold=0.0,
            merge_same_class_only=True,
            merge_strategy="max",
            gap_tolerance_ms=0.0,
        ),
        class_names={i: name for i, name in enumerate(class_names or [])},
    )
    pipeline.postprocessor = EventPostProcessor(no_merge_config)

    all_raw_events = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running inference...", total=len(test_sounds))

        for sound in test_sounds:
            audio_path = sound["audio_path"]
            progress.update(task, description=f"Inference: {Path(audio_path).name[:40]}")

            try:
                result = pipeline.process_audio(Path(audio_path))

                # Convert event dicts back to AudioEvent objects in an EventList
                from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList

                events = []
                for e in result.events:
                    events.append(
                        AudioEvent(
                            start_ms=e["start_ms"],
                            end_ms=e["end_ms"],
                            class_id=e.get("class_id", 0),
                            class_name=e.get("class_name"),
                            score=e.get("score", 1.0),
                            min_freq_hz=e.get("min_freq_hz"),
                            max_freq_hz=e.get("max_freq_hz"),
                            source_windows=e.get("source_windows", []),
                        )
                    )

                event_list = EventList(
                    events=events,
                    audio_path=str(audio_path),
                    duration_ms=result.duration_ms,
                )
                all_raw_events.append(event_list)

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                all_raw_events.append(EventList())

            progress.advance(task)

    return all_raw_events


def display_eval_table(eval_result, title: str = "Evaluation Results") -> None:
    """Display evaluation results in a formatted table.

    Args:
        eval_result: EvaluationResult to display.
        title: Table title.

    """
    table = Table(title=title)
    table.add_column("Class", style="cyan", width=25)
    table.add_column("TP", style="green", width=6, justify="right")
    table.add_column("FP", style="red", width=6, justify="right")
    table.add_column("FN", style="yellow", width=6, justify="right")
    table.add_column("Precision", style="blue", width=10, justify="right")
    table.add_column("Recall", style="blue", width=10, justify="right")
    table.add_column("F1", style="magenta", width=10, justify="right")
    table.add_column("Avg IoU", style="dim", width=10, justify="right")

    for cid in sorted(eval_result.per_class.keys()):
        m = eval_result.per_class[cid]
        table.add_row(
            f"{m.class_name} ({cid})",
            str(m.true_positives),
            str(m.false_positives),
            str(m.false_negatives),
            f"{m.precision:.3f}",
            f"{m.recall:.3f}",
            f"{m.f1:.3f}",
            f"{m.avg_iou:.3f}",
        )

    # Overall row
    o = eval_result.overall
    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{o.true_positives}[/bold]",
        f"[bold]{o.false_positives}[/bold]",
        f"[bold]{o.false_negatives}[/bold]",
        f"[bold]{o.precision:.3f}[/bold]",
        f"[bold]{o.recall:.3f}[/bold]",
        f"[bold]{o.f1:.3f}[/bold]",
        f"[bold]{o.avg_iou:.3f}[/bold]",
    )

    console.print(table)


def display_config_table(config, class_names: dict[int, str] | None = None) -> None:
    """Display merge config as a table.

    Args:
        config: ClassWiseMergeConfig.
        class_names: Class name mapping.

    """
    class_names = class_names or {}

    table = Table(title="Optimized Merge Parameters")
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green", width=20)

    table.add_row("Default delta_time_ms", f"{config.default_params.delta_time_ms}")
    table.add_row("Default delta_freq_hz", f"{config.default_params.delta_freq_hz}")
    table.add_row("Default score_strategy", config.default_params.score_strategy)
    table.add_row("Score threshold", f"{config.score_threshold}")
    table.add_row("Min duration (ms)", f"{config.min_duration_ms}")

    console.print(table)

    if config.class_params:
        class_table = Table(title="Per-Class Parameters")
        class_table.add_column("Class", style="cyan", width=25)
        class_table.add_column("delta_time_ms", style="green", width=15, justify="right")
        class_table.add_column("delta_freq_hz", style="green", width=15, justify="right")
        class_table.add_column("score_strategy", style="yellow", width=15)

        for cid, params in sorted(config.class_params.items()):
            name = class_names.get(cid, f"class_{cid}")
            class_table.add_row(
                f"{name} ({cid})",
                f"{params.delta_time_ms}",
                f"{params.delta_freq_hz}",
                params.score_strategy,
            )

        console.print(class_table)


def main() -> int:
    """Main entry point for merge parameter optimization."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Merge Parameter Optimizer[/bold blue]\n")

    # Validate inputs
    if not args.metadata.exists():
        console.print(f"[red]Error:[/red] Metadata not found: {args.metadata}")
        return 1
    if not args.weights.exists():
        console.print(f"[red]Error:[/red] Weights not found: {args.weights}")
        return 1
    if not args.chunking_config.exists():
        console.print(f"[red]Error:[/red] Chunking config not found: {args.chunking_config}")
        return 1

    # Load test data
    console.print(f"[cyan]Loading test data from:[/cyan] {args.metadata}")
    test_sounds = load_test_data(args.metadata, args.audio_dir)

    if not test_sounds:
        console.print("[red]Error:[/red] No test audio files found in metadata")
        return 1

    if args.max_files:
        test_sounds = test_sounds[: args.max_files]

    # Filter by duration (reads file headers only, no full load)
    if args.max_duration:
        from rf_detr_finetuning.dataprocessor.io import filter_audio_by_duration

        all_paths = [s["audio_path"] for s in test_sounds]
        kept_paths = set(filter_audio_by_duration(all_paths, args.max_duration))
        before_count = len(test_sounds)
        test_sounds = [s for s in test_sounds if s["audio_path"] in kept_paths]
        if len(test_sounds) < before_count:
            console.print(
                f"[yellow]Duration filter ({args.max_duration}s):[/yellow] "
                f"kept {len(test_sounds)} / {before_count} files"
            )

    console.print(f"[green]Found {len(test_sounds)} test audio files[/green]")

    # Build class mapping from model checkpoint
    from rf_detr_finetuning.utils import load_class_names

    class_names = load_class_names(None, None)

    # Try to load class names from checkpoint
    try:
        import torch

        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        if hasattr(checkpoint.get("args", object()), "class_names"):
            class_names = checkpoint["args"].class_names
            console.print(f"[cyan]Classes from checkpoint:[/cyan] {class_names}")
    except Exception:
        pass

    label_to_class = build_class_mapping(test_sounds, class_names)
    class_names_dict = {v: k for k, v in label_to_class.items()}

    console.print(f"[cyan]Class mapping:[/cyan] {label_to_class}")

    # Build ground truth AudioEvents for each file
    gt_per_file = [events_to_audio_events(sound["events"], label_to_class) for sound in test_sounds]

    total_gt = sum(len(gt) for gt in gt_per_file)
    console.print(f"[cyan]Total ground truth events:[/cyan] {total_gt}")

    # Run inference to get raw detections
    console.print("\n[bold]Running inference on test files...[/bold]")
    raw_events_per_file = run_inference_on_test_files(
        test_sounds,
        weights_path=args.weights,
        chunking_config_path=args.chunking_config,
        model_size=args.model_size,
        device=args.device,
        class_names=class_names,
        confidence=args.confidence,
    )

    total_raw = sum(len(e) for e in raw_events_per_file)
    console.print(f"[green]Total raw detections:[/green] {total_raw}")

    # Evaluate-only mode
    if args.evaluate_only:
        if not args.merge_config or not args.merge_config.exists():
            console.print("[red]Error:[/red] --merge-config required for --evaluate-only mode")
            return 1

        import yaml

        from rf_detr_finetuning.eventprocessor.evaluator import evaluate_multi_file
        from rf_detr_finetuning.eventprocessor.merger import ClassWiseMergeConfig, ClassWiseMerger

        with open(args.merge_config) as f:
            config_data = yaml.safe_load(f)

        config = ClassWiseMergeConfig.from_dict(config_data)
        merger = ClassWiseMerger(config)

        merged = [merger.merge(events) for events in raw_events_per_file]
        file_pairs = list(zip(merged, gt_per_file))
        eval_result = evaluate_multi_file(
            file_pairs,
            iou_threshold=args.iou_threshold,
            class_names=class_names_dict,
        )

        console.print()
        display_eval_table(eval_result, "Evaluation with Current Config")
        return 0

    # Optimization
    from rf_detr_finetuning.eventprocessor.optimizer import (
        GlobalSearchSpace,
        SearchSpace,
        optimize_default_only,
        optimize_per_class,
        save_config_yaml,
    )

    default_search = SearchSpace(
        delta_time_ms=args.delta_time_values,
        delta_freq_hz=args.delta_freq_values,
        score_strategy=["max"],
    )

    global_search = GlobalSearchSpace(
        score_threshold=args.score_threshold_values,
        min_duration_ms=args.min_duration_values,
    )

    total_combos = default_search.num_combinations * global_search.num_combinations
    console.print(f"\n[bold]Optimizing merge parameters ({total_combos} combinations)...[/bold]")

    # Progress tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        if args.per_class:
            class_ids = sorted(label_to_class.values())
            per_class_combos = default_search.num_combinations * len(class_ids)
            total_with_per_class = total_combos + per_class_combos
            task_id = progress.add_task("Optimizing...", total=total_with_per_class)

            def progress_cb(phase: str, current: int, total: int) -> None:
                if phase == "defaults":
                    progress.update(task_id, completed=current, description=f"Phase 1: defaults ({current}/{total})")
                else:
                    base = total_combos
                    progress.update(
                        task_id,
                        completed=base + current,
                        description=f"Phase 2: {phase} ({current}/{total})",
                    )

            result = optimize_per_class(
                raw_events_per_file,
                gt_per_file,
                class_ids=class_ids,
                default_search=default_search,
                global_search=global_search,
                iou_threshold=args.iou_threshold,
                class_names=class_names_dict,
                progress_callback=progress_cb,
            )
        else:
            task_id = progress.add_task("Optimizing...", total=total_combos)

            def progress_cb_default(current: int, total: int) -> None:
                progress.update(task_id, completed=current, description=f"Searching ({current}/{total})")

            result = optimize_default_only(
                raw_events_per_file,
                gt_per_file,
                default_search=default_search,
                global_search=global_search,
                iou_threshold=args.iou_threshold,
                class_names=class_names_dict,
                progress_callback=progress_cb_default,
            )

    # Display results
    if result.best_config and result.best_eval:
        console.print(f"\n[bold green]Optimization complete! Best F1: {result.best_f1:.4f}[/bold green]")
        console.print(f"Trials evaluated: {result.trials_evaluated}")

        console.print()
        display_config_table(result.best_config, class_names_dict)

        console.print()
        display_eval_table(result.best_eval, "Best Configuration Results")

        # Save config
        save_config_yaml(result.best_config, str(args.output), class_names_dict)
        console.print(f"\n[green]Optimized config saved to:[/green] {args.output}")

        # Save detailed results
        if args.results_json:
            args.results_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"[green]Detailed results saved to:[/green] {args.results_json}")
    else:
        console.print("[red]Optimization failed: no valid configuration found[/red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
