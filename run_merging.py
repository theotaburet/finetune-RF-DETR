#!/usr/bin/env python3
"""Event merging CLI for RF-DETR audio detection.

Merges detections from overlapping chunks into coherent events using
class-wise parameters (Delta_Time, Delta_Hz).

Usage:
    # Merge events using config file
    python run_merging.py \
        --input events.json \
        --config config/merging.yaml \
        --output-dir output/merged/

    # Merge with inline parameters
    python run_merging.py \
        --input events.json \
        --output-dir output/merged/ \
        --delta-time 500 \
        --delta-freq 500

    # Use class-wise parameters from config
    python run_merging.py \
        --input predictions.json \
        --config config/merging.yaml \
        --visualize

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rf_detr_finetuning.eventprocessor import EventList
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
    ClassWiseMerger,
)

# Configure logging with Rich
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
        description="Merge overlapping chunk detections into coherent events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input events JSON file (required)",
    )

    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--delta-time",
        type=float,
        default=500.0,
        help="Default delta time in ms (if no config file)",
    )
    config_group.add_argument(
        "--delta-freq",
        type=float,
        default=500.0,
        help="Default delta frequency in Hz (if no config file)",
    )
    config_group.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score to keep events",
    )
    config_group.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum event duration in ms",
    )
    config_group.add_argument(
        "--score-strategy",
        type=str,
        default="max",
        choices=["max", "avg", "weighted"],
        help="Score combination strategy",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for merged events",
    )
    output_group.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format",
    )

    # Debug options
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Create visualization of merging process",
    )
    debug_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_merge_config(args: argparse.Namespace) -> ClassWiseMergeConfig:
    """Load merge configuration from file or arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        ClassWiseMergeConfig instance.

    """
    if args.config and args.config.exists():
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            config_data = yaml.safe_load(f)

        return _parse_config_dict(config_data)

    # Build config from CLI arguments
    default_params = ClassMergeParams(
        delta_time_ms=args.delta_time,
        delta_freq_hz=args.delta_freq,
        score_strategy=args.score_strategy,
    )

    return ClassWiseMergeConfig(
        class_params={},  # No class-specific params from CLI
        default_params=default_params,
        score_threshold=args.score_threshold,
        min_duration_ms=args.min_duration,
    )


def _parse_config_dict(config_data: dict[str, Any]) -> ClassWiseMergeConfig:
    """Parse configuration from dictionary.

    Args:
        config_data: Configuration dictionary from YAML.

    Returns:
        ClassWiseMergeConfig instance.

    """
    # Parse default params
    default_data = config_data.get("default", {})
    default_params = ClassMergeParams(
        delta_time_ms=default_data.get("delta_time_ms", 500.0),
        delta_freq_hz=default_data.get("delta_freq_hz", 500.0),
        min_overlap_ratio=default_data.get("min_overlap_ratio"),
        score_strategy=default_data.get("score_strategy", "max"),
    )

    # Parse class-specific params
    class_params = {}
    classes_data = config_data.get("classes", {})
    for class_id_str, class_data in classes_data.items():
        class_id = int(class_id_str)
        class_params[class_id] = ClassMergeParams(
            delta_time_ms=class_data.get("delta_time_ms", default_params.delta_time_ms),
            delta_freq_hz=class_data.get("delta_freq_hz", default_params.delta_freq_hz),
            min_overlap_ratio=class_data.get("min_overlap_ratio"),
            score_strategy=class_data.get("score_strategy", default_params.score_strategy),
        )

    # Parse filtering params
    filtering_data = config_data.get("filtering", {})
    score_threshold = filtering_data.get("score_threshold", 0.0)
    min_duration_ms = filtering_data.get("min_duration_ms", 0.0)
    max_duration_ms = filtering_data.get("max_duration_ms")

    return ClassWiseMergeConfig(
        class_params=class_params,
        default_params=default_params,
        score_threshold=score_threshold,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
    )


def load_events(input_path: Path) -> EventList:
    """Load events from JSON file.

    Args:
        input_path: Path to events JSON file.

    Returns:
        EventList instance.

    """
    logger.info(f"Loading events from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    # Try loading as EventList format first
    if "events" in data:
        return EventList.load(input_path)

    # Try loading as COCO format
    if "annotations" in data:
        return _load_coco_format(data, input_path)

    raise ValueError(f"Unknown events format in {input_path}")


def _load_coco_format(data: dict, input_path: Path) -> EventList:
    """Convert COCO format to EventList.

    Args:
        data: COCO format dictionary.
        input_path: Source file path.

    Returns:
        EventList instance.

    """
    from rf_detr_finetuning.eventprocessor import AudioEvent

    events = []
    annotations = data.get("annotations", [])
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Group annotations by image
    anns_by_image: dict[int, list] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Convert to events
    for img_id, anns in anns_by_image.items():
        img_info = images.get(img_id, {})
        start_ms = img_info.get("start_ms", 0.0)

        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
            x, y, w, h = bbox

            # Convert bbox to time/freq (approximate)
            # Assuming bbox is in pixels, need to know image dimensions
            img_width = img_info.get("width", 640)
            img_height = img_info.get("height", 640)

            # Temporal extent from bbox x coordinate
            # This is approximate - actual conversion depends on your chunking params
            time_start_ms = start_ms + (x / img_width) * 3200  # Assuming 3200ms chunks
            time_end_ms = start_ms + ((x + w) / img_width) * 3200

            # Frequency extent from bbox y coordinate
            # Assuming spectrogram covers 0-Nyquist Hz
            freq_max = (1 - y / img_height) * 24000  # Assuming 48kHz sample rate
            freq_min = (1 - (y + h) / img_height) * 24000

            category_id = ann.get("category_id", 0)
            score = ann.get("score", ann.get("confidence", 1.0))

            event = AudioEvent(
                start_ms=time_start_ms,
                end_ms=time_end_ms,
                class_id=category_id,
                class_name=categories.get(category_id),
                score=score,
                min_freq_hz=max(0, freq_min),
                max_freq_hz=freq_max,
                source_windows=[img_id],
            )
            events.append(event)

    return EventList(
        events=events,
        audio_path=str(input_path),
        class_names=categories,
    )


def save_events(events: EventList, output_path: Path, format: str = "json") -> None:
    """Save merged events to file.

    Args:
        events: Merged EventList.
        output_path: Output file path.
        format: Output format (json or csv).

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        events.save(output_path)
        logger.info(f"Saved merged events to {output_path}")
    elif format == "csv":
        _save_csv(events, output_path)
        logger.info(f"Saved merged events to {output_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def _save_csv(events: EventList, output_path: Path) -> None:
    """Save events to CSV format.

    Args:
        events: EventList to save.
        output_path: Output CSV file path.

    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_ms",
                "end_ms",
                "duration_ms",
                "class_id",
                "class_name",
                "score",
                "min_freq_hz",
                "max_freq_hz",
            ]
        )

        for event in events:
            writer.writerow(
                [
                    event.start_ms,
                    event.end_ms,
                    event.duration_ms,
                    event.class_id,
                    event.class_name,
                    event.score,
                    event.min_freq_hz,
                    event.max_freq_hz,
                ]
            )


def print_config_table(config: ClassWiseMergeConfig) -> None:
    """Print configuration as a nice table.

    Args:
        config: Merge configuration.

    """
    console.print("\n[bold cyan]Merge Configuration[/bold cyan]")

    # Default params
    table = Table(title="Default Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Delta Time", f"{config.default_params.delta_time_ms} ms")
    table.add_row("Delta Frequency", f"{config.default_params.delta_freq_hz} Hz")
    table.add_row("Score Strategy", config.default_params.score_strategy)
    table.add_row("Score Threshold", str(config.score_threshold))
    table.add_row("Min Duration", f"{config.min_duration_ms} ms")

    console.print(table)

    # Class-specific params
    if config.class_params:
        console.print("\n[bold cyan]Class-Specific Parameters[/bold cyan]")
        class_table = Table(title="Per-Class Settings")
        class_table.add_column("Class ID", style="cyan")
        class_table.add_column("Delta Time (ms)", style="green")
        class_table.add_column("Delta Freq (Hz)", style="green")
        class_table.add_column("Score Strategy", style="yellow")

        for class_id, params in sorted(config.class_params.items()):
            class_table.add_row(
                str(class_id),
                str(params.delta_time_ms),
                str(params.delta_freq_hz),
                params.score_strategy,
            )

        console.print(class_table)


def print_merge_summary(before: EventList, after: EventList) -> None:
    """Print summary of merging results.

    Args:
        before: Events before merging.
        after: Events after merging.

    """
    console.print("\n[bold green]Merge Results[/bold green]")

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    reduction = 100 * (1 - len(after) / len(before)) if len(before) > 0 else 0

    table.add_row("Input Events", str(len(before)))
    table.add_row("Merged Events", str(len(after)))
    table.add_row("Reduction", f"{reduction:.1f}%")

    # Per-class stats
    before_counts = before.get_class_counts()
    after_counts = after.get_class_counts()

    for class_id in sorted(set(list(before_counts.keys()) + list(after_counts.keys()))):
        before_count = before_counts.get(class_id, 0)
        after_count = after_counts.get(class_id, 0)
        class_name = after.class_names.get(class_id, f"class_{class_id}")
        table.add_row(
            f"  {class_name}",
            f"{before_count} → {after_count}",
        )

    console.print(table)


def visualize_merging(
    before: EventList,
    after: EventList,
    output_path: Path,
) -> None:
    """Create visualization of merging process.

    Args:
        before: Events before merging.
        after: Events after merging.
        output_path: Output image path.

    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Colors for classes
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    # Top plot: Before merging
    ax1 = axes[0]
    ax1.set_title("Before Merging", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Event")

    for i, event in enumerate(before.events):
        color = colors[event.class_id % len(colors)]
        rect = mpatches.Rectangle(
            (event.start_ms, i - 0.3),
            event.end_ms - event.start_ms,
            0.6,
            facecolor=color,
            edgecolor="black",
            alpha=0.7,
        )
        ax1.add_patch(rect)

        mid_time = (event.start_ms + event.end_ms) / 2
        ax1.text(
            mid_time,
            i,
            f"{event.score:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if event.score > 0.5 else "black",
        )

    ax1.set_xlim(
        min(e.start_ms for e in before.events) - 100,
        max(e.end_ms for e in before.events) + 100,
    )
    ax1.set_ylim(-0.5, len(before) - 0.5)
    ax1.set_yticks(range(len(before)))
    ax1.grid(True, axis="x", alpha=0.3)

    # Bottom plot: After merging
    ax2 = axes[1]
    ax2.set_title("After Merging", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Event")

    for i, event in enumerate(after.events):
        color = colors[event.class_id % len(colors)]
        rect = mpatches.Rectangle(
            (event.start_ms, i - 0.3),
            event.end_ms - event.start_ms,
            0.6,
            facecolor=color,
            edgecolor="black",
            alpha=0.7,
        )
        ax2.add_patch(rect)

        mid_time = (event.start_ms + event.end_ms) / 2
        merged_count = event.metadata.get("merged_count", 1)
        ax2.text(
            mid_time,
            i,
            f"{event.score:.2f}\n({merged_count})",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if event.score > 0.5 else "black",
        )

    ax2.set_xlim(
        min(e.start_ms for e in before.events) - 100,
        max(e.end_ms for e in before.events) + 100,
    )
    ax2.set_ylim(-0.5, len(after) - 0.5)
    ax2.set_yticks(range(len(after)))
    ax2.grid(True, axis="x", alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors[i], label=after.class_names.get(i, f"class_{i}"))
        for i in set(e.class_id for e in before.events)
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Event Merging[/bold blue]\n")

    try:
        # Load configuration
        config = load_merge_config(args)
        print_config_table(config)

        # Load input events
        events = load_events(args.input)
        logger.info(f"Loaded {len(events)} events from {args.input}")

        # Apply merging
        console.print("\n[bold]Merging events...[/bold]")
        merger = ClassWiseMerger(config)
        merged_events = merger.merge(events)

        # Print summary
        print_merge_summary(events, merged_events)

        # Save results
        output_file = args.output_dir / f"merged_events.{args.format}"
        save_events(merged_events, output_file, args.format)

        # Create visualization if requested
        if args.visualize:
            viz_path = args.output_dir / "merging_visualization.png"
            visualize_merging(events, merged_events, viz_path)

        console.print("\n[bold green]Done![/bold green]")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
