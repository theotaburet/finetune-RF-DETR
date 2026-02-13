#!/usr/bin/env python3
"""Inference CLI for RF-DETR audio detection.

Usage:
    # Inference on images
    python run_inference.py --image path/to/spectrogram.png --weights output/checkpoint_best.pth

    # Inference on directory
    python run_inference.py --image-dir data/test_spectrograms/ --weights model.pth

    # Inference on audio file
    python run_inference.py --audio path/to/audio.flac --chunking-config config/chunking.yaml

    # Batch inference on audio files
    python run_inference.py --audio-dir data/audio/ --output results.json

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
)

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with RF-DETR audio detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Single spectrogram image to process",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of spectrogram images",
    )
    input_group.add_argument(
        "--audio",
        type=Path,
        help="Single audio file to process",
    )
    input_group.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory of audio files",
    )

    # Model settings
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to model weights/checkpoint",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large"],
        default="base",
        help="RF-DETR model size",
    )

    # Audio processing settings
    parser.add_argument(
        "--chunking-config",
        type=Path,
        help="Audio chunking configuration YAML",
    )

    # Detection settings
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS merging",
    )
    parser.add_argument(
        "--merge-config",
        type=Path,
        help="Event merging configuration YAML (uses class-wise merging if provided)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images",
    )

    # Class names
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        help="List of class names",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        help="JSON file with class names",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    # Misc
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_class_names(args: argparse.Namespace) -> list[str] | None:
    """Load class names from arguments.

    Args:
        args: Parsed arguments.

    Returns:
        List of class names or None.

    """
    if args.class_names:
        return args.class_names

    if args.classes_file and args.classes_file.exists():
        with open(args.classes_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "categories" in data:
            return [cat["name"] for cat in data["categories"]]

    return None


def _parse_class_wise_merge_config(config_data: dict) -> ClassWiseMergeConfig:
    """Parse class-wise merge configuration from dict.

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


def run_image_inference(args: argparse.Namespace) -> int:
    """Run inference on image(s).

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.

    """
    from rf_detr_finetuning.predictor import RFDETRPredictor, predict_directory

    class_names = load_class_names(args)

    # Create predictor
    predictor = RFDETRPredictor(
        model_size=args.model_size,
        weights_path=args.weights,
        device=args.device,
        class_names=class_names,
    )

    if args.image:
        # Single image
        console.print(f"Processing: {args.image}")
        result = predictor.predict(str(args.image), args.confidence)

        # Display results
        display_detections(result, args.image.name)

        # Save if requested
        if args.output:
            save_results([result], args.output)

    elif args.image_dir:
        # Directory of images
        console.print(f"Processing directory: {args.image_dir}")
        results = predict_directory(
            predictor,
            args.image_dir,
            output_path=args.output,
            confidence_threshold=args.confidence,
        )

        console.print(f"\nProcessed {len(results)} images")
        total_dets = sum(len(r.detections) for r in results)
        console.print(f"Total detections: {total_dets}")

    return 0


def run_audio_inference(args: argparse.Namespace) -> int:
    """Run inference on audio file(s).

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.

    """
    from rf_detr_finetuning.eventprocessor import (
        EventPostProcessor,
        MergeConfig,
        PostProcessorConfig,
    )
    from rf_detr_finetuning.predictor import AudioPredictor, RFDETRPredictor

    # Validate chunking config
    if not args.chunking_config:
        console.print("[red]Error:[/red] --chunking-config required for audio inference")
        return 1

    if not args.chunking_config.exists():
        console.print(f"[red]Error:[/red] Config not found: {args.chunking_config}")
        return 1

    class_names = load_class_names(args)

    # Create predictor
    base_predictor = RFDETRPredictor(
        model_size=args.model_size,
        weights_path=args.weights,
        device=args.device,
        class_names=class_names,
    )

    audio_predictor = AudioPredictor.from_config(
        predictor=base_predictor,
        chunking_config_path=args.chunking_config,
    )

    # Create post-processor
    # Load time_per_pixel from config
    import yaml

    with open(args.chunking_config) as f:
        config = yaml.safe_load(f)

    fft_config = config.get("fft", {})
    time_per_pixel_ms = fft_config.get("hop_ms", 10.0)

    # Load class-wise merge config if provided
    if args.merge_config and args.merge_config.exists():
        console.print(f"[cyan]Loading merge config:[/cyan] {args.merge_config}")
        with open(args.merge_config) as f:
            merge_config_data = yaml.safe_load(f)
        class_wise_config = _parse_class_wise_merge_config(merge_config_data)

        post_config = PostProcessorConfig(
            time_per_pixel_ms=time_per_pixel_ms,
            confidence_threshold=args.confidence,
            class_wise_merge_config=class_wise_config,
            class_names={i: name for i, name in enumerate(class_names)} if class_names else {},
        )
    else:
        # Use traditional IoU-based merging
        post_config = PostProcessorConfig(
            time_per_pixel_ms=time_per_pixel_ms,
            confidence_threshold=args.confidence,
            merge_config=MergeConfig(iou_threshold=args.iou_threshold),
            class_names={i: name for i, name in enumerate(class_names)} if class_names else {},
        )
    post_processor = EventPostProcessor(post_config)

    all_results = []

    if args.audio:
        # Single audio file
        audio_files = [args.audio]
    else:
        # Directory of audio files
        audio_files = sorted(args.audio_dir.glob("*.flac"))
        audio_files.extend(sorted(args.audio_dir.glob("*.wav")))
        audio_files.extend(sorted(args.audio_dir.glob("*.mp3")))

    console.print(f"Processing {len(audio_files)} audio file(s)\n")

    for audio_path in audio_files:
        console.print(f"[cyan]Processing:[/cyan] {audio_path.name}")

        # Run window-level prediction
        window_result = audio_predictor.predict(audio_path, args.confidence)

        # Convert to events
        events = post_processor.process(window_result)

        console.print(f"  Windows: {len(window_result.window_predictions)}, Events: {len(events)}")

        # Display events
        if args.verbose:
            for event in events:
                console.print(
                    f"    [{event.start_ms:.0f}-{event.end_ms:.0f}ms] "
                    f"{event.class_name or event.class_id} "
                    f"(score={event.score:.2f})"
                )

        all_results.append(
            {
                "audio_path": str(audio_path),
                "duration_ms": window_result.duration_ms,
                "num_windows": len(window_result.window_predictions),
                "events": events.to_dict()["events"],
            }
        )

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to:[/green] {args.output}")

    return 0


def display_detections(result, image_name: str) -> None:
    """Display detections in a table.

    Args:
        result: PredictionResult.
        image_name: Image filename.

    """
    table = Table(title=f"Detections in {image_name}")
    table.add_column("#", style="dim")
    table.add_column("Class", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Bbox", style="yellow")

    for i, det in enumerate(result.detections):
        bbox_str = f"[{det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f}]"
        class_str = det.class_name or str(det.class_id)
        table.add_row(str(i + 1), class_str, f"{det.score:.3f}", bbox_str)

    console.print(table)


def save_results(results: list, output_path: Path) -> None:
    """Save results to JSON.

    Args:
        results: List of PredictionResults.
        output_path: Output file path.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for result in results:
        data.append(
            {
                "image_path": str(result.image_path) if result.image_path else None,
                "detections": [d.to_dict() for d in result.detections],
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Results saved to:[/green] {output_path}")


def main() -> int:
    """Main inference entrypoint."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Audio Detection Inference[/bold blue]\n")

    try:
        if args.image or args.image_dir:
            return run_image_inference(args)
        elif args.audio or args.audio_dir:
            return run_audio_inference(args)
        else:
            console.print("[red]Error:[/red] No input specified")
            return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logger.exception("Inference error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
