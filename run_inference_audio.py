#!/usr/bin/env python3
"""Audio inference pipeline for RF-DETR detection.

Full pipeline: Audio → Chunks → Spectrograms → Detection → Event Merging → Output

Usage:
    # Single audio file
    python run_inference_audio.py \
        --audio data/audio/sample.flac \
        --weights output/checkpoint_best.pth \
        --config config/chunking.yaml \
        --output results/sample_events.json

    # Directory of audio files
    python run_inference_audio.py \
        --audio-dir data/audio/ \
        --weights output/checkpoint_best.pth \
        --config config/chunking.yaml \
        --output-dir results/

    # With visualization
    python run_inference_audio.py \
        --audio data/audio/sample.flac \
        --weights output/checkpoint_best.pth \
        --config config/chunking.yaml \
        --visualize --viz-dir output/viz/

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rf_detr_finetuning.utils import load_class_names

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
        description="Run RF-DETR detection on audio files with full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--audio",
        type=Path,
        help="Single audio file to process",
    )
    input_group.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory of audio files to process",
    )
    input_group.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".flac", ".wav", ".mp3"],
        help="Audio file extensions to process",
    )

    # Model options
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
        help="Device for inference (cuda, cpu)",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Audio chunking configuration YAML",
    )
    proc_group.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold",
    )
    proc_group.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS merging",
    )
    proc_group.add_argument(
        "--merge-gap-ms",
        type=float,
        default=100.0,
        help="Maximum gap (ms) for temporal merging",
    )
    proc_group.add_argument(
        "--merge-config",
        type=Path,
        help="Event merging configuration YAML (uses class-wise merging if provided)",
    )
    proc_group.add_argument(
        "--min-event-duration-ms",
        type=float,
        default=0.0,
        help="Minimum event duration to keep",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for single file results",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch results",
    )
    output_group.add_argument(
        "--format",
        choices=["json", "csv", "raven"],
        default="json",
        help="Output format",
    )

    # Class names
    output_group.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        help="List of class names (order matches class IDs)",
    )
    output_group.add_argument(
        "--classes-file",
        type=Path,
        help="JSON file with class names or COCO annotations",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images",
    )
    viz_group.add_argument(
        "--viz-dir",
        type=Path,
        help="Directory for visualizations",
    )
    viz_group.add_argument(
        "--save-chunks",
        action="store_true",
        help="Save individual chunk images with detections",
    )

    # Debug options
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    debug_group.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process",
    )
    debug_group.add_argument(
        "--max-duration",
        type=float,
        help="Skip audio files longer than this duration in seconds (for testing)",
    )

    return parser.parse_args()


@dataclass
class AudioInferenceResult:
    """Result from audio inference pipeline.

    Attributes:
        audio_path: Source audio file path.
        duration_ms: Audio duration in milliseconds.
        sample_rate: Audio sample rate.
        num_windows: Number of spectrogram windows processed.
        events: List of detected events.
        window_detections: Raw per-window detections (optional).

    """

    audio_path: Path
    duration_ms: float
    sample_rate: int
    num_windows: int
    events: list[dict[str, Any]] = field(default_factory=list)
    window_detections: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "audio_path": str(self.audio_path),
            "duration_ms": self.duration_ms,
            "duration_s": self.duration_ms / 1000,
            "sample_rate": self.sample_rate,
            "num_windows": self.num_windows,
            "num_events": len(self.events),
            "events": self.events,
        }

    def to_raven(self) -> str:
        """Convert to Raven selection table format."""
        lines = [
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\tScore"
        ]
        for i, event in enumerate(self.events, 1):
            start_s = event["start_ms"] / 1000
            end_s = event["end_ms"] / 1000
            low_freq = event.get("min_freq_hz", "")
            high_freq = event.get("max_freq_hz", "")
            annotation = event.get("class_name", event.get("class_id", ""))
            score = event.get("score", "")

            lines.append(
                f"{i}\tSpectrogram 1\t1\t{start_s:.4f}\t{end_s:.4f}\t{low_freq}\t{high_freq}\t{annotation}\t{score:.3f}"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Convert to CSV format."""
        lines = ["audio_path,start_ms,end_ms,duration_ms,class_id,class_name,score,min_freq_hz,max_freq_hz"]
        for event in self.events:
            duration = event["end_ms"] - event["start_ms"]
            lines.append(
                f"{self.audio_path},{event['start_ms']:.1f},{event['end_ms']:.1f},{duration:.1f},"
                f"{event.get('class_id', '')},"
                f"{event.get('class_name', '')},"
                f"{event.get('score', ''):.3f},"
                f"{event.get('min_freq_hz', '')},"
                f"{event.get('max_freq_hz', '')}"
            )
        return "\n".join(lines)


class AudioInferencePipeline:
    """Full audio inference pipeline.

    Handles: Audio loading → Chunking → Spectrogram → Detection → Merging

    """

    def __init__(
        self,
        weights_path: Path,
        config_path: Path,
        model_size: str = "base",
        device: str = "cuda",
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        merge_gap_ms: float = 100.0,
        min_event_duration_ms: float = 0.0,
        merge_config_path: Path | None = None,
    ) -> None:
        """Initialize the inference pipeline.

        Args:
            weights_path: Path to model checkpoint.
            config_path: Path to chunking config YAML.
            model_size: RF-DETR model size.
            device: Inference device.
            class_names: Optional list of class names.
            confidence_threshold: Minimum detection confidence.
            iou_threshold: IoU threshold for NMS.
            merge_gap_ms: Gap tolerance for temporal merging.
            min_event_duration_ms: Minimum event duration.
            merge_config_path: Optional path to class-wise merge config YAML.

        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.model_size = model_size
        self.device = device
        self.class_names = class_names or []
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.merge_gap_ms = merge_gap_ms
        self.min_event_duration_ms = min_event_duration_ms
        self.merge_config_path = merge_config_path

        # Load components
        self._init_chunker()
        self._init_predictor()
        self._init_postprocessor()

    def _init_chunker(self) -> None:
        """Initialize audio chunker from config."""
        from rf_detr_finetuning.dataprocessor import (
            AudioChunker,
            load_chunking_config_from_yaml,
        )

        self.fft_config, self.chunk_config, self.preproc_config = load_chunking_config_from_yaml(self.config_path)

        self.chunker = AudioChunker(
            fft_config=self.fft_config,
            chunk_config=self.chunk_config,
            preprocessing_config=self.preproc_config,
        )

        logger.debug(f"Chunker initialized: window={self.chunk_config.window_duration_ms}ms")

    def _init_predictor(self) -> None:
        """Initialize RF-DETR predictor."""
        from rf_detr_finetuning.predictor import RFDETRPredictor

        self.predictor = RFDETRPredictor(
            model_size=self.model_size,
            weights_path=self.weights_path,
            device=self.device,
            class_names=self.class_names,
        )

        logger.debug(f"Predictor initialized: {self.model_size} on {self.device}")

    def _init_postprocessor(self) -> None:
        """Initialize event post-processor."""
        from rf_detr_finetuning.eventprocessor import (
            EventPostProcessor,
            MergeConfig,
            PostProcessorConfig,
        )

        class_names_dict = {i: name for i, name in enumerate(self.class_names)}

        # Load class-wise merge config if provided, otherwise use IoU-based merging
        if self.merge_config_path and self.merge_config_path.exists():
            logger.info(f"Loading class-wise merge config from {self.merge_config_path}")
            import yaml

            with open(self.merge_config_path) as f:
                merge_config_data = yaml.safe_load(f)
            class_wise_config = self._parse_class_wise_merge_config(merge_config_data)

            self.postproc_config = PostProcessorConfig(
                time_per_pixel_ms=self.fft_config.hop_ms,
                confidence_threshold=self.confidence_threshold,
                min_event_duration_ms=self.min_event_duration_ms,
                class_wise_merge_config=class_wise_config,
                class_names=class_names_dict,
            )
        else:
            # Use traditional IoU-based merging
            merge_config = MergeConfig(
                iou_threshold=self.iou_threshold,
                score_threshold=self.confidence_threshold,
                merge_same_class_only=True,
                merge_strategy="max",
                gap_tolerance_ms=self.merge_gap_ms,
            )

            self.postproc_config = PostProcessorConfig(
                time_per_pixel_ms=self.fft_config.hop_ms,
                confidence_threshold=self.confidence_threshold,
                min_event_duration_ms=self.min_event_duration_ms,
                merge_config=merge_config,
                class_names=class_names_dict,
            )

        self.postprocessor = EventPostProcessor(self.postproc_config)

    def _parse_class_wise_merge_config(self, config_data: dict) -> Any:
        """Parse class-wise merge configuration from dict.

        Args:
            config_data: Configuration dictionary from YAML.

        Returns:
            ClassWiseMergeConfig instance.

        """
        from rf_detr_finetuning.eventprocessor.merger import ClassWiseMergeConfig

        return ClassWiseMergeConfig.from_dict(config_data)

    def process_audio(
        self,
        audio_path: Path,
        return_window_detections: bool = False,
    ) -> AudioInferenceResult:
        """Process a single audio file through the full pipeline.

        Args:
            audio_path: Path to audio file.
            return_window_detections: Include raw per-window detections.

        Returns:
            AudioInferenceResult with detected events.

        """
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb, load_audio_file, normalize_to_range
        from rf_detr_finetuning.predictor import WindowPrediction

        # Load audio
        audio, sr = load_audio_file(str(audio_path))
        duration_ms = len(audio) / sr * 1000

        logger.debug(f"Loaded audio: {duration_ms:.0f}ms @ {sr}Hz")

        # Process into chunks (no events needed for inference)
        chunks = self.chunker.chunk_audio_file(audio_path)

        logger.debug(f"Generated {len(chunks)} chunks")

        # Run inference on each chunk
        window_predictions = []
        for chunk in chunks:
            if chunk.spectrogram is None:
                continue

            # Convert spectrogram to RGB image
            normalized = normalize_to_range(chunk.spectrogram, 0, 255)
            if normalized.ndim == 2:
                rgb = grayscale_to_rgb(normalized.astype(np.uint8))
            else:
                rgb = normalized.astype(np.uint8)

            # Run detection
            result = self.predictor.predict(rgb, self.confidence_threshold)

            # Create window prediction
            wp = WindowPrediction(
                detections=result.detections,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                window_index=chunk.chunk_index,
                spectrogram_shape=chunk.spectrogram.shape,
                is_padded=chunk.is_padded,
            )
            window_predictions.append(wp)

        # Merge window predictions into events
        from rf_detr_finetuning.predictor.audio import AudioPredictionResult

        audio_prediction = AudioPredictionResult(
            window_predictions=window_predictions,
            audio_path=audio_path,
            duration_ms=duration_ms,
            sample_rate=sr,
        )

        event_list = self.postprocessor.process(audio_prediction)

        # Convert events to dict format
        events = [e.to_dict() for e in event_list.events]

        # Build result
        result = AudioInferenceResult(
            audio_path=audio_path,
            duration_ms=duration_ms,
            sample_rate=sr,
            num_windows=len(chunks),
            events=events,
        )

        if return_window_detections:
            result.window_detections = [
                {
                    "window_index": wp.window_index,
                    "start_ms": wp.start_ms,
                    "end_ms": wp.end_ms,
                    "detections": [d.to_dict() for d in wp.detections],
                }
                for wp in window_predictions
            ]

        return result

    def get_chunks_with_detections(
        self,
        audio_path: Path,
    ) -> list[tuple[np.ndarray, list, dict]]:
        """Get chunks with their detections for visualization.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of (spectrogram, detections, metadata) tuples.

        """
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb, normalize_to_range

        # Load and chunk
        chunks = self.chunker.chunk_audio_file(audio_path)

        results = []
        for chunk in chunks:
            if chunk.spectrogram is None:
                continue

            # Convert and detect
            normalized = normalize_to_range(chunk.spectrogram, 0, 255)
            if normalized.ndim == 2:
                rgb = grayscale_to_rgb(normalized.astype(np.uint8))
            else:
                rgb = normalized.astype(np.uint8)
            result = self.predictor.predict(rgb, self.confidence_threshold)

            metadata = {
                "chunk_index": chunk.chunk_index,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "is_padded": chunk.is_padded,
            }

            results.append((chunk.spectrogram, result.detections, metadata))

        return results


def save_visualizations(
    pipeline: AudioInferencePipeline,
    audio_path: Path,
    output_dir: Path,
    save_chunks: bool = False,
) -> None:
    """Save visualization images for an audio file.

    Args:
        pipeline: Inference pipeline.
        audio_path: Audio file path.
        output_dir: Output directory.
        save_chunks: Whether to save individual chunk visualizations.

    """
    try:
        import supervision as sv
    except ImportError:
        logger.warning("supervision not installed, skipping visualizations")
        return

    from rf_detr_finetuning.dataprocessor import grayscale_to_rgb, normalize_to_range

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get chunks with detections
    chunks_data = pipeline.get_chunks_with_detections(audio_path)

    if save_chunks:
        # Save individual chunk images
        for spec, detections, meta in chunks_data:
            if not detections:
                continue

            # Convert to image
            normalized = normalize_to_range(spec, 0, 255).astype(np.uint8)
            rgb = grayscale_to_rgb(normalized)

            # Draw detections
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
            confidence = np.array([d.score for d in detections])
            class_id = np.array([d.class_id for d in detections])

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

            annotator = sv.BoxAnnotator(thickness=2)
            annotated = annotator.annotate(rgb.copy(), sv_detections)

            # Save
            chunk_name = f"{audio_path.stem}_chunk{meta['chunk_index']:04d}_det.png"
            Image.fromarray(annotated).save(output_dir / chunk_name)


def display_results(result: AudioInferenceResult) -> None:
    """Display results in a formatted table.

    Args:
        result: Inference result to display.

    """
    table = Table(title=f"Detections in {result.audio_path.name}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Start (s)", style="cyan", width=10)
    table.add_column("End (s)", style="cyan", width=10)
    table.add_column("Duration", style="green", width=10)
    table.add_column("Class", style="yellow", width=20)
    table.add_column("Score", style="magenta", width=8)

    for i, event in enumerate(result.events, 1):
        start_s = event["start_ms"] / 1000
        end_s = event["end_ms"] / 1000
        duration = event["duration_ms"] / 1000
        class_name = event.get("class_name") or str(event.get("class_id", "?"))
        score = event.get("score", 0)

        table.add_row(
            str(i),
            f"{start_s:.2f}",
            f"{end_s:.2f}",
            f"{duration:.2f}s",
            class_name,
            f"{score:.3f}",
        )

    console.print(table)


def main() -> int:
    """Main audio inference entrypoint."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Audio Inference Pipeline[/bold blue]\n")

    # Validate inputs
    if not args.audio and not args.audio_dir:
        console.print("[red]Error:[/red] Either --audio or --audio-dir must be specified")
        return 1

    if not args.weights.exists():
        console.print(f"[red]Error:[/red] Weights not found: {args.weights}")
        return 1

    if not args.config.exists():
        console.print(f"[red]Error:[/red] Config not found: {args.config}")
        return 1

    # Load class names
    class_names = load_class_names(args.class_names, args.classes_file)
    if class_names:
        console.print(f"[cyan]Classes:[/cyan] {class_names}")

    # Initialize pipeline
    console.print(f"[cyan]Loading model:[/cyan] RF-DETR {args.model_size}")
    console.print(f"[cyan]Weights:[/cyan] {args.weights}")
    console.print(f"[cyan]Config:[/cyan] {args.config}")
    console.print()

    try:
        pipeline = AudioInferencePipeline(
            weights_path=args.weights,
            config_path=args.config,
            model_size=args.model_size,
            device=args.device,
            class_names=class_names,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            merge_gap_ms=args.merge_gap_ms,
            min_event_duration_ms=args.min_event_duration_ms,
            merge_config_path=args.merge_config,
        )
    except Exception as e:
        console.print(f"[red]Error initializing pipeline:[/red] {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Find audio files
    if args.audio:
        audio_files = [args.audio]
    else:
        audio_files = []
        for ext in args.extensions:
            audio_files.extend(args.audio_dir.glob(f"*{ext}"))
            audio_files.extend(args.audio_dir.glob(f"*{ext.upper()}"))
        audio_files = sorted(audio_files)

    if not audio_files:
        console.print("[red]Error:[/red] No audio files found")
        return 1

    if args.max_files:
        audio_files = audio_files[: args.max_files]

    # Filter by duration (reads file headers only, no full load)
    if args.max_duration:
        from rf_detr_finetuning.dataprocessor.io import filter_audio_by_duration

        before_count = len(audio_files)
        audio_files = filter_audio_by_duration(audio_files, args.max_duration)
        if len(audio_files) < before_count:
            console.print(
                f"[yellow]Duration filter ({args.max_duration}s):[/yellow] "
                f"kept {len(audio_files)} / {before_count} files"
            )

    console.print(f"[green]Processing {len(audio_files)} audio file(s)[/green]\n")

    # Process files
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(audio_files))

        for audio_path in audio_files:
            progress.update(task, description=f"Processing {audio_path.name}...")

            try:
                result = pipeline.process_audio(
                    audio_path,
                    return_window_detections=args.verbose,
                )
                all_results.append(result)

                # Visualizations
                if args.visualize:
                    viz_dir = args.viz_dir or Path("output/visualizations")
                    save_visualizations(
                        pipeline,
                        audio_path,
                        viz_dir / audio_path.stem,
                        save_chunks=args.save_chunks,
                    )

            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

            progress.advance(task)

    console.print()

    # Display results for single file
    if len(all_results) == 1:
        display_results(all_results[0])

    # Summary for batch
    total_events = sum(len(r.events) for r in all_results)
    console.print("\n[bold green]Inference complete![/bold green]")
    console.print(f"  Files processed: {len(all_results)}")
    console.print(f"  Total events detected: {total_events}")

    # Save results
    if args.output or args.output_dir:
        output_dir = args.output_dir or args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(all_results) == 1 and args.output:
            # Single file output
            output_path = args.output
            result = all_results[0]

            if args.format == "json":
                with open(output_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
            elif args.format == "csv":
                with open(output_path, "w") as f:
                    f.write(result.to_csv())
            elif args.format == "raven":
                with open(output_path, "w") as f:
                    f.write(result.to_raven())

            console.print(f"[green]Results saved to:[/green] {output_path}")

        else:
            # Batch output
            for result in all_results:
                stem = result.audio_path.stem

                if args.format == "json":
                    output_path = output_dir / f"{stem}_events.json"
                    with open(output_path, "w") as f:
                        json.dump(result.to_dict(), f, indent=2)
                elif args.format == "csv":
                    output_path = output_dir / f"{stem}_events.csv"
                    with open(output_path, "w") as f:
                        f.write(result.to_csv())
                elif args.format == "raven":
                    output_path = output_dir / f"{stem}_events.txt"
                    with open(output_path, "w") as f:
                        f.write(result.to_raven())

            # Also save combined results
            combined_path = output_dir / "all_results.json"
            with open(combined_path, "w") as f:
                json.dump([r.to_dict() for r in all_results], f, indent=2)

            console.print(f"[green]Results saved to:[/green] {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
