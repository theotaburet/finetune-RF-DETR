#!/usr/bin/env python3
"""Data preprocessing CLI for RF-DETR audio detection.

Converts audio files and their annotations into COCO-format spectrogram chunks
ready for training.

Usage:
    # Process audio directory with metadata
    python run_dataprocessing.py \
        --audio-dir data/audio/ \
        --metadata-dir data/metadata/ \
        --output-dir data/chunked_coco/ \
        --config config/audio_chunking.yaml

    # Process single audio file
    python run_dataprocessing.py \
        --audio data/audio/sample.flac \
        --metadata data/metadata/sample.json \
        --output-dir data/chunked_coco/

    # Visualize chunks (debug mode)
    python run_dataprocessing.py \
        --audio-dir data/audio/ \
        --metadata-dir data/metadata/ \
        --output-dir data/chunked_coco/ \
        --visualize --debug-dir output/debug/

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

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
        description="Preprocess audio files into COCO-format spectrogram chunks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory containing audio files",
    )
    input_group.add_argument(
        "--audio",
        type=Path,
        help="Single audio file to process",
    )
    input_group.add_argument(
        "--metadata-dir",
        type=Path,
        help="Directory containing metadata JSON files",
    )
    input_group.add_argument(
        "--metadata",
        type=Path,
        help="Single metadata JSON file",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for COCO dataset",
    )
    output_group.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios",
    )
    output_group.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split data, output all to single folder",
    )

    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
        help="FFT hop duration in milliseconds",
    )
    config_group.add_argument(
        "--fft-ms",
        type=float,
        default=25.0,
        help="FFT window duration in milliseconds",
    )
    config_group.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands",
    )
    config_group.add_argument(
        "--window-duration-ms",
        type=float,
        default=6400.0,
        help="Chunk window duration in milliseconds",
    )
    config_group.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.2,
        help="Overlap ratio between chunks (0.0-1.0)",
    )
    config_group.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Target image size (width and height)",
    )
    config_group.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.3,
        help="Minimum overlap ratio for including an event",
    )

    # Preprocessing options
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument(
        "--agc",
        action="store_true",
        help="Apply Automatic Gain Control",
    )
    preproc_group.add_argument(
        "--agc-target-db",
        type=float,
        default=-20.0,
        help="AGC target level in dB",
    )
    preproc_group.add_argument(
        "--detrend",
        action="store_true",
        help="Apply detrending to audio",
    )
    preproc_group.add_argument(
        "--preemphasis",
        type=float,
        default=0.0,
        help="Preemphasis coefficient (0 to disable)",
    )

    # Debug options
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images with bounding boxes",
    )
    debug_group.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory for debug visualizations",
    )
    debug_group.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)",
    )
    debug_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[Any, Any, Any, dict]:
    """Load configuration from file or arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple of (fft_config, chunk_config, preprocessing_config, spectrogram_config).

    """
    from rf_detr_finetuning.dataprocessor import (
        AGCConfig,
        ChunkConfig,
        PreprocessingConfig,
        TimeBasedFFTConfig,
        load_chunking_config_from_yaml,
    )

    if args.config and args.config.exists():
        logger.info(f"Loading config from {args.config}")
        return load_chunking_config_from_yaml(args.config)

    # Build config from CLI arguments
    fft_config = TimeBasedFFTConfig(
        hop_ms=args.hop_ms,
        fft_ms=args.fft_ms,
        n_mels=args.n_mels,
    )

    chunk_config = ChunkConfig(
        window_duration_ms=args.window_duration_ms,
        overlap_ratio=args.overlap_ratio,
        min_overlap_with_event_ratio=args.min_overlap_ratio,
        target_width=args.target_size,
        target_height=args.target_size,
    )

    agc_config = None
    if args.agc:
        agc_config = AGCConfig(
            enabled=True,
            target_db=args.agc_target_db,
        )

    preproc_config = PreprocessingConfig(
        agc=agc_config,
        detrend=args.detrend,
        preemphasis_coef=args.preemphasis if args.preemphasis > 0 else None,
    )

    return fft_config, chunk_config, preproc_config, {"freq_scale": "mel", "fmin": 0.0, "fmax": None}


def find_audio_metadata_pairs(
    audio_dir: Path | None,
    audio_file: Path | None,
    metadata_dir: Path | None,
    metadata_file: Path | None,
) -> list[tuple[Path, Path]]:
    """Find matching audio and metadata file pairs.

    Args:
        audio_dir: Directory of audio files.
        audio_file: Single audio file.
        metadata_dir: Directory of metadata files.
        metadata_file: Single metadata file.

    Returns:
        List of (audio_path, metadata_path) tuples.

    """
    pairs = []

    if audio_file and metadata_file:
        # Single file mode
        if audio_file.exists() and metadata_file.exists():
            pairs.append((audio_file, metadata_file))
        return pairs

    if not audio_dir or not audio_dir.exists():
        return pairs

    # Find all audio files
    audio_extensions = [".flac", ".wav", ".mp3", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))

    # Match with metadata
    for audio_path in sorted(audio_files):
        stem = audio_path.stem

        # Try to find matching metadata
        metadata_path = None
        if metadata_dir and metadata_dir.exists():
            for ext in [".json", ".JSON"]:
                candidate = metadata_dir / f"{stem}{ext}"
                if candidate.exists():
                    metadata_path = candidate
                    break

        if metadata_path:
            pairs.append((audio_path, metadata_path))
        else:
            logger.warning(f"No metadata found for {audio_path.name}")

    return pairs


def load_events_from_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load events from metadata JSON file.

    Args:
        metadata_path: Path to metadata JSON.

    Returns:
        List of event dictionaries.

    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    events = metadata.get("events", [])

    # Normalize event format
    normalized = []
    for event in events:
        # Handle different field names
        start_ms = event.get("start_ms") or event.get("start_time_ms") or event.get("onset_ms", 0)
        end_ms = event.get("end_ms") or event.get("end_time_ms") or event.get("offset_ms", 0)

        # Extract label (prefer label_hierarchy's last part)
        hierarchy = event.get("label_hierarchy", "")
        if isinstance(hierarchy, str) and hierarchy:
            label = hierarchy.split(" + ")[-1]
        else:
            label = event.get("annotation") or event.get("label", "unknown")

        # Get frequency bounds if available
        min_freq = event.get("min_freq_hz") or event.get("low_freq_hz")
        max_freq = event.get("max_freq_hz") or event.get("high_freq_hz")

        normalized.append(
            {
                "start_ms": float(start_ms),
                "end_ms": float(end_ms),
                "label": label,
                "min_freq_hz": float(min_freq) if min_freq else None,
                "max_freq_hz": float(max_freq) if max_freq else None,
                "is_file_level": event.get("is_file_level", False),
            }
        )

    return normalized


def save_chunk_as_image(
    spectrogram: np.ndarray,
    output_path: Path,
) -> None:
    """Save spectrogram chunk as PNG image.

    Args:
        spectrogram: 2D spectrogram array.
        output_path: Output file path.

    """
    from rf_detr_finetuning.dataprocessor import grayscale_to_rgb

    # Convert to RGB (already uint8 0-255 from chunker)
    rgb = grayscale_to_rgb(spectrogram)

    # Save as PNG
    img = Image.fromarray(rgb)
    img.save(output_path)


def save_debug_visualization(
    spectrogram: np.ndarray,
    bboxes: list,
    output_path: Path,
    class_names: dict[int, str] | None = None,
) -> None:
    """Save visualization with bounding boxes drawn.

    Args:
        spectrogram: 2D spectrogram array.
        bboxes: List of ChunkBbox objects.
        output_path: Output file path.
        class_names: Optional class ID to name mapping.

    """
    from rf_detr_finetuning.audio_chunking import draw_bboxes_on_spectrogram

    # Draw bboxes on spectrogram
    img = draw_bboxes_on_spectrogram(spectrogram, bboxes, class_names=class_names)
    img.save(output_path)


def build_coco_dataset(
    chunks_data: list[dict],
    categories: dict[str, int],
) -> dict:
    """Build COCO-format dataset dictionary.

    Args:
        chunks_data: List of chunk dictionaries with image info and annotations.
        categories: Mapping of category name to ID.

    Returns:
        COCO format dictionary.

    """
    images = []
    annotations = []
    ann_id = 1

    for chunk in chunks_data:
        images.append(
            {
                "id": chunk["image_id"],
                "file_name": chunk["file_name"],
                "width": chunk["width"],
                "height": chunk["height"],
            }
        )

        for bbox in chunk["bboxes"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": chunk["image_id"],
                    "category_id": bbox["category_id"],
                    "bbox": bbox["bbox"],  # COCO format: [x, y, width, height]
                    "area": bbox["bbox"][2] * bbox["bbox"][3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_categories = [
        {"id": cat_id, "name": cat_name} for cat_name, cat_id in sorted(categories.items(), key=lambda x: x[1])
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }


def main() -> int:
    """Main preprocessing entrypoint."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Audio Data Preprocessing[/bold blue]\n")

    # Validate inputs
    if not args.audio_dir and not args.audio:
        console.print("[red]Error:[/red] Either --audio-dir or --audio must be specified")
        return 1

    # Load configuration
    fft_config, chunk_config, preproc_config, spec_config = load_config(args)

    console.print(
        f"[cyan]FFT config:[/cyan] hop={fft_config.hop_ms}ms, fft={fft_config.fft_ms}ms, mels={fft_config.n_mels}"
    )
    console.print(
        f"[cyan]Chunk config:[/cyan] window={chunk_config.window_duration_ms}ms, overlap={chunk_config.overlap_ratio}"
    )
    console.print(f"[cyan]Target size:[/cyan] {chunk_config.target_width}x{chunk_config.target_height}")
    console.print()

    # Find audio-metadata pairs
    pairs = find_audio_metadata_pairs(
        args.audio_dir,
        args.audio,
        args.metadata_dir,
        args.metadata,
    )

    if not pairs:
        console.print("[red]Error:[/red] No audio-metadata pairs found")
        return 1

    if args.max_files:
        pairs = pairs[: args.max_files]

    console.print(f"[green]Found {len(pairs)} audio files with metadata[/green]\n")

    # Create output directories
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        debug_dir = args.debug_dir or (output_dir / "debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Create chunker
    from rf_detr_finetuning.dataprocessor import AudioChunker, load_audio_file

    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preproc_config,
        freq_scale=spec_config["freq_scale"],
        fmin=spec_config["fmin"],
        fmax=spec_config["fmax"],
    )

    # Process all files
    all_chunks_data = []
    categories: dict[str, int] = {}
    image_id = 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing audio files...", total=len(pairs))

        for audio_path, metadata_path in pairs:
            progress.update(task, description=f"Processing {audio_path.name}...")

            try:
                # Load events
                raw_events = load_events_from_metadata(metadata_path)

                # Load audio
                audio, sample_rate = load_audio_file(audio_path)
                fmax = chunker.fmax if chunker.fmax else sample_rate / 2

                # Map events to chunker format with category IDs
                events = []
                for event in raw_events:
                    label = event.get("label", "unknown")
                    if label not in categories:
                        categories[label] = len(categories) + 1
                    cat_id = categories[label]

                    hz_min = event.get("min_freq_hz")
                    hz_max = event.get("max_freq_hz")

                    events.append(
                        {
                            "time_start_ms": event.get("start_ms", 0.0),
                            "time_end_ms": event.get("end_ms", 0.0),
                            "hz_min": hz_min if hz_min is not None else chunker.fmin,
                            "hz_max": hz_max if hz_max is not None else fmax,
                            "category": label,
                            "category_id": cat_id,
                            "is_file_level": event.get("is_file_level", False),
                        }
                    )

                # Process with chunker
                chunks = chunker.chunk_audio(audio, sample_rate, events, source_uuid=audio_path.stem)

                logger.debug(f"Generated {len(chunks)} chunks from {audio_path.name}")

                # Save each chunk
                for chunk in chunks:
                    # Generate filename
                    chunk_filename = f"{audio_path.stem}_chunk{chunk.chunk_index:04d}.png"
                    image_path = images_dir / chunk_filename

                    # Save spectrogram as image
                    save_chunk_as_image(chunk.spectrogram, image_path)

                    # Build bbox annotations
                    chunk_bboxes = []
                    for bbox in chunk.bboxes:
                        chunk_bboxes.append(
                            {
                                "category_id": bbox.category_id,
                                "bbox": bbox.to_coco_bbox(),  # [x, y, w, h]
                            }
                        )

                    all_chunks_data.append(
                        {
                            "image_id": image_id,
                            "file_name": chunk_filename,
                            "width": chunk.spectrogram.shape[1],
                            "height": chunk.spectrogram.shape[0],
                            "bboxes": chunk_bboxes,
                            "source_audio": str(audio_path),
                            "start_ms": chunk.start_ms,
                            "end_ms": chunk.end_ms,
                        }
                    )

                    # Save debug visualization if requested
                    if args.visualize:
                        debug_path = debug_dir / f"debug_{chunk_filename}"
                        class_names = {v: k for k, v in categories.items()}
                        save_debug_visualization(
                            chunk.spectrogram,
                            chunk.bboxes,
                            debug_path,
                            class_names,
                        )

                    image_id += 1

            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

            progress.advance(task)

    console.print()

    # Build and save COCO dataset
    if args.no_split:
        # Single output file
        coco_data = build_coco_dataset(all_chunks_data, categories)
        annotations_path = output_dir / "_annotations.coco.json"
        with open(annotations_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        console.print(f"[green]Saved annotations to:[/green] {annotations_path}")

    else:
        # Split into train/val/test
        import random
        import shutil

        random.seed(42)
        random.shuffle(all_chunks_data)

        n = len(all_chunks_data)
        train_ratio, val_ratio, test_ratio = args.split_ratios
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": all_chunks_data[:n_train],
            "valid": all_chunks_data[n_train : n_train + n_val],
            "test": all_chunks_data[n_train + n_val :],
        }

        for split_name, split_data in splits.items():
            if not split_data:
                continue

            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Copy images to split directory (non-destructive)
            for chunk in split_data:
                src = images_dir / chunk["file_name"]
                dst = split_dir / chunk["file_name"]
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

            # Update file names and save annotations
            coco_data = build_coco_dataset(split_data, categories)
            annotations_path = split_dir / "_annotations.coco.json"
            with open(annotations_path, "w") as f:
                json.dump(coco_data, f, indent=2)

            console.print(f"[green]{split_name}:[/green] {len(split_data)} images → {split_dir}")

    # Summary
    console.print()
    console.print("[bold green]Processing complete![/bold green]")
    console.print(f"  Total chunks: {len(all_chunks_data)}")
    console.print(f"  Total annotations: {sum(len(c['bboxes']) for c in all_chunks_data)}")
    console.print(f"  Categories: {list(categories.keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
