"""Chunk audio dataset organized by class folders into COCO format.

Usage:
    python scripts/chunk_audio_dataset.py \
        --input-dir ./my_audio_dataset \
        --output-dir ./chunked_coco \
        --config config/audio_chunking.yaml \
        --draw-bboxes

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from rf_detr_finetuning.audio_chunking import (
    AudioChunker,
    draw_bboxes_on_spectrogram,
    load_chunking_config_from_yaml,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def extract_category_from_path(audio_path: Path, root_dir: Path) -> str:
    """Extract category name from folder structure.

    Args:
        audio_path: Path to audio file (e.g., .../biological-cetacean-odontoceti/dolphin_001.flac)
        root_dir: Root dataset directory (e.g., .../my_audio_dataset)

    Returns:
        Category name (e.g., "odontoceti" from "biological-cetacean-odontoceti")

    """
    relative = audio_path.relative_to(root_dir)
    category_folder = relative.parts[0]

    # Extract last part after dashes (biological-cetacean-odontoceti → odontoceti)
    if "-" in category_folder:
        return category_folder.split("-")[-1]
    return category_folder


def chunk_audio_dataset(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    draw_bboxes: bool = False,
    debug_colormap: str | None = None,
    audio_extensions: tuple[str, ...] = (".flac", ".wav", ".mp3"),
) -> None:
    """Chunk audio dataset organized by class folders.

    Args:
        input_dir: Input directory containing class folders with audio files.
        output_dir: Output directory for chunked images and metadata.
        config_path: Path to chunking configuration YAML.
        draw_bboxes: Whether to draw bounding boxes on output images.
        debug_colormap: Optional colormap for debugging visualization.
        audio_extensions: Tuple of allowed audio file extensions.

    Expected structure:
        input_dir/
        ├── biological-cetacean-odontoceti/
        │   ├── dolphin_001.flac
        │   └── dolphin_002.flac
        ├── anthropogenic-ship-cargo/
        │   └── cargo_001.flac
        └── ...

    Output structure:
        output_dir/
        ├── images/
        │   ├── dolphin_001_chunk0000.png
        │   ├── dolphin_001_chunk0001.png
        │   └── ...
        ├── debug/  (if --draw-bboxes)
        │   ├── dolphin_001_chunk0000_debug.png
        │   └── ...
        └── _annotations.coco.json

    Args:
        input_dir: Root directory with class folders
        output_dir: Output directory for chunks
        config_path: Path to chunking config YAML
        draw_bboxes: Whether to draw debug bboxes
        audio_extensions: Tuple of valid audio file extensions

    """
    console = Console()

    # Load config
    fft_config, chunk_config, preprocessing_config = load_chunking_config_from_yaml(config_path)
    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preprocessing_config,
    )

    # Setup output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    debug_dir = None
    if draw_bboxes:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    # Find all audio files recursively
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))

    if not audio_files:
        console.print(f"[red]No audio files found in {input_dir}[/red]")
        return

    console.print(f"[green]Found {len(audio_files)} audio files[/green]")
    console.print(f"[blue]Config: {config_path}[/blue]")
    console.print(f"  target_size: {chunk_config.target_width}x{chunk_config.target_height}")
    console.print(f"  window_duration: {chunk_config.window_duration_ms:.0f}ms")
    console.print(f"  overlap_ratio: {chunk_config.overlap_ratio:.0%}")
    console.print(f"  min_chunk_content_ratio: {chunk_config.min_chunk_content_ratio:.0%}")

    # Build COCO dataset
    coco_data = {
        "info": {
            "description": "Audio spectrogram dataset",
            "version": "1.0",
            "year": 2026,
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_map = {}  # category_name -> category_id
    image_id = 1
    annotation_id = 1

    total_chunks = 0
    failed_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing audio files...", total=len(audio_files))

        for audio_path in audio_files:
            try:
                # Extract category from folder structure
                category_name = extract_category_from_path(audio_path, input_dir)

                # Register category if new
                if category_name not in category_map:
                    category_id = len(category_map)
                    category_map[category_name] = category_id
                    coco_data["categories"].append(
                        {
                            "id": category_id,
                            "name": category_name,
                            "supercategory": "audio_event",
                        }
                    )
                else:
                    category_id = category_map[category_name]

                # Load audio file
                from ezakodio.io import load_audio

                audio_tensor, sample_rate = load_audio(str(audio_path), mono=True, device="cpu")
                audio = audio_tensor.cpu().numpy().flatten()

                # Apply preprocessing to FULL audio file (not per-chunk)
                if preprocessing_config is not None:
                    from rf_detr_finetuning.audio_preprocessing import preprocess_audio

                    audio, preprocess_metadata = preprocess_audio(audio, sample_rate, preprocessing_config)
                    logger.debug(
                        f"Preprocessed {audio_path.name}: "
                        f"gain={preprocess_metadata.get('gain_applied_db', 0):.2f}dB, "
                        f"RMS {preprocess_metadata['original_rms_db']:.2f}→{preprocess_metadata['final_rms_db']:.2f}dB"
                    )
                duration_ms = (len(audio) / sample_rate) * 1000
                fmax = sample_rate / 2

                # Try to load JSON metadata for frequency bounds
                # Look for matching .json file (same stem as audio file)
                json_path = audio_path.with_suffix(".json")
                hz_min = 0.0
                hz_max = fmax

                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            meta = json.load(f)
                        # Extract frequency bounds from metadata
                        hz_min = float(meta.get("hz_min", 0.0))
                        hz_max = float(meta.get("hz_max", fmax))
                        # Override category from label_hierarchy if available
                        hierarchy = meta.get("label_hierarchy", "")
                        if hierarchy:
                            if " > " in hierarchy:
                                category_name = hierarchy.split(" > ")[-1].strip()
                            else:
                                category_name = hierarchy
                        # Ensure category is registered
                        if category_name not in category_map:
                            category_id = len(category_map)
                            category_map[category_name] = category_id
                            coco_data["categories"].append(
                                {
                                    "id": category_id,
                                    "name": category_name,
                                    "supercategory": "audio_event",
                                }
                            )
                        else:
                            category_id = category_map[category_name]
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to load metadata {json_path}: {e}")

                events = [
                    {
                        "time_start_ms": 0,
                        "time_end_ms": duration_ms,
                        "hz_min": hz_min,
                        "hz_max": hz_max,
                        "category": category_name,
                        "category_id": category_id,
                        "is_file_level": True,
                    }
                ]

                # Chunk audio
                chunks = chunker.chunk_audio(
                    audio,
                    sample_rate,
                    events=events,
                    source_uuid=audio_path.stem,
                )

                # Save chunks
                for chunk in chunks:
                    chunk_id = chunk.get_chunk_id()
                    img_filename = f"{chunk_id}.png"
                    img_path = images_dir / img_filename

                    # Save spectrogram
                    Image.fromarray(chunk.spectrogram).save(img_path)

                    # Add to COCO images
                    coco_data["images"].append(
                        {
                            "id": image_id,
                            "file_name": img_filename,
                            "width": chunk.spectrogram.shape[1],
                            "height": chunk.spectrogram.shape[0],
                            "audio_metadata": {
                                "source_file": audio_path.name,
                                "category": category_name,
                                "chunk_index": chunk.chunk_index,
                                "start_ms": chunk.start_ms,
                                "end_ms": chunk.end_ms,
                                "is_padded": chunk.is_padded,
                                "padding_ms": chunk.padding_amount_ms,
                            },
                        }
                    )

                    # Add bboxes to COCO annotations
                    for bbox in chunk.bboxes:
                        coco_data["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": bbox.category_id,
                                "bbox": bbox.to_coco_bbox(),
                                "area": bbox.width * bbox.height,
                                "iscrowd": 0,
                                "attributes": {
                                    "hz_min": bbox.hz_min,
                                    "hz_max": bbox.hz_max,
                                    "time_start_ms": bbox.original_time_start_ms,
                                    "time_end_ms": bbox.original_time_end_ms,
                                    "overlap_ratio": bbox.overlap_ratio,
                                },
                            }
                        )
                        annotation_id += 1

                    # Draw debug bboxes
                    if debug_dir and chunk.bboxes:
                        debug_path = debug_dir / f"{chunk_id}_debug.png"
                        debug_spec = chunk.spectrogram
                        if debug_colormap:
                            import matplotlib.pyplot as plt

                            cmap = plt.get_cmap(debug_colormap)
                            spec_norm = debug_spec.astype(np.float32)
                            spec_norm = (spec_norm - spec_norm.min()) / (spec_norm.max() - spec_norm.min() + 1e-8)
                            spec_rgba = cmap(spec_norm)
                            debug_spec = (spec_rgba[..., :3] * 255).astype(np.uint8)
                        draw_bboxes_on_spectrogram(debug_spec, chunk.bboxes, debug_path)

                    image_id += 1

                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")
                failed_files.append(audio_path.name)

            progress.update(task, advance=1)

    # Save COCO JSON
    coco_path = output_dir / "_annotations.coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    # Print summary
    console.print("\n[green bold]✓ Dataset Processing Complete[/green bold]")
    console.print(f"  Files processed: {len(audio_files) - len(failed_files)}/{len(audio_files)}")
    console.print(f"  Total chunks: {total_chunks}")
    console.print(f"  Categories: {len(category_map)}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  COCO annotations: {coco_path}")

    if debug_dir:
        console.print(f"  Debug images: {debug_dir}")

    if failed_files:
        console.print(f"\n[yellow]Failed files ({len(failed_files)}):[/yellow]")
        for fname in failed_files[:10]:
            console.print(f"  - {fname}")
        if len(failed_files) > 10:
            console.print(f"  ... and {len(failed_files) - 10} more")

    # Print category summary
    console.print("\n[cyan]Categories:[/cyan]")
    for cat_name, cat_id in sorted(category_map.items(), key=lambda x: x[1]):
        num_annotations = sum(1 for ann in coco_data["annotations"] if ann["category_id"] == cat_id)
        console.print(f"  [{cat_id}] {cat_name}: {num_annotations} annotations")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Chunk audio dataset organized by class folders into COCO format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory with class folders (e.g., biological-cetacean-odontoceti/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for chunked dataset",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/audio_chunking.yaml"),
        help="Path to chunking config YAML (default: config/audio_chunking.yaml)",
    )
    parser.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Generate debug images with bboxes drawn",
    )
    parser.add_argument(
        "--debug-colormap",
        type=str,
        default=None,
        help="Optional colormap for debug images (e.g., magma, viridis, inferno, plasma)",
    )
    parser.add_argument(
        "--audio-extensions",
        nargs="+",
        default=[".flac", ".wav", ".mp3"],
        help="Audio file extensions to process (default: .flac .wav .mp3)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return

    chunk_audio_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        draw_bboxes=args.draw_bboxes,
        debug_colormap=args.debug_colormap,
        audio_extensions=tuple(args.audio_extensions),
    )


if __name__ == "__main__":
    main()
