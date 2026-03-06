"""COCO dataset export functionality using the new AudioChunker.

This module provides functions for exporting audio datasets to COCO format for object detection training, using the new
modular AudioChunker.

"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TimeRemainingColumn

from rf_detr_finetuning.dataprocessor import AudioChunker

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CategoryRegistry:
    """Registry for managing category mappings.

    Supports frequency-aware category generation where the same annotation can map to different categories based on
    frequency range.

    """

    categories: dict[str, int] = field(default_factory=dict)
    frequency_bins: list[tuple[float, float, str]] | None = None

    def __post_init__(self):
        """Post-initialization hook for dataclass."""
        self._next_id = 0

    def get_or_create(self, name: str) -> int:
        """Get category ID, creating if necessary."""
        if name not in self.categories:
            self.categories[name] = self._next_id
            self._next_id += 1
        return self.categories[name]

    def get_category_with_frequency(
        self,
        base_name: str,
        hz_min: float,
        hz_max: float,
    ) -> tuple[int, str]:
        """Get category considering frequency range.

        If frequency_bins is set, creates compound categories like:
        - "ship_noise_low_freq" for events below 500 Hz
        - "sonar_high_freq" for events above 5000 Hz

        Args:
            base_name: Base annotation name.
            hz_min: Minimum frequency of the event.
            hz_max: Maximum frequency of the event.

        Returns:
            Tuple of (category_id, category_name).

        """
        if self.frequency_bins is None:
            cat_id = self.get_or_create(base_name)
            return cat_id, base_name

        # Find matching frequency bin
        center_freq = (hz_min + hz_max) / 2
        suffix = ""

        for bin_min, bin_max, bin_name in self.frequency_bins:
            if bin_min <= center_freq < bin_max:
                suffix = f"_{bin_name}"
                break

        full_name = f"{base_name}{suffix}"
        cat_id = self.get_or_create(full_name)
        return cat_id, full_name

    def to_coco_categories(self) -> list[dict[str, Any]]:
        """Export categories in COCO format."""
        return [
            {"id": cat_id, "name": name, "supercategory": "audio_event"}
            for name, cat_id in sorted(self.categories.items(), key=lambda x: x[1])
        ]


@dataclass
class COCODatasetBuilder:
    """Builder for COCO-format dataset."""

    info: dict[str, Any] = field(
        default_factory=lambda: {
            "description": "Audio spectrogram dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat(),
        }
    )
    licenses: list[dict[str, Any]] = field(default_factory=lambda: [{"id": 1, "name": "Unknown", "url": ""}])
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)

    _image_id: int = field(default=1, init=False)
    _annotation_id: int = field(default=1, init=False)

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Add an image to the dataset.

        Returns:
            Image ID.

        """
        image_entry = {
            "id": self._image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            image_entry.update(extra)

        self.images.append(image_entry)
        image_id = self._image_id
        self._image_id += 1
        return image_id

    def add_annotation(
        self,
        bbox: list[float],
        image_id: int,
        category_id: int,
        area: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Add an annotation to the dataset.

        Args:
            bbox: COCO format bbox [x, y, width, height].
            image_id: ID of the image this annotation belongs to.
            category_id: Category ID.
            area: Optional area of the bbox. Calculated if not provided.
            extra: Optional extra metadata.

        Returns:
            Annotation ID.

        """
        if area is None:
            area = bbox[2] * bbox[3]  # width * height

        annotation = {
            "id": self._annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }

        if extra:
            annotation.update(extra)

        self.annotations.append(annotation)
        annotation_id = self._annotation_id
        self._annotation_id += 1
        return annotation_id

    def set_categories(self, category_registry: CategoryRegistry) -> None:
        """Set categories from registry."""
        self.categories = category_registry.to_coco_categories()

    def to_dict(self) -> dict[str, Any]:
        """Export dataset to dictionary."""
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, output_path: Path | str) -> None:
        """Save dataset to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved COCO dataset to {output_path}")


def parse_frequency_bins(bins_str: str | None) -> list[tuple[float, float, str]] | None:
    """Parse frequency bins from string.

    Format: "min1-max1:name1,min2-max2:name2,..."
    Example: "0-500:low,500-2000:mid,2000-8000:high"

    Args:
        bins_str: Frequency bins string or None.

    Returns:
        List of (min_freq, max_freq, name) tuples or None.

    """
    if not bins_str:
        return None

    bins = []
    for bin_spec in bins_str.split(","):
        range_part, name = bin_spec.split(":")
        min_freq, max_freq = map(float, range_part.split("-"))
        bins.append((min_freq, max_freq, name))

    return bins


def validate_coco_dataset(coco_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate COCO dataset format.

    Args:
        coco_data: COCO dataset dictionary.

    Returns:
        Tuple of (is_valid, list_of_errors).

    """
    errors = []

    # Check required keys
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            errors.append(f"Missing required key: {key}")

    if errors:
        return False, errors

    # Validate images
    image_ids = set()
    for img in coco_data["images"]:
        if "id" not in img:
            errors.append("Image missing 'id' field")
        else:
            image_ids.add(img["id"])

    # Validate annotations
    for ann in coco_data["annotations"]:
        if "image_id" not in ann:
            errors.append("Annotation missing 'image_id' field")
        elif ann["image_id"] not in image_ids:
            errors.append(f"Annotation references non-existent image: {ann['image_id']}")

        if "bbox" in ann:
            bbox = ann["bbox"]
            if len(bbox) != 4:
                errors.append(f"Invalid bbox format: {bbox}")

    return len(errors) == 0, errors


def iterate_audio_dataset(
    input_dir: Path | str,
    audio_extensions: tuple[str, ...] = (".flac", ".wav", ".mp3", ".ogg"),
) -> Iterator[tuple[Path, Path]]:
    """Iterate over audio files with matching JSON metadata.

    Args:
        input_dir: Directory containing audio and JSON files.
        audio_extensions: Audio file extensions to look for.

    Yields:
        Tuples of (audio_path, json_path).

    """
    input_dir = Path(input_dir)

    for ext in audio_extensions:
        for audio_path in input_dir.glob(f"*{ext}"):
            json_path = audio_path.with_suffix(".json")
            if json_path.exists():
                yield audio_path, json_path
            else:
                logger.warning(f"No JSON metadata found for {audio_path}")


def convert_audio_to_coco(
    input_dir: str | Path,
    output_dir: str | Path,
    chunker: AudioChunker | None = None,
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    frequency_bins: list[tuple[float, float, str]] | None = None,
    audio_extensions: tuple[str, ...] = (".flac", ".wav", ".mp3", ".ogg"),
    random_seed: int | None = 42,
) -> Path:
    """Convert an audio dataset to COCO format using the new AudioChunker.

    This function processes audio files, chunks them into spectrograms,
    and exports them in COCO format for object detection training.

    Args:
        input_dir: Input directory containing audio files and JSON metadata.
        output_dir: Output directory for the COCO dataset.
        chunker: AudioChunker instance. If None, uses default configuration.
        split_ratios: Train/valid/test split ratios.
        frequency_bins: Optional frequency bins for frequency-aware categories.
            Example: [(0, 500, "low"), (500, 2000, "mid"), (2000, 8000, "high")]
        audio_extensions: Audio file extensions to process.
        random_seed: Random seed for reproducible splits.

    Returns:
        Path to output directory.

    Example:
        >>> chunker = AudioChunker(
        ...     fft_config=TimeBasedFFTConfig(hop_ms=10.0, n_mels=128),
        ...     chunk_config=ChunkConfig(target_size=640, overlap_ms=1280.0),
        ... )
        >>> convert_audio_to_coco(
        ...     input_dir="data/audio",
        ...     output_dir="data/coco",
        ...     chunker=chunker,
        ... )

    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Use default chunker if not provided
    if chunker is None:
        chunker = AudioChunker()

    # Collect all audio files
    audio_files = list(iterate_audio_dataset(input_dir, audio_extensions))

    if not audio_files:
        raise ValueError(f"No audio files with JSON metadata found in {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files to process")

    # Initialize category registry
    category_registry = CategoryRegistry(frequency_bins=frequency_bins)

    # Shuffle for random split
    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(audio_files)
    else:
        random.shuffle(audio_files)

    # Split into train/valid/test
    n_total = len(audio_files)
    n_train = int(n_total * split_ratios[0])
    n_valid = int(n_total * split_ratios[1])

    splits = {
        "train": audio_files[:n_train],
        "valid": audio_files[n_train : n_train + n_valid],
        "test": audio_files[n_train + n_valid :],
    }

    # Stats tracking
    stats = {"processed": 0, "failed": 0, "chunks": 0, "annotations": 0}

    # Process each split
    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        for split_name, split_files in splits.items():
            if not split_files:
                continue

            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            coco_builder = COCODatasetBuilder()
            task = progress.add_task(f"[cyan]{split_name}", total=len(split_files))

            for audio_path, json_path in split_files:
                try:
                    # Load metadata
                    with open(json_path, encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Extract events from metadata
                    events = metadata.get("events", [])
                    if not events and "annotation" in metadata:
                        # Single event from old-style metadata
                        events = [
                            {
                                "time_start_ms": metadata.get("time_start_ms", 0),
                                "time_end_ms": metadata.get("time_end_ms", metadata.get("duration", 0)),
                                "hz_min": metadata.get("hz_min", 0),
                                "hz_max": metadata.get("hz_max", 22050),
                                "category": metadata.get("annotation", "unknown"),
                                "category_id": 0,
                            }
                        ]

                    # Build frequency-aware category mapping from original events.
                    # Maps (original_category_id) -> frequency-aware category_id so we can
                    # remap the bboxes that come back from the chunker.
                    event_category_map: dict[int, int] = {}
                    for event in events:
                        label = event.get("label_hierarchy", "").split(" + ")[-1] or event.get("category", "unknown")
                        hz_min = event.get("hz_min", 0)
                        hz_max = event.get("hz_max", 22050)
                        orig_cat_id = event.get("category_id", 0)

                        cat_id, _cat_name = category_registry.get_category_with_frequency(label, hz_min, hz_max)
                        event_category_map[orig_cat_id] = cat_id

                    # Chunk audio
                    chunks = chunker.chunk_audio_file(audio_path, json_path)

                    if not chunks:
                        logger.warning(f"No chunks generated for {audio_path}")
                        stats["failed"] += 1
                        progress.advance(task)
                        continue

                    # Save chunks and create COCO entries
                    for chunk in chunks:
                        if chunk.spectrogram is None:
                            continue

                        # Save spectrogram image
                        chunk_id = chunk.get_chunk_id()
                        img_filename = f"{chunk_id}.png"
                        img_path = split_dir / img_filename

                        # Convert to PIL Image and save (always RGB for consistency)
                        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb

                        if chunk.spectrogram.ndim == 2:
                            rgb = grayscale_to_rgb(chunk.spectrogram)
                            img = Image.fromarray(rgb, mode="RGB")
                        else:
                            # Already 3-channel (RGB) spectrogram
                            img = Image.fromarray(chunk.spectrogram, mode="RGB")

                        img.save(img_path)

                        # Add image to COCO
                        height, width = chunk.spectrogram.shape[:2]
                        extra = {
                            "source_uuid": chunk.source_uuid,
                            "chunk_index": chunk.chunk_index,
                            "start_ms": chunk.start_ms,
                            "end_ms": chunk.end_ms,
                        }
                        image_id = coco_builder.add_image(img_filename, width, height, extra)

                        # Add annotations
                        for bbox in chunk.bboxes:
                            coco_bbox = bbox.to_coco_bbox()
                            area = bbox.width * bbox.height

                            # Remap category_id through frequency-aware mapping
                            remapped_cat_id = event_category_map.get(bbox.category_id, bbox.category_id)

                            ann_extra = {
                                "original_time_start_ms": bbox.original_time_start_ms,
                                "original_time_end_ms": bbox.original_time_end_ms,
                                "hz_min": bbox.hz_min,
                                "hz_max": bbox.hz_max,
                                "overlap_ratio": bbox.overlap_ratio,
                            }

                            coco_builder.add_annotation(coco_bbox, image_id, remapped_cat_id, area, ann_extra)
                            stats["annotations"] += 1

                        stats["chunks"] += 1

                    stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process {audio_path}: {e}")
                    stats["failed"] += 1

                progress.advance(task)

            # Set categories and save COCO JSON
            coco_builder.set_categories(category_registry)
            coco_path = split_dir / "_annotations.coco.json"
            coco_builder.save(coco_path)

    # Print summary
    logger.info("COCO Export Summary")
    logger.info(f"Processed: {stats['processed']} files")
    logger.info(f"Failed: {stats['failed']} files")
    logger.info(f"Total chunks: {stats['chunks']}")
    logger.info(f"Total annotations: {stats['annotations']}")
    logger.info(f"Categories: {len(category_registry.categories)}")
    logger.info(f"Output directory: {output_dir}")

    return output_dir
