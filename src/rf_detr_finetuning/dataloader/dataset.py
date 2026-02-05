"""Dataset classes for audio chunk detection.

Provides PyTorch-compatible datasets for:
- Pre-chunked spectrogram images (COCOAudioDataset)
- On-the-fly chunking from raw audio (AudioChunkDataset)
- In-memory datasets for small data (InMemoryChunkDataset)

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DetectionTarget:
    """Detection target for a single image/chunk.

    Attributes:
        boxes: Bounding boxes in XYXY format, shape (N, 4).
        labels: Class labels, shape (N,).
        image_id: Unique identifier for the image.
        area: Area of each box, shape (N,).
        iscrowd: Crowd flag for each box, shape (N,).

    """

    boxes: torch.Tensor
    labels: torch.Tensor
    image_id: int
    area: torch.Tensor | None = None
    iscrowd: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Compute derived fields if not provided."""
        if self.area is None and len(self.boxes) > 0:
            # Compute area from XYXY boxes
            widths = self.boxes[:, 2] - self.boxes[:, 0]
            heights = self.boxes[:, 3] - self.boxes[:, 1]
            self.area = widths * heights
        elif self.area is None:
            self.area = torch.tensor([], dtype=torch.float32)

        if self.iscrowd is None:
            self.iscrowd = torch.zeros(len(self.labels), dtype=torch.int64)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by DETR-style models."""
        return {
            "boxes": self.boxes,
            "labels": self.labels,
            "image_id": torch.tensor([self.image_id]),
            "area": self.area,
            "iscrowd": self.iscrowd,
        }


class COCOAudioDataset(Dataset):
    """Dataset for pre-chunked spectrogram images in COCO format.

    Loads images and annotations from a COCO-format JSON file.
    Expects images to be spectrogram PNGs with associated bounding boxes.

    Args:
        annotation_file: Path to COCO annotations JSON.
        image_dir: Directory containing spectrogram images.
        transform: Optional transform to apply to images.
        target_transform: Optional transform to apply to targets.

    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path | None = None,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        """Initialize COCO detection dataset.

        Args:
            annotation_file: Path to COCO annotations JSON.
            image_dir: Directory containing images. If None, inferred from annotations.
            transform: Optional image transform.
            target_transform: Optional target transform.

        """
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

        # Load COCO annotations
        with open(self.annotation_file) as f:
            self.coco_data = json.load(f)

        # Determine image directory
        if image_dir is not None:
            self.image_dir = Path(image_dir)
        else:
            # Default to same directory as annotations or 'images' subfolder
            parent = self.annotation_file.parent
            if (parent / "images").exists():
                self.image_dir = parent / "images"
            else:
                self.image_dir = parent

        # Build indices
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = list(self.images.keys())

        # Build annotation index (image_id -> list of annotations)
        self.img_to_anns: dict[int, list[dict]] = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self.img_to_anns:
                self.img_to_anns[img_id].append(ann)

        # Build category mapping
        self.categories = {cat["id"]: cat for cat in self.coco_data.get("categories", [])}
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(self.categories.keys()))}

        logger.info(
            f"Loaded {len(self.images)} images with {sum(len(anns) for anns in self.img_to_anns.values())} annotations"
        )

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, DetectionTarget]:
        """Get image and detection target.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (image_tensor, detection_target).

        """
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        # Load image
        img_path = self.image_dir / img_info["file_name"]
        image = self._load_image(img_path)

        # Get annotations for this image
        anns = self.img_to_anns[image_id]

        # Convert annotations to boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # COCO format: [x, y, width, height] -> XYXY
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            # Map category ID to contiguous index
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        target = DetectionTarget(
            boxes=boxes_tensor,
            labels=labels_tensor,
            image_id=image_id,
        )

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image as tensor.

        Args:
            path: Path to image file.

        Returns:
            Image tensor of shape (C, H, W).

        """
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    @property
    def num_classes(self) -> int:
        """Number of object classes (excluding background)."""
        return len(self.categories)

    def get_category_names(self) -> list[str]:
        """Get list of category names in order of index."""
        return [self.categories[cat_id]["name"] for cat_id in sorted(self.categories.keys())]


class AudioChunkDataset(Dataset):
    """Dataset that chunks audio files on-the-fly.

    Loads raw audio files and generates spectrogram chunks during iteration.
    Useful for training when you want different random chunking each epoch.

    Args:
        audio_files: List of paths to audio files.
        metadata_files: List of paths to corresponding metadata JSON files.
        chunker: AudioChunker instance for processing.
        transform: Optional transform to apply to chunk images.

    """

    def __init__(
        self,
        audio_files: list[str | Path],
        metadata_files: list[str | Path],
        chunker: Any,  # AudioChunker
        transform: Any | None = None,
    ) -> None:
        """Initialize audio chunk dataset.

        Args:
            audio_files: List of audio file paths.
            metadata_files: List of metadata file paths.
            chunker: AudioChunker instance for processing.
            transform: Optional transform.

        """
        self.audio_files = [Path(f) for f in audio_files]
        self.metadata_files = [Path(f) for f in metadata_files]
        self.chunker = chunker
        self.transform = transform

        if len(self.audio_files) != len(self.metadata_files):
            raise ValueError(
                f"Number of audio files ({len(self.audio_files)}) must match "
                f"number of metadata files ({len(self.metadata_files)})"
            )

        # Pre-compute total chunks per file for indexing
        self._file_chunks: list[int] = []
        self._cumulative_chunks: list[int] = [0]
        self._precompute_chunk_counts()

    def _precompute_chunk_counts(self) -> None:
        """Estimate chunk counts for each audio file."""
        for audio_path in self.audio_files:
            # Estimate based on file duration and chunk window
            # This is approximate; actual count may differ
            try:
                from rf_detr_finetuning.dataprocessor import load_audio_file

                audio, sr = load_audio_file(str(audio_path))
                duration_ms = len(audio) / sr * 1000
                window_ms = self.chunker.chunk_config.window_duration_ms
                overlap_ms = self.chunker.chunk_config.overlap_ms
                step_ms = window_ms - overlap_ms

                if duration_ms <= window_ms:
                    n_chunks = 1
                else:
                    n_chunks = int(np.ceil((duration_ms - window_ms) / step_ms)) + 1

                self._file_chunks.append(n_chunks)
                self._cumulative_chunks.append(self._cumulative_chunks[-1] + n_chunks)
            except Exception as e:
                logger.warning(f"Could not estimate chunks for {audio_path}: {e}")
                self._file_chunks.append(1)
                self._cumulative_chunks.append(self._cumulative_chunks[-1] + 1)

    def __len__(self) -> int:
        """Return total number of chunks across all audio files."""
        return self._cumulative_chunks[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, DetectionTarget]:
        """Get chunk and detection target.

        Args:
            idx: Global chunk index.

        Returns:
            Tuple of (chunk_tensor, detection_target).

        """
        # Find which file this index belongs to
        file_idx = 0
        for i, cum in enumerate(self._cumulative_chunks[1:], 1):
            if idx < cum:
                file_idx = i - 1
                break

        chunk_idx = idx - self._cumulative_chunks[file_idx]

        # Load and process the file
        audio_path = self.audio_files[file_idx]
        metadata_path = self.metadata_files[file_idx]

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        events = metadata.get("events", [])

        # Process with chunker
        chunks = self.chunker.process_file(str(audio_path), events)

        # Handle case where actual chunks differ from estimate
        if chunk_idx >= len(chunks):
            chunk_idx = len(chunks) - 1

        chunk = chunks[chunk_idx]

        # Convert spectrogram to tensor
        image = self._spectrogram_to_tensor(chunk.spectrogram)

        # Convert bboxes to target
        boxes = []
        labels = []
        for bbox in chunk.bboxes:
            # COCO format: [x, y, width, height] -> XYXY
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            boxes.append([x, y, x + w, y + h])
            labels.append(bbox.class_id)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        target = DetectionTarget(
            boxes=boxes_tensor,
            labels=labels_tensor,
            image_id=idx,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def _spectrogram_to_tensor(self, spectrogram: np.ndarray) -> torch.Tensor:
        """Convert spectrogram array to tensor.

        Args:
            spectrogram: 2D spectrogram array.

        Returns:
            Tensor of shape (3, H, W) for RGB.

        """
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb, normalize_to_range

        # Normalize to 0-255 range
        normalized = normalize_to_range(spectrogram, 0, 255)

        # Convert to RGB
        rgb = grayscale_to_rgb(normalized)

        # Scale to 0-1 and convert to tensor
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)

        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor


@dataclass
class InMemoryChunkDataset(Dataset):
    """In-memory dataset for pre-computed chunks.

    Stores all chunks in memory for fast access during training.
    Best for smaller datasets that fit in RAM.

    Args:
        chunks: List of (spectrogram, bboxes, metadata) tuples.
        class_names: List of class names for label mapping.
        transform: Optional transform to apply.

    """

    chunks: list[tuple[np.ndarray, list[Any], dict]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    transform: Any | None = None

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, DetectionTarget]:
        """Get chunk and target."""
        spectrogram, bboxes, metadata = self.chunks[idx]

        # Convert spectrogram to tensor
        from rf_detr_finetuning.dataprocessor import grayscale_to_rgb, normalize_to_range

        normalized = normalize_to_range(spectrogram, 0, 255)
        rgb = grayscale_to_rgb(normalized)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Convert bboxes
        boxes = []
        labels = []
        for bbox in bboxes:
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            boxes.append([x, y, x + w, y + h])
            labels.append(bbox.class_id)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        target = DetectionTarget(
            boxes=boxes_tensor,
            labels=labels_tensor,
            image_id=idx,
        )

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, target

    @classmethod
    def from_chunker(
        cls,
        audio_files: list[str | Path],
        metadata_files: list[str | Path],
        chunker: Any,
        class_names: list[str] | None = None,
        transform: Any | None = None,
    ) -> InMemoryChunkDataset:
        """Create dataset by processing audio files with a chunker.

        Args:
            audio_files: List of audio file paths.
            metadata_files: List of corresponding metadata paths.
            chunker: AudioChunker instance.
            class_names: Optional list of class names.
            transform: Optional transform.

        Returns:
            Populated InMemoryChunkDataset.

        """
        chunks = []

        for audio_path, meta_path in zip(audio_files, metadata_files):
            with open(meta_path) as f:
                metadata = json.load(f)

            events = metadata.get("events", [])
            file_chunks = chunker.process_file(str(audio_path), events)

            for chunk in file_chunks:
                chunk_meta = {
                    "source_file": str(audio_path),
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "is_padded": chunk.is_padded,
                }
                chunks.append((chunk.spectrogram, chunk.bboxes, chunk_meta))

        return cls(
            chunks=chunks,
            class_names=class_names or [],
            transform=transform,
        )
