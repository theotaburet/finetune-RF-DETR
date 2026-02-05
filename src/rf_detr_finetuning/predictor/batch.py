"""Batch prediction utilities.

Provides efficient batch inference for directories and datasets.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from rf_detr_finetuning.predictor.inference import (
    PredictionResult,
    Predictor,
)

logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction handler for efficient inference.

    Args:
        predictor: Base predictor instance.
        batch_size: Images per batch.

    """

    def __init__(
        self,
        predictor: Predictor,
        batch_size: int = 8,
    ) -> None:
        """Initialize batch predictor.

        Args:
            predictor: Single-image predictor instance.
            batch_size: Number of images per batch.

        """
        self.predictor = predictor
        self.batch_size = batch_size

    def predict_images(
        self,
        images: list[np.ndarray | str | Path],
        confidence_threshold: float = 0.5,
        show_progress: bool = True,
    ) -> list[PredictionResult]:
        """Run prediction on list of images.

        Args:
            images: List of images (arrays or paths).
            confidence_threshold: Minimum confidence.
            show_progress: Show progress bar.

        Returns:
            List of prediction results.

        """
        results = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
            ) as progress:
                task = progress.add_task("Predicting...", total=len(images))

                for image in images:
                    result = self.predictor.predict(image, confidence_threshold)
                    results.append(result)
                    progress.advance(task)
        else:
            for image in images:
                result = self.predictor.predict(image, confidence_threshold)
                results.append(result)

        return results

    def predict_generator(
        self,
        images: Iterator[np.ndarray | str | Path],
        confidence_threshold: float = 0.5,
    ) -> Iterator[PredictionResult]:
        """Lazily predict on image generator.

        Args:
            images: Iterator of images.
            confidence_threshold: Minimum confidence.

        Yields:
            PredictionResult for each image.

        """
        for image in images:
            yield self.predictor.predict(image, confidence_threshold)


def predict_directory(
    predictor: Predictor,
    image_dir: str | Path,
    output_path: str | Path | None = None,
    confidence_threshold: float = 0.5,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    recursive: bool = False,
) -> list[PredictionResult]:
    """Run prediction on all images in a directory.

    Args:
        predictor: Predictor instance.
        image_dir: Directory containing images.
        output_path: Optional path to save results JSON.
        confidence_threshold: Minimum confidence.
        extensions: Valid image extensions.
        recursive: Search subdirectories.

    Returns:
        List of prediction results.

    """
    image_dir = Path(image_dir)

    # Find all images
    if recursive:
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(f"*{ext}"))
    else:
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)
    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    # Run predictions
    batch_predictor = BatchPredictor(predictor)
    results = batch_predictor.predict_images(
        image_paths,
        confidence_threshold=confidence_threshold,
    )

    # Add paths to results
    for result, path in zip(results, image_paths):
        result.image_path = path

    # Save results if requested
    if output_path:
        _save_results(results, output_path)

    return results


def predict_dataset(
    predictor: Predictor,
    annotation_file: str | Path,
    image_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    confidence_threshold: float = 0.5,
) -> list[PredictionResult]:
    """Run prediction on a COCO-format dataset.

    Args:
        predictor: Predictor instance.
        annotation_file: Path to COCO annotations JSON.
        image_dir: Directory containing images (inferred from annotations if None).
        output_path: Optional path to save results.
        confidence_threshold: Minimum confidence.

    Returns:
        List of prediction results.

    """
    annotation_file = Path(annotation_file)

    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Determine image directory
    if image_dir is None:
        parent = annotation_file.parent
        if (parent / "images").exists():
            image_dir = parent / "images"
        else:
            image_dir = parent
    else:
        image_dir = Path(image_dir)

    # Build image list
    images_info = coco_data.get("images", [])
    image_paths = [image_dir / img["file_name"] for img in images_info]
    image_ids = [img["id"] for img in images_info]

    logger.info(f"Running inference on {len(image_paths)} images from dataset")

    # Run predictions
    batch_predictor = BatchPredictor(predictor)
    results = batch_predictor.predict_images(
        image_paths,
        confidence_threshold=confidence_threshold,
    )

    # Add metadata
    for result, path, img_id in zip(results, image_paths, image_ids):
        result.image_path = path
        result.image_id = img_id

    # Save results
    if output_path:
        _save_results(results, output_path)
        _save_coco_results(results, output_path)

    return results


def _save_results(
    results: list[PredictionResult],
    output_path: str | Path,
) -> None:
    """Save results to JSON file.

    Args:
        results: List of prediction results.
        output_path: Output file path.

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for result in results:
        data.append(
            {
                "image_path": str(result.image_path) if result.image_path else None,
                "image_id": result.image_id,
                "detections": [d.to_dict() for d in result.detections],
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(results)} predictions to {output_path}")


def _save_coco_results(
    results: list[PredictionResult],
    output_path: str | Path,
) -> None:
    """Save results in COCO detection format for evaluation.

    Args:
        results: List of prediction results.
        output_path: Base output path (will append _coco.json).

    """
    output_path = Path(output_path)
    coco_path = output_path.with_name(output_path.stem + "_coco.json")

    coco_results = []
    for result in results:
        if result.image_id is None:
            continue

        for det in result.detections:
            coco_results.append(
                {
                    "image_id": result.image_id,
                    "category_id": det.class_id,
                    "bbox": [det.x1, det.y1, det.width, det.height],
                    "score": det.score,
                }
            )

    with open(coco_path, "w") as f:
        json.dump(coco_results, f, indent=2)

    logger.info(f"Saved COCO-format results to {coco_path}")
