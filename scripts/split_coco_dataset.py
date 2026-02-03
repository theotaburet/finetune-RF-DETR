#!/usr/bin/env python3
"""Split COCO dataset into train/val/test splits.

This script takes a single COCO format dataset and splits it into
train/validation/test sets while maintaining the COCO format structure.

Usage:
    python scripts/split_coco_dataset.py \
        --input-coco data/chunked_coco/_annotations.coco.json \
        --input-images data/chunked_coco/images \
        --output-dir data/chunked_coco_split \
        --train-ratio 0.7 \
        --val-ratio 0.2 \
        --test-ratio 0.1

"""

import argparse
import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_coco(
    input_coco_path: str,
    input_images_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> None:
    """Split COCO dataset into train/val/test.

    Args:
        input_coco_path: Path to input COCO JSON file.
        input_images_dir: Directory containing input images.
        output_dir: Output directory for split datasets.
        train_ratio: Proportion for training set (default: 0.7).
        val_ratio: Proportion for validation set (default: 0.2).
        test_ratio: Proportion for test set (default: 0.1).
        random_seed: Random seed for reproducibility (default: 42).

    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Load COCO
    print(f"Loading COCO from: {input_coco_path}")
    with open(input_coco_path) as f:
        coco = json.load(f)

    print(f"Total images: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    # Split images
    images = coco["images"]
    train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=random_seed)
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed,
    )

    # Process each split
    for split_name, split_imgs in [
        ("train", train_imgs),
        ("valid", val_imgs),
        ("test", test_imgs),
    ]:
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split_name} split...")

        # Copy images
        for img in split_imgs:
            src = Path(input_images_dir) / img["file_name"]
            if not src.exists():
                # Try without images/ prefix
                src = Path(input_images_dir).parent / img["file_name"]
            if not src.exists():
                print(f"Warning: Image not found: {src}")
                continue

            dst = split_dir / Path(img["file_name"]).name
            shutil.copy(src, dst)

            # Update file_name to be just the basename
            img["file_name"] = Path(img["file_name"]).name

        # Filter annotations
        image_ids = {img["id"] for img in split_imgs}
        split_annos = [a for a in coco["annotations"] if a["image_id"] in image_ids]

        # Save split COCO
        split_coco = {
            "images": split_imgs,
            "annotations": split_annos,
            "categories": coco["categories"],
        }
        coco_path = split_dir / "_annotations.coco.json"
        with open(coco_path, "w") as f:
            json.dump(split_coco, f, indent=2)

        print(f"  ✓ {split_name}: {len(split_imgs)} images, {len(split_annos)} annotations")
        print(f"  ✓ Saved to: {coco_path}")

    print(f"\n✓ Dataset split complete: {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val/test")
    parser.add_argument(
        "--input-coco",
        required=True,
        help="Path to input COCO JSON file",
    )
    parser.add_argument(
        "--input-images",
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    split_coco(
        input_coco_path=args.input_coco,
        input_images_dir=args.input_images,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
