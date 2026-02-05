#!/usr/bin/env python3
"""Run inference on test images and save predictions with evaluation metrics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table

from rf_detr_finetuning.predict import prediction

console = Console()


def load_coco_annotations(annotations_path: Path) -> dict:
    """Load ground truth annotations from COCO JSON file."""
    with open(annotations_path) as f:
        return json.load(f)


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate Intersection over Union between two boxes.

    Args:
        box1: [x, y, width, height] in COCO format.
        box2: [x, y, width, height] in COCO format.

    Returns:
        IoU score between 0 and 1.

    """
    # Convert to [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_predictions(
    predictions: dict,
    ground_truth: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """Calculate precision, recall, and mAP metrics.

    Args:
        predictions: Dictionary of predictions per image.
        ground_truth: COCO annotations dict.
        iou_threshold: IoU threshold for matching predictions to GT.

    Returns:
        Dictionary with evaluation metrics.

    """
    # Build GT lookup by image filename
    gt_by_image = {}
    image_id_to_filename = {img["id"]: img["file_name"] for img in ground_truth["images"]}
    categories = {cat["id"]: cat["name"] for cat in ground_truth["categories"]}

    for ann in ground_truth["annotations"]:
        img_id = ann["image_id"]
        filename = image_id_to_filename[img_id]
        if filename not in gt_by_image:
            gt_by_image[filename] = []
        gt_by_image[filename].append({"bbox": ann["bbox"], "category_id": ann["category_id"], "matched": False})

    # Calculate metrics per class
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0})

    for pred in predictions:
        img_name = pred["image"]
        gt_annotations = gt_by_image.get(img_name, [])

        # Track which GT boxes have been matched
        for gt_ann in gt_annotations:
            class_stats[gt_ann["category_id"]]["total_gt"] += 1

        # Match predictions to ground truth
        for pred_box in pred["predictions"]:
            pred_class = pred_box["class_id"]
            pred_bbox = pred_box["bbox"]  # [x1, y1, x2, y2] from xyxy
            # Convert xyxy to xywh for IoU calculation
            pred_bbox_xywh = [
                pred_bbox[0],
                pred_bbox[1],
                pred_bbox[2] - pred_bbox[0],
                pred_bbox[3] - pred_bbox[1],
            ]

            best_iou = 0.0
            best_match_idx = -1

            # Find best matching GT box
            for idx, gt_ann in enumerate(gt_annotations):
                if gt_ann["category_id"] == pred_class and not gt_ann["matched"]:
                    iou = calculate_iou(pred_bbox_xywh, gt_ann["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx

            # Check if prediction matches GT
            if best_iou >= iou_threshold and best_match_idx >= 0:
                class_stats[pred_class]["tp"] += 1
                gt_annotations[best_match_idx]["matched"] = True
            else:
                class_stats[pred_class]["fp"] += 1

        # Count false negatives (unmatched GT)
        for gt_ann in gt_annotations:
            if not gt_ann["matched"]:
                class_stats[gt_ann["category_id"]]["fn"] += 1

    # Calculate precision, recall, F1 per class
    metrics = {}
    for class_id, stats in class_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_name = categories.get(class_id, f"class_{class_id}")
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_gt": stats["total_gt"],
        }

    # Calculate overall metrics
    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    return metrics


def main() -> None:
    """Run inference on test set and evaluate."""
    parser = argparse.ArgumentParser(description="Run inference on test set with evaluation")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("output/checkpoint_best_ema.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("split_dataset/test"),
        help="Directory containing test images and annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/test_predictions"),
        help="Directory to save prediction visualizations and results",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["nano", "small", "base", "medium", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save annotated images with predictions",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    annotations_path = args.test_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        console.print(f"[red]Error: Annotations not found at {annotations_path}[/red]")
        return

    console.print(f"[cyan]Loading annotations from {annotations_path}...[/cyan]")
    ground_truth = load_coco_annotations(annotations_path)

    # Build class names mapping
    class_names = {cat["id"]: cat["name"] for cat in ground_truth["categories"]}
    console.print(f"[green]Classes: {class_names}[/green]")

    # Get test images
    test_images = sorted(args.test_dir.glob("*.png"))
    if args.max_images:
        test_images = test_images[: args.max_images]

    console.print(f"[green]Found {len(test_images)} test images[/green]")
    console.print(f"[cyan]Model: {args.model_path}[/cyan]")
    console.print(f"[cyan]Confidence threshold: {args.confidence}[/cyan]")
    console.print(f"[cyan]IoU threshold: {args.iou_threshold}[/cyan]")

    # Run predictions
    results = []
    for img_path in track(test_images, description="Running inference..."):
        # Get annotated image
        annotated_image = prediction(
            image_path=str(img_path),
            model_size=args.model_size,
            model_path=str(args.model_path),
            confidence=args.confidence,
            class_names=class_names,
        )

        # Save visualization if requested
        if args.save_visualizations:
            output_path = args.output_dir / f"{img_path.stem}_pred.png"
            # Convert BGR to RGB for saving
            Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

        # Re-run prediction to get raw detections for stats
        # (We need the actual detection objects, not just the visualization)
        import warnings

        import torch

        from rf_detr_finetuning.finetune import MAP_MODEL_SIZE

        ModelClass = MAP_MODEL_SIZE[args.model_size.lower()]

        # Load image
        image = plt.imread(str(img_path))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        # Load model (cache this outside loop for speed)
        if not hasattr(main, "_model_cache"):
            checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
            bias = checkpoint["model"].get("class_embed.bias")
            num_classes = bias.shape[0] - 1 if bias is not None else None

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if num_classes is not None:
                    model = ModelClass(pretrain_weights=str(args.model_path), num_classes=num_classes)
                else:
                    model = ModelClass(pretrain_weights=str(args.model_path))
            main._model_cache = model

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            detections = main._model_cache.predict(image, confidence=args.confidence)

        # Store results
        if detections is not None and len(detections) > 0:
            predictions_list = []
            for idx in range(len(detections)):
                predictions_list.append(
                    {
                        "class_id": int(detections.class_id[idx]),
                        "class_name": class_names.get(int(detections.class_id[idx]), "unknown"),
                        "confidence": float(detections.confidence[idx]),
                        "bbox": [float(x) for x in detections.xyxy[idx]],
                    }
                )

            results.append(
                {
                    "image": img_path.name,
                    "num_detections": len(detections),
                    "predictions": predictions_list,
                }
            )
        else:
            results.append({"image": img_path.name, "num_detections": 0, "predictions": []})

    # Save results summary
    summary_path = args.output_dir / "predictions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate predictions
    console.print("\n[bold cyan]Evaluating predictions...[/bold cyan]")
    metrics = evaluate_predictions(results, ground_truth, args.iou_threshold)

    # Save metrics
    metrics_path = args.output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print statistics table
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("GT Total", justify="right")

    for class_name, stats in sorted(metrics.items()):
        if class_name == "overall":
            continue
        table.add_row(
            class_name,
            f"{stats['precision']:.3f}",
            f"{stats['recall']:.3f}",
            f"{stats['f1']:.3f}",
            str(stats["tp"]),
            str(stats["fp"]),
            str(stats["fn"]),
            str(stats["total_gt"]),
        )

    # Add overall row
    overall = metrics["overall"]
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{overall['precision']:.3f}[/bold]",
        f"[bold]{overall['recall']:.3f}[/bold]",
        f"[bold]{overall['f1']:.3f}[/bold]",
        f"[bold]{overall['tp']}[/bold]",
        f"[bold]{overall['fp']}[/bold]",
        f"[bold]{overall['fn']}[/bold]",
        "-",
    )

    console.print(table)

    # Print summary
    total_detections = sum(r["num_detections"] for r in results)
    console.print(f"\n[green]✓ Predictions saved to {args.output_dir}[/green]")
    console.print(f"[cyan]Summary: {summary_path}[/cyan]")
    console.print(f"[cyan]Metrics: {metrics_path}[/cyan]")
    console.print("\n[bold]Processing Summary:[/bold]")
    console.print(f"  Images processed: {len(results)}")
    console.print(f"  Total detections: {total_detections}")
    console.print(f"  Avg detections/image: {total_detections / len(results):.1f}")


if __name__ == "__main__":
    main()
