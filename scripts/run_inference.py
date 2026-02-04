#!/usr/bin/env python3
"""Run inference on test images and save predictions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rfdetr import DETR
from rich.console import Console
from rich.progress import track

console = Console()

# Configuration
MODEL_PATH = "output/checkpoint_best_ema.pth"
TEST_DIR = Path("split_dataset/test/images")
OUTPUT_DIR = Path("output/predictions")
CONFIDENCE_THRESHOLD = 0.3
CLASS_NAMES = {0: "cargo", 1: "passenger ship", 2: "odontoceti"}

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
console.print(f"[cyan]Loading model from {MODEL_PATH}...[/cyan]")
model = DETR(model_size="small")
model.load(MODEL_PATH)

# Get test images
test_images = sorted(TEST_DIR.glob("*.png"))[:20]  # First 20 images
console.print(f"[green]Found {len(test_images)} test images[/green]")

# Run predictions
results = []
for img_path in track(test_images, description="Running inference..."):
    # Predict
    detections = model.predict(
        str(img_path),
        conf=CONFIDENCE_THRESHOLD,
        visualize=False,
    )

    # Save visualization
    if detections is not None and len(detections) > 0:
        # Load original image
        img = np.array(Image.open(img_path))

        # Draw predictions
        from rf_detr_finetuning.predict import prediction

        visual = prediction(
            image_path=str(img_path),
            model_size="small",
            model_path=MODEL_PATH,
            confidence=CONFIDENCE_THRESHOLD,
            class_names=CLASS_NAMES,
        )

        # Save
        output_path = OUTPUT_DIR / f"{img_path.stem}_pred.png"
        plt.imsave(str(output_path), visual)

        # Store results
        results.append(
            {
                "image": img_path.name,
                "num_detections": len(detections),
                "predictions": [
                    {
                        "class_id": int(det[5]),
                        "class_name": CLASS_NAMES.get(int(det[5]), "unknown"),
                        "confidence": float(det[4]),
                        "bbox": [float(x) for x in det[:4]],
                    }
                    for det in detections
                ],
            }
        )

# Save results summary
summary_path = OUTPUT_DIR / "predictions_summary.json"
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)

console.print(f"\n[green]✓ Predictions saved to {OUTPUT_DIR}[/green]")
console.print(f"[cyan]Summary: {summary_path}[/cyan]")

# Print stats
total_detections = sum(r["num_detections"] for r in results)
console.print("\n[bold]Statistics:[/bold]")
console.print(f"  Images processed: {len(results)}")
console.print(f"  Total detections: {total_detections}")
console.print(f"  Avg detections/image: {total_detections / len(results):.1f}")
