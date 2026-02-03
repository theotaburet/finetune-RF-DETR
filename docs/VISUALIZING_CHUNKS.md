# Visualizing Chunked Spectrograms with Bboxes

Quick guide for testing and visualizing chunked audio spectrograms with bounding boxes.

## During Development: Run Tests with Visual Output

### View Chunked Spectrograms from Real Files

```bash
# Run the real files test with verbose output
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

# Output images are saved to: output/test_chunks_debug/
```

**Output:**

- Debug images with bboxes drawn: `output/test_chunks_debug/*_debug.png`
- Console shows chunk info: time ranges, bbox counts, padding status

### What You'll See

The test creates one PNG per chunk with:

- **Spectrogram**: Time (x-axis) vs Frequency (y-axis)
- **Green boxes**: Annotated events (drawn with supervision library)
- **Labels**: Event category names

**Console output:**

```
Chunk 00:      0-  5000ms | 2 bbox(es) | normal | file_chunk0000_debug.png
  └─ Bbox 0: [ 120,  45, 230,  89] | label=bird_call | overlap=0.85
  └─ Bbox 1: [ 380, 120, 180, 150] | label=engine_noise | overlap=0.62
```

## Production: Generate Debug Images from CLI

### Option A: Using --draw-bboxes Flag

```bash
# Chunk audio and generate debug images
uv run rf-detr convert chunk-audio \
    --input-dir ./data/my_audio_dataset \
    --output-dir ./data/chunked_output \
    --config-file ./config/audio_chunking.yaml \
    --draw-bboxes \
    --debug-output-dir ./data/chunked_output/debug_visualizations

# Debug images saved to: data/chunked_output/debug_visualizations/
```

### Option B: Using Python Script

```python
from pathlib import Path
from rf_detr_finetuning import (
    AudioChunker,
    ChunkConfig,
    TimeBasedFFTConfig,
    draw_bboxes_on_spectrogram,
)
from PIL import Image

# Configure chunker
chunker = AudioChunker(
    fft_config=TimeBasedFFTConfig(fft_ms=25, hop_ms=10, n_mels=128),
    chunk_config=ChunkConfig(
        window_duration_ms=5000,
        overlap_ms=1000,
        target_width=640,
        target_height=640,
    ),
)

# Process single file
audio_path = Path("data/my_audio_dataset/sample.flac")
json_path = Path("data/my_audio_dataset/sample.json")

chunks = chunker.chunk_audio_file(audio_path, json_path)

# Save debug images
output_dir = Path("output/debug_chunks")
output_dir.mkdir(parents=True, exist_ok=True)

for i, chunk in enumerate(chunks):
    # Draw bboxes on spectrogram
    debug_img = draw_bboxes_on_spectrogram(chunk.spectrogram, chunk.bboxes)

    # Save with descriptive filename
    filename = f"{audio_path.stem}_chunk{i:04d}_debug.png"
    Image.fromarray(debug_img).save(output_dir / filename)

    print(
        f"Chunk {i}: {chunk.start_ms:.0f}-{chunk.end_ms:.0f}ms, "
        f"{len(chunk.bboxes)} bboxes -> {filename}"
    )
```

## Interpreting Debug Images

### Understanding the Visualization

**Axes:**

- **X-axis (horizontal)**: Time in milliseconds (left = start, right = end)
- **Y-axis (vertical)**: Frequency in Hz (bottom = low freq, top = high freq)
- **Colors**: Brighter = more energy at that time/frequency

**Bounding Boxes:**

- **Green rectangles**: Annotated events from JSON file
- **Position**: Time span (width) × Frequency range (height)
- **Labels**: Category names (e.g., "bird_call", "engine_noise")

### Example Interpretation

```
Image: audio_001_chunk0002_debug.png
Time range: 8000-13000ms (5-second window)
```

If you see:

- **Wide box**: Long-duration event (e.g., 2-second bird song)
- **Tall box**: Broadband event (e.g., 100-5000 Hz engine rumble)
- **Small box**: Short, narrow-band event (e.g., 200ms chirp at 3-4kHz)
- **Box at edge**: Event may be split across chunks (overlap helps)

### Common Issues to Check

| Issue               | What to Look For                             | Fix                                                  |
| ------------------- | -------------------------------------------- | ---------------------------------------------------- |
| **Missing bboxes**  | Events too small or at chunk boundaries      | Lower `min_overlap_ratio` to 0.1                     |
| **Wrong positions** | Bboxes not aligned with spectrogram features | Verify `fft_ms`/`hop_ms` match your data             |
| **Too many chunks** | Hundreds of images for short audio           | Increase `window_duration_ms` or reduce `overlap_ms` |
| **Partial events**  | Boxes cut off at edges                       | Increase `overlap_ms` (e.g., 1000→2000ms)            |
| **Empty chunks**    | No bboxes but audio has content              | Check JSON file has correct time/freq ranges         |

## Quick Checklist Before Training

✅ **Visual inspection:**

1. Run test: `pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s`
2. Open `output/test_chunks_debug/*.png` in image viewer
3. Verify green boxes align with visible spectrogram features
4. Check time ranges match expected event durations

✅ **Statistics check:**

```python
import json

with open("data/chunked_output/_annotations.coco.json") as f:
    coco = json.load(f)

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {[c['name'] for c in coco['categories']]}")

# Check bbox distribution
bbox_counts = {}
for anno in coco["annotations"]:
    cat_id = anno["category_id"]
    cat_name = next(c["name"] for c in coco["categories"] if c["id"] == cat_id)
    bbox_counts[cat_name] = bbox_counts.get(cat_name, 0) + 1

print(f"Bbox distribution: {bbox_counts}")
```

✅ **Expected output:**

- At least 100+ chunks for meaningful training
- Balanced bbox distribution (no class with \<10% of data)
- Bboxes visible and aligned in debug images

## Batch Visualization Script

For visualizing many chunks at once:

```python
#!/usr/bin/env python3
"""Generate debug images for all chunks in a dataset."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import supervision as sv


def visualize_coco_chunks(coco_json_path: str, output_dir: str):
    """Create debug images from COCO dataset."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Group annotations by image
    img_to_annos = {}
    for anno in coco["annotations"]:
        img_id = anno["image_id"]
        if img_id not in img_to_annos:
            img_to_annos[img_id] = []
        img_to_annos[img_id].append(anno)

    # Process each image
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_data in coco["images"]:
        img_id = img_data["id"]
        img_file = Path(coco_json_path).parent / img_data["file_name"]

        if not img_file.exists():
            continue

        # Load image
        img = np.array(Image.open(img_file))

        # Get bboxes
        annos = img_to_annos.get(img_id, [])
        if not annos:
            continue

        # Draw bboxes
        bboxes = []
        labels = []
        for anno in annos:
            x, y, w, h = anno["bbox"]
            bboxes.append([x, y, x + w, y + h])

            cat_id = anno["category_id"]
            cat_name = next(c["name"] for c in coco["categories"] if c["id"] == cat_id)
            labels.append(cat_name)

        detections = sv.Detections(
            xyxy=np.array(bboxes), class_id=np.arange(len(bboxes))
        )

        # Annotate
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        annotated = box_annotator.annotate(img.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        # Save
        out_file = output_path / f"{Path(img_data['file_name']).stem}_debug.png"
        Image.fromarray(annotated).save(out_file)
        print(f"Saved: {out_file.name} ({len(bboxes)} bboxes)")


if __name__ == "__main__":
    visualize_coco_chunks(
        "data/chunked_output/_annotations.coco.json", "output/coco_debug_images"
    )
```

Save as `scripts/visualize_chunks.py` and run:

```bash
python scripts/visualize_chunks.py
```

## Advanced: Compare Ground Truth vs Predictions

After training, compare model predictions with ground truth:

```python
from rf_detr_finetuning import prediction
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

test_images = Path("data/chunked_coco_split/test").glob("*.png")

for img_path in list(test_images)[:10]:
    # Get prediction
    pred_visual = prediction(
        image_path=str(img_path),
        model_path="output/checkpoint_best_total.pth",
        model_size="small",
        confidence=0.5,
    )

    # Load ground truth debug image (if exists)
    gt_debug_path = Path("data/chunked_output/debug") / f"{img_path.stem}_debug.png"

    if gt_debug_path.exists():
        gt_visual = np.array(Image.open(gt_debug_path))

        # Side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(gt_visual)
        ax1.set_title("Ground Truth")
        ax1.axis("off")

        ax2.imshow(pred_visual)
        ax2.set_title("Prediction")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(f"output/comparison/{img_path.stem}_comparison.png")
        plt.close()
```

This creates side-by-side images showing ground truth bboxes vs predicted bboxes.
