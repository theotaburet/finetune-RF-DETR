# Audio Training Workflow with RF-DETR

Complete guide for training RF-DETR on variable-length audio using sliding window chunking.

## Dataset Structure

### Input: Raw Audio Dataset

```
my_audio_dataset/
├── audio_file_001.flac          # Audio file
├── audio_file_001.json          # Annotations (same basename)
├── audio_file_002.wav
├── audio_file_002.json
├── audio_file_003.flac
├── audio_file_003.json
└── ...
```

### JSON Annotation Format

Each `.json` file contains temporal bounding boxes:

```json
{
  "sample_rate": 48000,
  "duration_ms": 12500,
  "events": [
    {
      "start_time_ms": 1000,
      "end_time_ms": 2500,
      "min_frequency_hz": 500,
      "max_frequency_hz": 2000,
      "label": "bird_call"
    },
    {
      "start_time_ms": 5000,
      "end_time_ms": 8000,
      "min_frequency_hz": 100,
      "max_frequency_hz": 800,
      "label": "engine_noise"
    }
  ]
}
```

**Required fields:**

- `sample_rate`: Audio sample rate (e.g., 48000)
- `duration_ms`: Total duration in milliseconds
- `events`: List of temporal events with:
  - `start_time_ms`, `end_time_ms`: Time boundaries
  - `min_frequency_hz`, `max_frequency_hz`: Frequency range
  - `label`: Event category name

## Step 1: Configure Chunking

Create `config/my_chunking_config.yaml`:

```yaml
# Time-based FFT (sample-rate independent)
time_based_fft:
  fft_ms: 25.0          # 25ms FFT window
  hop_ms: 10.0          # 10ms hop (60% overlap)
  n_mels: 128           # Mel filterbanks

# Chunking parameters
chunk_config:
  window_duration_ms: 5000.0      # 5-second chunks
  overlap_ms: 1000.0              # 1-second overlap (20%)
  target_width: 640               # RF-DETR expects ~640x640
  target_height: 640
  padding_mode: "zero"            # zero, repeat, or reflect
  min_overlap_with_event_ratio: 0.3  # Event needs 30% in chunk
```

**Rationale:**

- **25ms FFT**: Standard speech/audio analysis window
- **10ms hop**: ~60% overlap for smooth spectrograms
- **5s chunks**: Balances context vs. memory (640x640 @ 10ms hop ≈ 6.4s)
- **1s overlap**: Ensures events near boundaries appear in multiple chunks
- **30% overlap threshold**: Filters out partial events at chunk edges

## Step 2: Chunk Audio to COCO Format

```bash
# Using YAML config (recommended)
uv run rf-detr convert chunk-audio \
    --input-dir ./data/my_audio_dataset \
    --output-dir ./data/chunked_coco \
    --config-file ./config/my_chunking_config.yaml \
    --draw-bboxes \
    --debug-output-dir ./data/chunked_coco/debug

# Using CLI arguments (quick testing)
uv run rf-detr convert chunk-audio \
    --input-dir ./data/my_audio_dataset \
    --output-dir ./data/chunked_coco \
    --window-duration-ms 5000 \
    --overlap-ms 1000 \
    --target-width 640 \
    --target-height 640 \
    --fft-ms 25 \
    --hop-ms 10 \
    --n-mels 128 \
    --padding-mode zero \
    --min-overlap-ratio 0.3 \
    --draw-bboxes
```

### Output Structure

```
data/chunked_coco/
├── _annotations.coco.json       # COCO format annotations
├── images/
│   ├── audio_file_001_chunk0000.png
│   ├── audio_file_001_chunk0001.png
│   ├── audio_file_002_chunk0000.png
│   └── ...
└── debug/                        # Visual bbox overlays (if --draw-bboxes)
    ├── audio_file_001_chunk0000_debug.png
    ├── audio_file_001_chunk0001_debug.png
    └── ...
```

**COCO Format:**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "images/audio_file_001_chunk0000.png",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  # COCO format: [x_min, y_min, w, h]
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "bird_call"},
    {"id": 2, "name": "engine_noise"}
  ]
}
```

## Step 3: Split Dataset (Train/Val/Test)

### Option A: Manual Split

```bash
# Create train/val/test directories
mkdir -p data/chunked_coco_split/{train,valid,test}

# Move 70% to train, 20% to valid, 10% to test
# Then manually split the COCO JSON or use jq/Python script
```

### Option B: Use Existing `convert_yolo_to_coco` Split Logic

The existing pipeline in `data.py` handles splits. You can adapt it:

```python
from rf_detr_finetuning import convert_yolo_to_coco
from pathlib import Path
import json
import shutil

# Load chunked COCO dataset
with open("data/chunked_coco/_annotations.coco.json") as f:
    coco_data = json.load(f)

# Split images into train/val/test (70/20/10)
from sklearn.model_selection import train_test_split

images = coco_data["images"]
train_imgs, test_imgs = train_test_split(images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(
    test_imgs, test_size=0.33, random_state=42
)  # 0.3*0.33 ≈ 0.1

# Create split directories
for split, split_imgs in [
    ("train", train_imgs),
    ("valid", val_imgs),
    ("test", test_imgs),
]:
    split_dir = Path(f"data/chunked_coco_split/{split}")
    split_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in split_imgs:
        src = Path("data/chunked_coco") / img["file_name"]
        dst = split_dir / Path(img["file_name"]).name
        shutil.copy(src, dst)

    # Filter annotations
    image_ids = {img["id"] for img in split_imgs}
    split_annos = [a for a in coco_data["annotations"] if a["image_id"] in image_ids]

    # Save split COCO JSON
    split_coco = {
        "images": split_imgs,
        "annotations": split_annos,
        "categories": coco_data["categories"],
    }
    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(split_coco, f, indent=2)
```

### Final Dataset Structure

```
data/chunked_coco_split/
├── train/
│   ├── _annotations.coco.json
│   ├── audio_file_001_chunk0000.png
│   ├── audio_file_002_chunk0000.png
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

## Step 4: Configure Training

Create `config/audio_training.yaml`:

```yaml
# Training configuration for RF-DETR on audio spectrograms

model:
  num_classes: 2  # UPDATE: Number of your categories (bird_call, engine_noise, etc.)

training:
  epochs: 50
  batch_size: 8   # Adjust based on GPU memory
  learning_rate: 0.0001
  weight_decay: 0.0001

data:
  train_ann_file: "train/_annotations.coco.json"
  val_ann_file: "valid/_annotations.coco.json"
  num_workers: 4

optimizer:
  type: "AdamW"
  lr: 0.0001
  weight_decay: 0.0001

lr_scheduler:
  type: "MultiStepLR"
  milestones: [30, 40]
  gamma: 0.1

augmentation:
  # Spectrogram-specific augmentations
  horizontal_flip: 0.5      # Time-axis flip
  random_crop: false        # Keep fixed 640x640
  color_jitter: true        # Intensity variations
```

## Step 5: Train the Model

```bash
# Train from pretrained RF-DETR weights
uv run rf-detr train \
    --config-file ./config/audio_training.yaml \
    --dataset ./data/chunked_coco_split \
    --model-size small

# Resume from checkpoint
uv run rf-detr train \
    --config-file ./config/audio_training.yaml \
    --dataset ./data/chunked_coco_split \
    --model-size small \
    --resume ./output/checkpoint.pth
```

Training outputs:

```
output/
├── checkpoint.pth              # Latest checkpoint
├── checkpoint_best_ema.pth     # Best EMA model
├── checkpoint_best_regular.pth # Best regular model
├── checkpoint_best_total.pth   # Best overall
├── log.txt                     # Training logs
├── results.json                # Evaluation metrics
└── metrics_plot.png            # Loss/mAP curves
```

## Step 6: Test & Visualize Results

### A. Run Inference on Test Spectrograms

```bash
# Predict on a single spectrogram
uv run rf-detr predict \
    --image-path ./data/chunked_coco_split/test/audio_file_005_chunk0002.png \
    --model-path ./output/checkpoint_best_total.pth \
    --model-size small \
    --confidence 0.5
```

Output: `output/prediction.png` with predicted bboxes overlaid

### B. Batch Inference with Python

```python
from rf_detr_finetuning import prediction
from pathlib import Path
import supervision as sv
import matplotlib.pyplot as plt

test_dir = Path("data/chunked_coco_split/test")
output_dir = Path("output/test_predictions")
output_dir.mkdir(exist_ok=True)

for img_path in test_dir.glob("*.png"):
    if img_path.stem.endswith("_debug"):
        continue

    # Run prediction
    visual = prediction(
        image_path=str(img_path),
        model_size="small",
        model_path="output/checkpoint_best_total.pth",
        confidence=0.5,
        class_names={1: "bird_call", 2: "engine_noise"},
    )

    # Save annotated result
    output_path = output_dir / f"{img_path.stem}_pred.png"
    plt.imsave(output_path, visual)
    print(f"Saved: {output_path}")
```

### C. Compare Ground Truth vs Predictions

```python
import json
import numpy as np
from PIL import Image
import supervision as sv

# Load ground truth COCO
with open("data/chunked_coco_split/test/_annotations.coco.json") as f:
    coco = json.load(f)

# Create image_id -> annotations mapping
img_to_annos = {}
for anno in coco["annotations"]:
    img_id = anno["image_id"]
    if img_id not in img_to_annos:
        img_to_annos[img_id] = []
    img_to_annos[img_id].append(anno)

# Visualize ground truth vs predictions
for img_data in coco["images"][:10]:  # First 10 images
    img_path = Path("data/chunked_coco_split/test") / Path(img_data["file_name"]).name
    img = np.array(Image.open(img_path))

    # Draw ground truth bboxes
    gt_bboxes = []
    for anno in img_to_annos.get(img_data["id"], []):
        x, y, w, h = anno["bbox"]
        gt_bboxes.append([x, y, x + w, y + h])

    if gt_bboxes:
        gt_detections = sv.Detections(
            xyxy=np.array(gt_bboxes), class_id=np.zeros(len(gt_bboxes), dtype=int)
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
        img_gt = box_annotator.annotate(img.copy(), gt_detections)

        # Save ground truth visualization
        Image.fromarray(img_gt).save(f"output/comparison/{img_path.stem}_gt.png")

    print(f"Processed: {img_path.name}")
```

## Testing Workflow Summary

### Quick Test Pipeline

```bash
# 1. Chunk a small sample dataset with debug outputs
uv run rf-detr convert chunk-audio \
    --input-dir ./data/sample_audio \
    --output-dir ./data/test_chunks \
    --config-file ./config/my_chunking_config.yaml \
    --draw-bboxes

# 2. Inspect debug images to verify bbox alignment
ls -lh data/test_chunks/debug/

# 3. Check COCO format
python -c "import json; print(json.dumps(json.load(open('data/test_chunks/_annotations.coco.json')), indent=2)[:500])"

# 4. If bboxes look good, chunk full dataset
uv run rf-detr convert chunk-audio \
    --input-dir ./data/full_audio_dataset \
    --output-dir ./data/chunked_full \
    --config-file ./config/my_chunking_config.yaml

# 5. Split dataset (use Python script above)
python scripts/split_coco_dataset.py

# 6. Train
uv run rf-detr train \
    --config-file ./config/audio_training.yaml \
    --dataset ./data/chunked_coco_split \
    --model-size small

# 7. Monitor training
tail -f output/log.txt

# 8. Test on holdout set
uv run rf-detr predict \
    --image-path ./data/chunked_coco_split/test/sample_chunk.png \
    --model-path ./output/checkpoint_best_total.pth \
    --model-size small
```

## Troubleshooting

### Issue: Chunks have no bboxes

**Cause:** `min_overlap_with_event_ratio` too high or events smaller than chunk window

**Solution:** Lower `min_overlap_with_event_ratio` to 0.1 or adjust `window_duration_ms`

### Issue: Too many chunks (millions)

**Cause:** Large overlap on long audio files

**Solution:** Reduce `overlap_ms` or increase `window_duration_ms`

### Issue: Bboxes at wrong positions

**Cause:** Incorrect time/frequency scaling

**Solution:** Verify `fft_ms`/`hop_ms` match spectrogram generation. Check debug images.

### Issue: Model not converging

**Cause:** Too few training samples or wrong augmentations

**Solution:**

- Increase overlap to generate more chunks
- Disable horizontal flip if temporal order matters
- Check class balance in COCO annotations

## Advanced: Custom Split Script

Save as `scripts/split_coco_dataset.py`:

```python
#!/usr/bin/env python3
"""Split COCO dataset into train/val/test."""

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
):
    """Split COCO dataset into train/val/test."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    # Load COCO
    with open(input_coco_path) as f:
        coco = json.load(f)

    # Split images
    images = coco["images"]
    train_imgs, temp_imgs = train_test_split(
        images, test_size=(1 - train_ratio), random_state=random_seed
    )
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

        # Copy images
        for img in split_imgs:
            src = Path(input_images_dir) / img["file_name"]
            dst = split_dir / Path(img["file_name"]).name
            shutil.copy(src, dst)
            # Update file_name to be relative
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
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(split_coco, f, indent=2)

        print(f"{split_name}: {len(split_imgs)} images, {len(split_annos)} annotations")


if __name__ == "__main__":
    split_coco(
        input_coco_path="data/chunked_coco/_annotations.coco.json",
        input_images_dir="data/chunked_coco/images",
        output_dir="data/chunked_coco_split",
    )
```

Run with:

```bash
python scripts/split_coco_dataset.py
```

## Next Steps

1. **Evaluate mAP**: Compare model performance across different chunking configs
2. **Optimize chunk size**: Try 3s/10s windows based on typical event duration
3. **Export model**: Convert to ONNX/TorchScript for deployment
4. **Real-time inference**: Process streaming audio with sliding window

Good luck with your audio detection model! 🎵
