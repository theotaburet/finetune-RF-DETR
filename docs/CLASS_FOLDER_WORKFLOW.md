# Audio Dataset Processing: Class Folder Structure

This guide explains how to process audio datasets organized by class folders and train RF-DETR.

## Dataset Structure

```
my_audio_dataset/
├── biological-cetacean-odontoceti/
│   ├── dolphin_001.flac
│   ├── dolphin_002.flac
│   └── ...
├── biological-cetacean-mysticeti/
│   ├── whale_001.flac
│   └── ...
├── anthropogenic-ship-cargo/
│   ├── cargo_001.flac
│   └── ...
├── anthropogenic-ship-tanker/
│   └── ...
└── biological-fish/
    └── ...
```

**Category Extraction:**

- Folder name: `biological-cetacean-odontoceti`
- Extracted category: `odontoceti` (last part after dashes)

## Complete Pipeline

### Step 1: Chunk Audio → COCO Format

```bash
python scripts/chunk_audio_dataset.py \
    --input-dir ./my_audio_dataset \
    --output-dir ./chunked_coco \
    --config config/audio_chunking.yaml \
    --draw-bboxes
```

**Output:**

```
chunked_coco/
├── images/
│   ├── dolphin_001_chunk0000.png  # 640x640 spectrogram
│   ├── dolphin_001_chunk0001.png
│   ├── whale_001_chunk0000.png
│   └── ...
├── debug/  # If --draw-bboxes
│   ├── dolphin_001_chunk0000_debug.png
│   └── ...
└── _annotations.coco.json
```

**COCO Format:**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "dolphin_001_chunk0000.png",
      "width": 640,
      "height": 640,
      "audio_metadata": {
        "source_file": "dolphin_001.flac",
        "category": "odontoceti",
        "chunk_index": 0,
        "start_ms": 0,
        "end_ms": 6400
      }
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [0, 77, 640, 430],
      "area": 275200
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "odontoceti",
      "supercategory": "audio_event"
    },
    {
      "id": 1,
      "name": "mysticeti",
      "supercategory": "audio_event"
    }
  ]
}
```

### Step 2: Split into Train/Val/Test

```bash
python scripts/split_coco_dataset.py \
    --input-coco ./chunked_coco/_annotations.coco.json \
    --input-images ./chunked_coco/images \
    --output-dir ./split_dataset \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --seed 42
```

**Output:**

```
split_dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── val/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### Step 3: Train RF-DETR

```bash
uv run python -m rf_detr_finetuning train \
    --config_file config/audio_train.yaml \
    --dataset ./split_dataset \
    --model_size small
```

**Or using positional arguments:**

```bash
uv run python -m rf_detr_finetuning train \
    config/audio_train.yaml \
    ./split_dataset \
    small
```

### Step 4: Predict on New Audio

```bash
# 1. Chunk new audio
python scripts/chunk_audio_dataset.py \
    --input-dir ./new_audio \
    --output-dir ./new_audio_chunks \
    --config config/audio_chunking.yaml

# 2. Predict on chunks
uv run python -m rf_detr_finetuning predict \
    --image-path ./new_audio_chunks/images/new_file_chunk0000.png \
    --model-path ./training_output/checkpoint_best.pth \
    --model-size small \
    --output-dir ./predictions
```

## Configuration

### config/audio_chunking.yaml

```yaml
fft:
  hop_ms: 10.0      # Time resolution: 10ms per pixel
  fft_ms: 25.0      # Frequency resolution: 25ms FFT window
  n_mels: 128       # Mel frequency bands

chunking:
  target_size: 640           # Auto: width=640, height=640, window=6400ms
  overlap_ratio: 0.2         # 20% overlap (1280ms)
  min_overlap_with_event_ratio: 0.3
  padding_mode: zero
  min_chunk_content_ratio: 0.5  # Drop chunks with <50% audio
  random_pad_position: true     # Augmentation for short audio

spectrogram:
  freq_scale: mel
  fmin: 0.0
  fmax: null  # Auto = Nyquist
```

## Customization

### If You Have Time/Frequency Annotations

Modify `scripts/chunk_audio_dataset.py` to read JSON files:

```python
# Instead of file-level annotation:
json_path = audio_path.with_suffix(".json")
if json_path.exists():
    with open(json_path) as f:
        meta = json.load(f)

    events = [
        {
            "time_start_ms": meta["time_start_ms"],
            "time_end_ms": meta["time_end_ms"],
            "hz_min": meta["hz_min"],
            "hz_max": meta["hz_max"],
            "category": category_name,
            "category_id": category_id,
            "is_file_level": False,  # Event-level (not full file)
        }
    ]
else:
    # Fallback to file-level
    events = [...]
```

### Custom Category Extraction

Modify `extract_category_from_path()`:

```python
def extract_category_from_path(audio_path: Path, root_dir: Path) -> str:
    """Your custom logic."""
    # Option 1: Use full folder name
    return relative.parts[0]

    # Option 2: Use hierarchical categories
    parts = relative.parts[0].split("-")
    return f"{parts[0]}_{parts[-1]}"  # biological_odontoceti

    # Option 3: Use filename patterns
    if "dolphin" in audio_path.stem:
        return "odontoceti"
```

## Visual Verification

```bash
# Generate debug images
python scripts/chunk_audio_dataset.py \
    --input-dir ./my_audio_dataset \
    --output-dir ./chunked_coco \
    --config config/audio_chunking.yaml \
    --draw-bboxes

# View debug images
eog ./chunked_coco/debug/*.png
```

**What to check:**

- ✅ Bboxes cover the visible frequency range
- ✅ No huge black padding regions
- ✅ Labels match category folders
- ✅ Time axis is consistent (no stretching)

## Troubleshooting

### Issue: Categories not extracted correctly

**Solution:** Modify `extract_category_from_path()` to match your folder naming

### Issue: Black padding blobs

**Solution:** Increase `target_size` or adjust `min_chunk_content_ratio`

### Issue: Missing annotations

**Solution:** Check `min_overlap_with_event_ratio` (lower = more permissive)

### Issue: Wrong frequency range

**Solution:** If you have JSON metadata, use `hz_min`/`hz_max` instead of `0`/`fmax`

## Performance Tips

- **Batch processing**: Script processes files sequentially; for large datasets, consider parallel processing
- **Disk space**: 640x640 PNG ≈ 100-200 KB per chunk; estimate: `num_files × avg_chunks_per_file × 150 KB`
- **Speed**: ~5-10 files/second on CPU (depends on audio length)

## Next Steps

1. Run chunking on a small subset first (verify debug images)
2. Process full dataset
3. Split into train/val/test
4. Train RF-DETR
5. Evaluate on test set
6. Deploy for inference on new audio files
