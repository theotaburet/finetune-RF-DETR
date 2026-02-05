# Scripts Directory

This directory contains utility scripts for preprocessing, inference, and evaluation.

## Inference Script

### run_inference.py

Run batch inference on the entire test set with evaluation metrics and visualizations.

**Features:**

- Processes all test images (or a subset with `--max_images`)
- Calculates precision, recall, and F1 scores per class
- Computes IoU-based matching with ground truth
- Saves prediction visualizations (optional)
- Generates detailed JSON reports

**Basic Usage:**

```bash
# Run on entire test set with default settings
uv run python scripts/run_inference.py

# Save visualizations with bboxes
uv run python scripts/run_inference.py --save_visualizations

# Test on first 50 images only
uv run python scripts/run_inference.py --max_images 50 --save_visualizations

# Custom confidence and IoU thresholds
uv run python scripts/run_inference.py \
    --confidence 0.25 \
    --iou_threshold 0.5 \
    --save_visualizations

# Different model checkpoint
uv run python scripts/run_inference.py \
    --model_path output/checkpoint_best_total.pth \
    --model_size small \
    --output_dir output/eval_total_checkpoint
```

**Arguments:**

- `--model_path`: Path to checkpoint (default: `output/checkpoint_best_ema.pth`)
- `--test_dir`: Directory with test images and `_annotations.coco.json` (default: `split_dataset/test`)
- `--output_dir`: Where to save results (default: `output/test_predictions`)
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--model_size`: Model size (default: `small`)
- `--iou_threshold`: IoU threshold for matching predictions to GT (default: 0.5)
- `--max_images`: Limit number of images to process (default: all)
- `--save_visualizations`: Save annotated images with bboxes

**Output Files:**

- `predictions_summary.json`: All detections per image
- `evaluation_metrics.json`: Precision/recall/F1 per class and overall
- `{image_name}_pred.png`: Annotated images (if `--save_visualizations` enabled)

**Example Output:**

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━┳━━━━┳━━━━┳━━━━━━━━━━┓
┃ Class         ┃ Precision┃  Recall┃ F1 Score ┃  TP┃  FP┃  FN┃ GT Total ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━╇━━━━╇━━━━╇━━━━━━━━━━┩
│ cargo         │    0.856 │  0.912 │    0.883 │ 145│  24│  14│      159 │
│ odontoceti    │    0.923 │  0.891 │    0.907 │ 234│  20│  29│      263 │
│ OVERALL       │    0.892 │  0.901 │    0.896 │ 379│  44│  43│        - │
└───────────────┴──────────┴────────┴──────────┴────┴────┴────┴──────────┘
```

## Other Scripts

- `chunk_audio_dataset.py`: Convert audio files to chunked spectrograms
- `split_coco_dataset.py`: Split COCO dataset into train/val/test
- `test_preprocessing_visual.py`: Visualize preprocessing parameter effects (see [README_PREPROCESSING_TEST.md](README_PREPROCESSING_TEST.md))
