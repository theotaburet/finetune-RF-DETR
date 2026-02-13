# Event Merging in Inference Pipeline

This document explains how to use class-wise event merging in the RF-DETR inference pipeline.

## Overview

The inference pipeline now supports two merging strategies:

1. **IoU-based merging** (default): Uses intersection-over-union threshold
2. **Class-wise merging**: Uses Delta_Time (temporal distance) and Delta_Hz (frequency distance) per class

## Configuration File

Create a YAML configuration file (e.g., `config/merging.yaml`):

```yaml
# Default merge parameters (applied to all classes unless overridden)
default:
  delta_time_ms: 500.0        # Maximum temporal distance to merge (milliseconds)
  delta_freq_hz: 500.0        # Maximum frequency distance to merge (Hz)
  score_strategy: "max"       # How to combine scores: "max", "avg", or "weighted"

# Class-specific merge parameters
classes:
  # bird_call: Short, distinct calls with tight temporal requirements
  0:
    delta_time_ms: 300        # Merge if within 300ms
    delta_freq_hz: 500        # Allow 500Hz frequency difference
    score_strategy: "max"

  # engine_noise: Long continuous sound, allow larger temporal gaps
  1:
    delta_time_ms: 800        # Merge if within 800ms
    delta_freq_hz: 200        # Strict frequency match
    score_strategy: "avg"

  # dog_bark: Medium duration sounds
  2:
    delta_time_ms: 500
    delta_freq_hz: 300
    score_strategy: "weighted"

# Post-merge filtering
filtering:
  score_threshold: 0.0        # Minimum score to keep (0.0-1.0)
  min_duration_ms: 0.0        # Minimum event duration in milliseconds
  max_duration_ms: null       # Maximum event duration (null = no limit)
```

## Usage

### run_inference_audio.py (Recommended)

```bash
# With class-wise merge config
python run_inference_audio.py \
    --audio data/audio/sample.flac \
    --weights output/checkpoint_best.pth \
    --config config/audio_chunking.yaml \
    --merge-config config/merging.yaml \
    --output results/events.json

# Directory batch processing with merge config
python run_inference_audio.py \
    --audio-dir data/audio/ \
    --weights output/checkpoint_best.pth \
    --config config/audio_chunking.yaml \
    --merge-config config/merging.yaml \
    --output-dir results/

# Without merge config (uses IoU-based merging)
python run_inference_audio.py \
    --audio data/audio/sample.flac \
    --weights output/checkpoint_best.pth \
    --config config/audio_chunking.yaml \
    --iou-threshold 0.5 \
    --output results/events.json
```

### run_inference.py

```bash
# With class-wise merge config
python run_inference.py \
    --audio data/audio/sample.flac \
    --weights output/checkpoint_best.pth \
    --chunking-config config/audio_chunking.yaml \
    --merge-config config/merging.yaml \
    --output results.json

# Without merge config (uses IoU-based merging)
python run_inference.py \
    --audio data/audio/sample.flac \
    --weights output/checkpoint_best.pth \
    --chunking-config config/audio_chunking.yaml \
    --iou-threshold 0.5 \
    --output results.json
```

## Parameters

### Delta_Time (Δt)

- **Type**: float (milliseconds)
- **Description**: Maximum temporal distance between events to consider them for merging
- **Usage**: Events within this time window (and within delta_freq_hz) will be merged
- **Example**: `300` means merge events within 300ms of each other

### Delta_Hz (Δf)

- **Type**: float (Hz)
- **Description**: Maximum frequency distance between events to consider them for merging
- **Usage**: Events within this frequency window (and within delta_time_ms) will be merged
- **Example**: `500` means merge events within 500Hz frequency range

### Score Strategy

- **Type**: string
- **Options**:
  - `"max"`: Take maximum score from merged events (conservative)
  - `"avg"`: Average scores of merged events (balanced)
  - `"weighted"`: Weight by duration (favors longer detections)

## How It Works

1. **Window Detection**: Audio is chunked into overlapping spectrogram windows
2. **Per-window Detection**: Model detects events in each window independently
3. **Class-wise Grouping**: Detections are grouped by class
4. **Merging**: For each class:
   - Events are sorted by start time
   - Adjacent events within Δt AND Δf are merged
   - Score strategy is applied
5. **Filtering**: Events are filtered by score and duration thresholds
6. **Output**: Merged events are saved to JSON/CSV

## Benefits

- **Class-specific behavior**: Different sound types can have different merge tolerances
- **Frequency-aware**: Prevents merging sounds at different frequencies (e.g., bird vs engine)
- **Intuitive parameters**: Time/frequency distances are more intuitive than IoU ratios
- **Better for multi-species**: Each species can have optimized merge parameters

## Testing

Generate test events and run merging:

```bash
# Generate test events
python experiments/generate_test_events.py

# Run merging with config
python run_merging.py \
    --input output/test_merging/test_events.json \
    --config config/merging.yaml \
    --output-dir output/test_merging/results \
    --visualize

# View results
cat output/test_merging/results/merged_events.json
```

## Troubleshooting

**Events not merging:**

- Check that Δt is large enough (events might be too far apart)
- Check that Δf is large enough (events might be at different frequencies)
- Ensure events are in the same class

**Too much merging:**

- Reduce Δt to prevent merging distant events
- Reduce Δf to prevent merging events at different frequencies
- Use `"max"` score strategy to preserve high-confidence scores

**Import errors:**

- Ensure all imports are at the top of the file
- Check that `rf_detr_finetuning.eventprocessor.merger` is properly imported
