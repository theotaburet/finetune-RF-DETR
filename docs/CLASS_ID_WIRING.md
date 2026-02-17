# Wiring Class IDs: From Training to Inference to Merging

This document explains how to properly wire class IDs across the entire pipeline to ensure consistent class-wise merging.

## Overview

The system uses **0-based integer class IDs** throughout:

- **Training**: `class_names: ["class_0", "class_1", "class_2"]` in `config/pipeline.yaml`
- **Model output**: Predicts integers 0, 1, 2...
- **Merging**: `classes.0`, `classes.1`, `classes.2` in `config/merging.yaml`
- **Inference output**: Events with `class_id: 0, 1, 2...` and `class_name: "class_0"`

## The Class ID Flow

### 1. Training Configuration (config/pipeline.yaml)

Define your classes as a **list** where **index = class_id**:

```yaml
name: marine_acoustic_pipeline
class_names:
  - humpback_whale     # class_id = 0
  - killer_whale       # class_id = 1
  - blue_whale         # class_id = 2
  - dolphin            # class_id = 3
  - ship_noise         # class_id = 4
  - fish_chorus        # class_id = 5
```

**Key rule**: The order in `class_names` determines the class IDs!

### 2. Merging Configuration (config/merging.yaml)

Use **integer keys** matching the indices from `class_names`:

```yaml
# Class-specific merge parameters
classes:
  # humpback_whale (class_id = 0)
  0:
    delta_time_ms: 3000
    delta_freq_hz: 400
    score_strategy: "max"

  # killer_whale (class_id = 1)
  1:
    delta_time_ms: 800
    delta_freq_hz: 600
    score_strategy: "max"

  # blue_whale (class_id = 2)
  2:
    delta_time_ms: 8000
    delta_freq_hz: 30
    score_strategy: "avg"

  # dolphin (class_id = 3)
  3:
    delta_time_ms: 400
    delta_freq_hz: 800
    score_strategy: "max"

  # ship_noise (class_id = 4)
  4:
    delta_time_ms: 5000
    delta_freq_hz: 100
    score_strategy: "avg"

  # fish_chorus (class_id = 5)
  5:
    delta_time_ms: 2000
    delta_freq_hz: 200
    score_strategy: "weighted"
```

### 3. Complete Working Example

Here's a complete marine acoustic setup:

**config/pipeline.yaml:**

```yaml
name: marine_acoustic_pipeline
class_names:
  - humpback_whale
  - killer_whale
  - blue_whale
  - dolphin
  - ship_noise

preprocess:
  enabled: true
  audio_dir: data/hydrophone/
  metadata_dir: data/annotations/
  output_dir: data/processed
  chunking_config: config/audio_chunking.yaml

train:
  enabled: true
  dataset_dir: data/split_dataset
  output_dir: output/
  model_size: base
  epochs: 50

infer:
  enabled: true
  weights: output/checkpoint_best.pth
  audio_dir: data/inference/
  output_dir: output/predictions/
```

**config/merging.yaml:**

```yaml
default:
  delta_time_ms: 2000
  delta_freq_hz: 150
  score_strategy: "max"

classes:
  0:  # humpback_whale
    delta_time_ms: 3000
    delta_freq_hz: 400
    score_strategy: "max"

  1:  # killer_whale
    delta_time_ms: 800
    delta_freq_hz: 600
    score_strategy: "max"

  2:  # blue_whale
    delta_time_ms: 8000
    delta_freq_hz: 30
    score_strategy: "avg"

  3:  # dolphin
    delta_time_ms: 400
    delta_freq_hz: 800
    score_strategy: "max"

  4:  # ship_noise
    delta_time_ms: 5000
    delta_freq_hz: 100
    score_strategy: "avg"

filtering:
  score_threshold: 0.3
  min_duration_ms: 200.0
```

## Using the Pipeline

### Training

```bash
python run_pipeline.py --config config/pipeline.yaml
```

The pipeline will:

1. Read `class_names` from `config/pipeline.yaml`
2. Train model with 5 output classes (0-4)
3. Save checkpoint to `output/checkpoint_best.pth`

### Inference with Merging

```bash
# Run inference with class-wise merging
python run_inference_audio.py \
    --audio data/hydrophone/recording.flac \
    --weights output/checkpoint_best.pth \
    --config config/audio_chunking.yaml \
    --merge-config config/merging.yaml \
    --class-names humpback_whale killer_whale blue_whale dolphin ship_noise \
    --output results/events.json
```

**Important**: Use `--class-names` to ensure inference uses the same class order as training!

## How Class IDs Are Used

### During Training

```python
# Model output: tensor of shape [batch, num_classes, 4]
# Each detection has: [x, y, w, h, class_id]
# class_id is 0, 1, 2, 3, or 4
```

### During Inference

```python
# Detection from model:
detection = {
    "bbox": [x1, y1, x2, y2],
    "score": 0.87,
    "class_id": 0,  # Model predicts integer
    "class_name": "humpback_whale",  # Looked up from class_names[0]
}
```

### During Merging

```python
# ClassWiseMerger groups by class_id:
#   class_id=0 (humpback_whale) → use params from classes.0
#   class_id=1 (killer_whale) → use params from classes.1

# Merge parameters from merging.yaml:
merge_params = config.class_params[event.class_id]
```

## Verifying the Wiring

### 1. Check Class Count

```bash
# After training, check the model knows about all classes
python -c "
from rf_detr_finetuning import RFDETRPredictor
pred = RFDETRPredictor(weights_path='output/checkpoint_best.pth')
print(f'Number of classes: {pred.num_classes}')
print(f'Class names: {pred.class_names}')
"
```

### 2. Check Merging Config

```bash
python -c "
import yaml
with open('config/merging.yaml') as f:
    config = yaml.safe_load(f)

with open('config/pipeline.yaml') as f:
    pipeline = yaml.safe_load(f)

print('Pipeline class_names:', pipeline.get('class_names', []))
print('Merging classes:', list(config.get('classes', {}).keys()))

# Check if all class_ids are covered
class_ids = set(range(len(pipeline.get('class_names', []))))
merging_ids = set(config.get('classes', {}).keys())
missing = class_ids - merging_ids
if missing:
    print(f'WARNING: Missing merge config for class_ids: {missing}')
else:
    print('✓ All classes have merge configuration')
"
```

### 3. Check Inference Output

```bash
# After inference, verify class_ids match
python -c "
import json
with open('results/events.json') as f:
    data = json.load(f)

print('Class names mapping:', data.get('class_names', {}))
print('\\nDetected events:')
for event in data.get('events', []):
    print(f\"  {event['class_name']} (id={event['class_id']}): {event['start_ms']}-{event['end_ms']}ms\")
"
```

## Common Issues and Solutions

### Issue: "Class X not found in merging config"

**Cause**: Merging config doesn't have entry for all class IDs

**Solution**: Add missing class to `config/merging.yaml`:

```yaml
classes:
  # ... existing classes ...
  5:  # fish_chorus (add this if missing)
    delta_time_ms: 2000
    delta_freq_hz: 200
    score_strategy: "weighted"
```

### Issue: "Wrong species being merged"

**Cause**: Class IDs don't match between pipeline.yaml and merging.yaml

**Solution**: Verify the order:

```bash
# Check pipeline.yaml
class_names: ["humpback", "killer_whale", "dolphin"]  # Index 0, 1, 2

# Check merging.yaml
classes:
  0:  # Must be humpback params
    delta_time_ms: 3000
  1:  # Must be killer_whale params
    delta_time_ms: 800
  2:  # Must be dolphin params
    delta_time_ms: 400
```

### Issue: "Class names mismatch in inference"

**Cause**: Inference using different class order than training

**Solution**: Always specify `--class-names` during inference:

```bash
python run_inference_audio.py \
    --audio ... \
    --class-names humpback_whale killer_whale blue_whale dolphin ship_noise \
    ...
```

Or use a classes file:

```bash
# Create classes.json
echo '["humpback_whale", "killer_whale", "blue_whale", "dolphin", "ship_noise"]' > classes.json

# Use in inference
python run_inference_audio.py \
    --audio ... \
    --classes-file classes.json \
    ...
```

## Best Practices

1. **Keep a master class list** in one place (e.g., `config/pipeline.yaml`)

2. **Document your classes** with comments:

   ```yaml
   class_names:
     - humpback_whale    # 0: Complex songs, 10 Hz - 4 kHz
     - killer_whale      # 1: Pulsed calls, 1-20 kHz
     - blue_whale        # 2: Infrasonic calls, 10-40 Hz
   ```

3. **Use consistent naming** across all config files

4. **Test the wiring** with a small dataset before full training

5. **Save class configuration** with your model:

   ```bash
   # Copy configs with model
   cp config/pipeline.yaml output/pipeline_config.yaml
   cp config/merging.yaml output/merging_config.yaml
   ```

6. **Version your configs** when adding/removing classes

## Quick Reference

| File                   | Format                                   | Example                    |
| ---------------------- | ---------------------------------------- | -------------------------- |
| `config/pipeline.yaml` | List: `class_names: ["A", "B", "C"]`     | Index 0=A, 1=B, 2=C        |
| `config/merging.yaml`  | Dict: `classes.0`, `classes.1`           | Integer keys match indices |
| Model output           | Integer: 0, 1, 2                         | Predicted class_id         |
| Inference CLI          | `--class-names A B C`                    | Same order as pipeline     |
| COCO annotations       | `"categories": [{"id": 1, "name": "A"}]` | 1-based in COCO            |

## Example: Complete Marine Acoustic Setup

```bash
# 1. Define classes in pipeline config
cat > config/pipeline.yaml << 'EOF'
name: marine_pipeline
class_names:
  - humpback_whale
  - killer_whale
  - blue_whale
  - dolphin
  - ship_noise

train:
  enabled: true
  dataset_dir: data/processed
  output_dir: output/
  epochs: 50
EOF

# 2. Define merge parameters
cat > config/merging.yaml << 'EOF'
classes:
  0: {delta_time_ms: 3000, delta_freq_hz: 400, score_strategy: "max"}
  1: {delta_time_ms: 800, delta_freq_hz: 600, score_strategy: "max"}
  2: {delta_time_ms: 8000, delta_freq_hz: 30, score_strategy: "avg"}
  3: {delta_time_ms: 400, delta_freq_hz: 800, score_strategy: "max"}
  4: {delta_time_ms: 5000, delta_freq_hz: 100, score_strategy: "avg"}
filtering:
  score_threshold: 0.3
  min_duration_ms: 200.0
EOF

# 3. Train
python run_pipeline.py --config config/pipeline.yaml

# 4. Inference with merging
python run_inference_audio.py \
    --audio data/hydrophone/recording.flac \
    --weights output/checkpoint_best.pth \
    --config config/audio_chunking.yaml \
    --merge-config config/merging.yaml \
    --class-names humpback_whale killer_whale blue_whale dolphin ship_noise \
    --output results/events.json

# 5. Verify output
python -c "
import json
with open('results/events.json') as f:
    data = json.load(f)
print('Detected events by class:')
for event in data['events']:
    print(f\"  {event['class_name']}: {event['start_ms']}-{event['end_ms']}ms\")
"
```
