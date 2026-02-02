# Frequency-Aware Audio Classification for RF-DETR

This guide explains how to use intelligent frequency-based classification where categories are automatically inferred from bbox coordinates (Hz and time values).

## The Problem

In audio event detection on spectrograms, the **same annotation can mean different things at different frequencies**:

- A horizontal line at 50 Hz = **ship noise**
- A horizontal line at 5000 Hz = **sonar pulse**
- A vertical line = **impulse/click**
- A blob shape = **call/vocalization**

Traditional classification ignores this frequency context, but with spectrograms, **pixel coordinates encode Hz and time information**.

## The Solution

We decode bbox coordinates back to Hz/time values and use them as **additional features for classification**:

```
Bbox Pixels → Decode → Hz/Time → Frequency Features → Enhanced Classification
(x, y, w, h)         (871 Hz)   (log_hz, band, bw)    "whale_call_low"
```

## Automatic Frequency Bin Inference

Instead of manually defining frequency bins, let the data decide:

```bash
# Automatically infer 3 frequency bins from data distribution
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /mnt/d/SoundBase/Sounds/SDK/geoacoustic \
  --output_dir data/audio_coco \
  --auto_infer_bins 3

# Output:
# Inferred 3 frequency bins from 7 samples:
#   low: 754.0 - 871.0 Hz
#   mid: 871.0 - 1039.0 Hz
#   high: 1039.0 - 8542.0 Hz
```

This uses percentile-based clustering to create optimal bins based on your data.

## Python API

### Decode Bbox to Hz/Time

```python
from rf_detr_finetuning import (
    decode_bbox_to_frequency,
    BoundingBox,
    FrequencyMapper,
    TimeMapper,
)

# Create a bbox (e.g., from model prediction)
bbox = BoundingBox(
    x=100.0, y=50.0, width=500.0, height=30.0, category_id=0, category_name="unknown"
)

# Create mappers for your spectrogram
freq_mapper = FrequencyMapper(n_mels=128, fmin=0.0, fmax=22050.0, sample_rate=44100)
time_mapper = TimeMapper(
    total_duration_ms=15000, total_frames=1000, sample_rate=44100, hop_length=512
)

# Decode bbox coordinates to Hz and time
freq_info = decode_bbox_to_frequency(bbox, freq_mapper, time_mapper)

print(f"Frequency: {freq_info.hz_min:.1f} - {freq_info.hz_max:.1f} Hz")
print(f"Center: {freq_info.hz_center:.1f} Hz")
print(f"Bandwidth: {freq_info.bandwidth_hz:.1f} Hz")
print(f"Duration: {freq_info.duration_ms:.1f} ms")
print(f"Band: {freq_info.get_frequency_band()}")
# Output:
# Frequency: 10500.2 - 12800.5 Hz
# Center: 11650.3 Hz
# Bandwidth: 2300.3 Hz
# Duration: 5800.0 ms
# Band: high
```

### Auto-Infer Bins from Data

```python
from rf_detr_finetuning import (
    infer_frequency_bins_from_data,
    iterate_audio_dataset,
)
from pathlib import Path

# Collect audio files
audio_files = list(
    iterate_audio_dataset(Path("/mnt/d/SoundBase/Sounds/SDK/geoacoustic"))
)

# Infer 4 frequency bins
bins = infer_frequency_bins_from_data(audio_files, n_bins=4)

for hz_min, hz_max, name in bins:
    print(f"{name}: {hz_min:.1f} - {hz_max:.1f} Hz")
# Output:
# very_low: 31.0 - 572.0 Hz
# low: 572.0 - 890.0 Hz
# high: 890.0 - 1780.0 Hz
# very_high: 1780.0 - 16512.0 Hz
```

### Full Conversion with Auto-Bins

```python
from rf_detr_finetuning import convert_audio_to_coco, SpectrogramConfig

output_path = convert_audio_to_coco(
    input_dir="/mnt/d/SoundBase/Sounds/SDK/geoacoustic",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(n_mels=128, hop_length=512),
    auto_infer_bins=5,  # Infer 5 bins automatically
)
```

## COCO Annotations with Frequency Info

The converted COCO dataset includes decoded frequency information:

```json
{
  "images": [{
    "id": 1,
    "file_name": "earthquake.png",
    "width": 1364,
    "height": 128,
    "audio_metadata": {
      "n_frames": 1364,
      "n_mels": 128,
      "frequency_info": {
        "hz_min": 0.0,
        "hz_max": 16205.0,
        "hz_center": 8102.5,
        "time_start_ms": 0.0,
        "time_end_ms": 15836.0,
        "duration_ms": 15836.0,
        "bandwidth_hz": 16205.0,
        "frequency_band": "high"
      }
    }
  }],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 0,
    "bbox": [0, 10.9, 1364, 117.1],
    "attributes": {
      "hz_min": 0.0,
      "hz_max": 16205.0
    }
  }]
}
```

## Using Frequency Features in Model Training

See [frequency_aware_classification.py](../examples/frequency_aware_classification.py) for a complete example.

### Extract Frequency Features

```python
import json
import math
from pathlib import Path

# Load COCO annotations
with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

# Extract frequency features for each annotation
for ann in coco["annotations"]:
    image_id = ann["image_id"]
    image = next(img for img in coco["images"] if img["id"] == image_id)

    freq_info = image["audio_metadata"]["frequency_info"]

    # Create feature vector
    ann["freq_features"] = {
        # Log scale because:
        # 1. Human perception is logarithmic (octaves)
        # 2. Matches mel scale used in spectrogram
        # 3. Numerical stability (20-20000 Hz → 1.3-4.3)
        # 4. Equal importance across spectrum
        "log_hz_center": math.log10(max(freq_info["hz_center"], 1.0)),
        "log_bandwidth": math.log10(max(freq_info["bandwidth_hz"], 1.0)),
        "bandwidth_norm": min(freq_info["bandwidth_hz"] / 10000.0, 1.0),
        "frequency_band": freq_info["frequency_band"],
    }
```

## Why Log Scale for Frequency?

Frequency features use **logarithmic scale** (`log10(Hz)`) instead of raw Hz values:

| Reason                      | Explanation                                                              | Example                                       |
| --------------------------- | ------------------------------------------------------------------------ | --------------------------------------------- |
| **Human perception**        | We perceive frequency in octaves (doubling), not linearly                | 100→200 Hz sounds similar to 1000→2000 Hz     |
| **Mel scale compatibility** | Spectrograms already use mel (quasi-log). Log Hz maintains consistency   | Mel = 2595 × log10(1 + Hz/700)                |
| **Numerical stability**     | Raw Hz creates huge ranges that destabilize neural networks              | 20 Hz vs 20000 Hz → log: 1.3 vs 4.3           |
| **Feature importance**      | Linear Hz would be dominated by high frequencies. Log gives equal weight | 100→200 Hz same log-distance as 5000→10000 Hz |

**Example**: For a 1000 Hz event, `log10(1000) = 3.0` is much better for ML than raw 1000.

### Extract Frequency Features

### Integrate with RF-DETR (Conceptual)

To use frequency features in RF-DETR, you would:

1. **Extract frequency info during training**:

   ```python
   # In your training loop
   for image_id, bbox, features in dataloader:
       freq_info = get_frequency_info(image_id, bbox)
       freq_features = encode_frequency(freq_info)  # [log_hz, bandwidth, ...]
   ```

2. **Add frequency conditioning layer**:

   ```python
   # Concatenate with visual features before classification
   visual_features = encoder(image)  # [batch, 256]
   freq_features = frequency_encoder(freq_info)  # [batch, 16]
   combined = torch.cat([visual_features, freq_features], dim=1)
   class_logits = classifier(combined)
   ```

3. **Train with frequency-aware loss**:

   - Penalize confusions between same annotation at different frequencies
   - Encourage learning frequency-specific patterns

## Frequency Band Classification

The decoded frequency info includes automatic band classification:

| Band         | Hz Range      | Typical Sources                      |
| ------------ | ------------- | ------------------------------------ |
| `infrasonic` | < 100 Hz      | Earthquakes, explosions, large ships |
| `very_low`   | 100-500 Hz    | Heavy machinery, small vessels       |
| `low`        | 500-2000 Hz   | Marine mammals, engines              |
| `mid`        | 2000-5000 Hz  | Dolphins, fish, some sonars          |
| `high`       | 5000-10000 Hz | Porpoises, high-freq sonars          |
| `very_high`  | > 10000 Hz    | Echolocators, ice noise, rain        |

## Hyperparameter Tuning: `n_bins`

**The number of frequency bins (`auto_infer_bins`) is a critical hyperparameter** that should be tuned for your specific dataset.

### How to Choose n_bins

| Dataset Characteristic        | Recommended n_bins | Reasoning                                            |
| ----------------------------- | ------------------ | ---------------------------------------------------- |
| Small dataset (\<100 samples) | 2-3                | Fewer bins = more samples per category               |
| Medium dataset (100-1000)     | 3-5                | Balance between specificity and data sufficiency     |
| Large dataset (>1000 samples) | 5-10               | Can support fine-grained frequency categories        |
| Narrow-band events            | 4-6                | More bins to distinguish small frequency differences |
| Broad-band events             | 2-3                | Fewer bins since events span wide ranges             |
| High frequency diversity      | 5-8                | Capture different frequency regions                  |
| Low frequency diversity       | 2-3                | Avoid over-splitting similar events                  |

### Tuning Strategy

```bash
# Grid search over n_bins
for n_bins in 2 3 4 5 6; do
  echo "Testing n_bins=$n_bins"

  # Convert with specific n_bins
  uv run -m rf_detr_finetuning convert audio-to-coco \
    --input_dir /path/to/audio \
    --output_dir data/audio_coco_bins_${n_bins} \
    --auto_infer_bins $n_bins

  # Train RF-DETR
  uv run -m rf_detr_finetuning train \
    --config config/audio_detection.yaml \
    --dataset data/audio_coco_bins_${n_bins} \
    --output output_bins_${n_bins}

  # Evaluate and log results
  # Compare mAP across different n_bins
done
```

### Validation Metrics

Monitor these to find optimal n_bins:

- **Samples per category**: Should be >10 for reliable training
- **mAP**: Higher is better, but check for overfitting
- **Per-category AP**: Ensure all bins perform well, not just average
- **Confusion matrix**: Look for confusions between adjacent frequency bins

```python
# Check category distribution after conversion
import json
from collections import Counter

with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

for cat_id, count in cat_counts.most_common():
    print(f"{id_to_name[cat_id]}: {count} samples")
    if count < 10:
        print(f"  ⚠️  WARNING: <10 samples, consider reducing n_bins")
```

## Frequency Band Classification

The decoded frequency info includes automatic band classification:

| Band         | Hz Range      | Typical Sources                      |
| ------------ | ------------- | ------------------------------------ |
| `infrasonic` | < 100 Hz      | Earthquakes, explosions, large ships |
| `very_low`   | 100-500 Hz    | Heavy machinery, small vessels       |
| `low`        | 500-2000 Hz   | Marine mammals, engines              |
| `mid`        | 2000-5000 Hz  | Dolphins, fish, some sonars          |
| `high`       | 5000-10000 Hz | Porpoises, high-freq sonars          |
| `very_high`  | > 10000 Hz    | Echolocators, ice noise, rain        |

## CLI Examples

```bash
# Basic conversion with auto-bins (ALWAYS set target_sr!)
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --auto_infer_bins 3 \
  --target_sr 22050

# With custom spectrogram settings
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --auto_infer_bins 5 \
  --n_mels 256 \
  --hop_length 256 \
  --target_sr 48000  # High sample rate for ultrasound

# Manual bins (old way, still supported)
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --frequency_bins "0,500,low;500,5000,mid;5000,22050,high" \
  --target_sr 22050  # Must match fmax in bins!
```

## Benefits

1. **Data-driven categories**: Bins adapt to your specific dataset
2. **Frequency-aware classification**: Model learns frequency-dependent patterns
3. **Interpretable features**: Hz/time values are physically meaningful
4. **Flexible**: Works with any spectrogram resolution
5. **Automatic**: No manual tuning of frequency thresholds

## Next Steps

1. ✅ Convert your audio dataset with `auto_infer_bins`
2. ✅ Check the `frequency_info` in COCO annotations
3. 🔨 Create a custom RF-DETR classifier that uses frequency features
4. 🔨 Train and evaluate frequency-aware vs. baseline models
5. 🔨 Analyze performance by frequency band

## ⚠️ CRITICAL: Handling Different Sample Rates

**Problem**: If your dataset has files with different sample rates (e.g., 16kHz, 44.1kHz, 48kHz), **pixel coordinates will map to different Hz values!**

### Why This Breaks Everything

| File            | Original SR | Nyquist (fmax) | Pixel y=50 means |
| --------------- | ----------- | -------------- | ---------------- |
| earthquake.flac | 16 kHz      | 8 kHz          | ~6400 Hz         |
| sonar.wav       | 44.1 kHz    | 22.05 kHz      | ~17600 Hz        |
| whale.flac      | 48 kHz      | 24 kHz         | ~19200 Hz        |

**Same pixel coordinate = different frequencies!** This causes:

- ❌ Inconsistent frequency bins (bins based on Hz, but pixels differ)
- ❌ Broken FrequencyMapper (different fmax per file)
- ❌ Model can't learn (same pattern has different meanings)

### Solution: ALWAYS Set target_sr

```python
from rf_detr_finetuning import convert_audio_to_coco, SpectrogramConfig

# ✅ CORRECT: Resample all files to 22.05 kHz
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=22050,  # CRITICAL: Fixed SR for all files!
        n_mels=128,
    ),
    auto_infer_bins=3,
)

# ❌ WRONG: Keeps original SR (breaks with mixed SRs)
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=None,  # BAD: Each file uses its own SR!
    ),
)
```

### Choosing target_sr

| Use Case                            | Recommended SR | Reasoning                                      |
| ----------------------------------- | -------------- | ---------------------------------------------- |
| Speech/music                        | 22050 Hz       | Captures up to 11 kHz (enough for human voice) |
| Low-frequency (seismic, infrasound) | 8000-16000 Hz  | Saves memory, faster processing                |
| High-fidelity (bats, ultrasound)    | 48000+ Hz      | Preserves high-frequency content               |
| General audio                       | 22050 Hz       | Good default, balances quality and size        |

### Verification

```bash
# Check if all files in dataset have consistent SR
for f in /path/to/audio/*.flac; do
  soxi -r "$f"  # Show sample rate
done | sort -u  # Unique SRs

# If you see multiple SRs → MUST set target_sr!
```

```python
# Verify consistent fmax in converted COCO dataset
import json

with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

fmax_values = set(
    img["audio_metadata"]["frequency_info"]["hz_max"] for img in coco["images"]
)

if len(fmax_values) > 1:
    print(f"⚠️  WARNING: Multiple fmax values: {fmax_values}")
    print("Dataset has mixed sample rates! Check target_sr setting.")
else:
    print(f"✅ All spectrograms have consistent fmax: {fmax_values.pop():.1f} Hz")
```

## Next Steps

1. ✅ Convert your audio dataset with `auto_infer_bins`
2. ✅ Check the `frequency_info` in COCO annotations
3. 🔨 Create a custom RF-DETR classifier that uses frequency features
4. 🔨 Train and evaluate frequency-aware vs. baseline models
5. 🔨 Analyze performance by frequency band

## See Also

- [AUDIO_TO_COCO_GUIDE.md](AUDIO_TO_COCO_GUIDE.md) - Main conversion guide
- [frequency_aware_classification.py](../examples/frequency_aware_classification.py) - Example code
