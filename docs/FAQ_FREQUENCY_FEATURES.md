# FAQ: Frequency-Aware Classification

## Q1: Why use log_hz instead of raw Hz values?

**Short answer**: Human perception + numerical stability + consistency with mel scale.

### Detailed Explanation

Frequency features use **logarithmic scale** (`log10(Hz)`) for four critical reasons:

#### 1. Human Perception is Logarithmic

We perceive frequency in **octaves** (doubling), not linear steps:

- 100 → 200 Hz sounds like the same jump as 1000 → 2000 Hz
- Musical notes are equally spaced in log frequency (A4=440 Hz, A5=880 Hz, A6=1760 Hz)

```python
# Linear scale: very different distances
hz1 = 200 - 100  # 100 Hz difference
hz2 = 2000 - 1000  # 1000 Hz difference → 10x larger!

# Log scale: same perceptual distance
log_hz1 = log10(200) - log10(100)  # 0.301
log_hz2 = log10(2000) - log10(1000)  # 0.301 → same!
```

#### 2. Matches Mel Scale

Spectrograms already use **mel scale**, which is quasi-logarithmic:

$$\\text{mel} = 2595 \\times \\log\_{10}\\left(1 + \\frac{\\text{Hz}}{700}\\right)$$

Using `log10(Hz)` for features maintains consistency with the spectrogram representation.

#### 3. Numerical Stability

Neural networks struggle with **huge value ranges**:

| Frequency | Raw Hz | log10(Hz) | Normalized for NN |
| --------- | ------ | --------- | ----------------- |
| 20 Hz     | 20     | 1.30      | ✅ Good           |
| 1000 Hz   | 1000   | 3.00      | ✅ Good           |
| 20000 Hz  | 20000  | 4.30      | ✅ Good           |

Without log:

- Low frequencies: 20-200 Hz (tiny values, get lost in gradients)
- High frequencies: 5000-20000 Hz (huge values, dominate loss)
- Range: 20-20000 → 1000:1 ratio

With log:

- All frequencies: 1.3-4.3 (compact range)
- Range: 1.3-4.3 → ~3:1 ratio
- Equal importance across spectrum

#### 4. Better Feature Distribution

Linear Hz features would be **biased toward high frequencies**:

```python
# Example: 10 frequency bins
# Linear spacing (0-20000 Hz, step=2000)
bins_linear = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
# Problem: Most audio events happen below 5000 Hz → only 2 bins cover them!

# Log spacing (gives more resolution where it matters)
bins_log = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
# Better: More bins for low frequencies where most events occur
```

### Example Code

```python
import math
import numpy as np

# Extract frequency features with log scale
freq_info = {
    "hz_center": 1000.0,
    "bandwidth_hz": 500.0,
}

features = {
    # Log Hz: 3.0 (manageable for NN)
    "log_hz_center": math.log10(freq_info["hz_center"]),
    # Log bandwidth: 2.7 (also manageable)
    "log_bandwidth": math.log10(freq_info["bandwidth_hz"]),
    # Alternative: normalize if you prefer linear
    "hz_center_norm": freq_info["hz_center"] / 10000.0,  # 0.1
}

# Neural network can use these directly:
# combined_features = concat([visual_features, log_hz_center, log_bandwidth])
```

______________________________________________________________________

## Q2: How should n_bins be tuned as a hyperparameter?

**Short answer**: Grid search 2-6 bins, balance samples-per-category vs frequency specificity.

### n_bins is a Critical Hyperparameter

The number of frequency bins controls the trade-off between:

- **Fine-grained classification** (more bins = more specific categories)
- **Sample efficiency** (fewer bins = more samples per category)

### Decision Matrix

| Dataset Characteristic               | Recommended n_bins | Reasoning                          |
| ------------------------------------ | ------------------ | ---------------------------------- |
| **Small dataset** (\<100 samples)    | 2-3                | Need ≥10 samples per category      |
| **Medium dataset** (100-1000)        | 3-5                | Balance specificity and data       |
| **Large dataset** (>1000)            | 5-10               | Can afford fine-grained bins       |
| **Narrow-band events** (e.g., tones) | 4-6                | Small frequency differences matter |
| **Broad-band events** (e.g., noise)  | 2-3                | Events span wide ranges anyway     |
| **High frequency diversity**         | 5-8                | Capture different regions          |
| **Low frequency diversity**          | 2-3                | Avoid empty bins                   |

### Tuning Workflow

```bash
#!/bin/bash
# Grid search for optimal n_bins

for n_bins in 2 3 4 5 6; do
  echo "========================================="
  echo "Testing n_bins=$n_bins"
  echo "========================================="

  # 1. Convert dataset
  uv run -m rf_detr_finetuning convert audio-to-coco \
    --input_dir /path/to/audio \
    --output_dir data/audio_coco_bins_${n_bins} \
    --auto_infer_bins $n_bins \
    --target_sr 22050

  # 2. Check category distribution (must have ≥10 samples per category)
  python -c "
import json
from collections import Counter

with open('data/audio_coco_bins_${n_bins}/train/_annotations.coco.json') as f:
    coco = json.load(f)

cat_counts = Counter(ann['category_id'] for ann in coco['annotations'])
id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

print(f'n_bins={n_bins} category distribution:')
for cat_id, count in sorted(cat_counts.items()):
    status = '✅' if count >= 10 else '⚠️  LOW'
    print(f'  {id_to_name[cat_id]}: {count} samples {status}')
"

  # 3. Train RF-DETR
  uv run -m rf_detr_finetuning train \
    --config config/audio_detection.yaml \
    --dataset data/audio_coco_bins_${n_bins} \
    --output output_bins_${n_bins} \
    --epochs 50

  # 4. Evaluate
  echo "Results for n_bins=$n_bins:"
  cat output_bins_${n_bins}/results.json | grep -E "mAP|AP_"

done

# 5. Compare results
echo ""
echo "========================================="
echo "Summary: Compare all n_bins"
echo "========================================="
for n_bins in 2 3 4 5 6; do
  mAP=$(cat output_bins_${n_bins}/results.json | jq -r '.mAP // .bbox_mAP')
  echo "n_bins=$n_bins: mAP=$mAP"
done
```

### Validation Metrics

Monitor these to choose optimal `n_bins`:

1. **Samples per category**: Must be ≥10 for reliable training

   ```python
   # Check after conversion
   cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
   min_samples = min(cat_counts.values())

   if min_samples < 10:
       print(f"⚠️ WARNING: Min {min_samples} samples/category → reduce n_bins")
   ```

2. **mAP (mean Average Precision)**: Higher is better

   - Too few bins: Model can't distinguish frequency-specific patterns
   - Too many bins: Not enough samples, overfitting, low mAP

3. **Per-category AP**: Check if all bins perform well

   ```bash
   # Look for categories with very low AP
   cat output/results.json | jq '.per_category_AP'
   # If some bins have AP<0.1 → too many bins for this dataset
   ```

4. **Confusion matrix**: Check for confusions between adjacent bins

   - If "low" confused with "mid" → consider merging bins
   - If "low" confused with "high" → problem is not frequency-specific

### Example: Analyzing Results

```python
import json
from pathlib import Path

# Compare different n_bins
results = {}
for n_bins in [2, 3, 4, 5, 6]:
    with open(f"output_bins_{n_bins}/results.json") as f:
        data = json.load(f)
        results[n_bins] = {
            "mAP": data.get("bbox_mAP", data.get("mAP", 0)),
            "per_category": data.get("per_category_AP", {}),
        }

# Find optimal n_bins
best_n_bins = max(results.items(), key=lambda x: x[1]["mAP"])[0]
print(f"Optimal n_bins: {best_n_bins} (mAP={results[best_n_bins]['mAP']:.3f})")

# Check if it's stable (avoid overfitting)
for n_bins, res in sorted(results.items()):
    category_aps = list(res["per_category"].values())
    min_ap = min(category_aps) if category_aps else 0
    print(f"n_bins={n_bins}: mAP={res['mAP']:.3f}, min_category_AP={min_ap:.3f}")
```

### Rule of Thumb

Start with:

- **n_bins = 3** for most datasets (low/mid/high)
- **n_bins = 2** if dataset \<50 samples
- **n_bins = 5** if dataset >500 samples and high frequency diversity

Then tune ±1 based on validation performance.

______________________________________________________________________

## Q3: How does this work with different sample rates in the training set?

**New answer**: Use **normalized frequency mode** to handle mixed SRs gracefully! ⭐

### The Solution: Two Modes

#### Mode 1: Resampling (Simple)

Resample all files to a fixed SR, use absolute Hz:

```python
from rf_detr_finetuning import convert_audio_to_coco, SpectrogramConfig

# Resample everything to 22.05 kHz
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=22050,  # Fixed SR
        normalize_frequency=False,  # Absolute Hz
    ),
)
```

**Pros**: Simple, uses absolute Hz values
**Cons**: Resampling artifacts, may lose fidelity

#### Mode 2: Normalized Frequency (Flexible) ⭐ **Recommended**

Keep original SRs, use frequency normalized to Nyquist (0-1):

```python
# Keep original sample rates, use normalized frequency
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=None,  # Keep original SR
        normalize_frequency=True,  # 0-1 scale
    ),
)
```

**Pros**: No resampling, preserves fidelity, SR-independent features
**Cons**: Features are 0-1, not absolute Hz (but more robust!)

### How Normalized Frequency Works

Frequencies are expressed **relative to Nyquist** (SR/2):

$$\\text{normalized_freq} = \\frac{\\text{Hz}}{\\text{Nyquist}} = \\frac{\\text{Hz}}{\\text{SR}/2}$$

**Example**:

- 1000 Hz in 16 kHz file (Nyquist=8kHz) → normalized_freq = 1000/8000 = **0.125**
- 1000 Hz in 48 kHz file (Nyquist=24kHz) → normalized_freq = 1000/24000 = **0.042**

This is intentional! 1000 Hz at different Nyquists has different perceptual positions in the spectrum.

### COCO Metadata Includes Both

```json
{
  "audio_metadata": {
    "sample_rate": 16000,
    "nyquist_hz": 8000.0,
    "normalize_frequency": true,
    "frequency_info": {
      "hz_min": 500.0,
      "hz_max": 3000.0,
      "normalized_freq_min": 0.0625,
      "normalized_freq_max": 0.375,
      "normalized_freq_center": 0.21875
    }
  }
}
```

### Training with Normalized Features

```python
# Extract SR-independent features
freq_info = image["audio_metadata"]["frequency_info"]

features = {
    # Normalized (SR-independent)
    "norm_freq_center": freq_info["normalized_freq_center"],
    "norm_bandwidth": freq_info["normalized_freq_max"]
    - freq_info["normalized_freq_min"],
    "log_norm_freq": math.log10(max(freq_info["normalized_freq_center"], 0.001)),
    # Optional: Include SR as additional context
    "log_sr": math.log10(image["audio_metadata"]["sample_rate"]),
}
```

______________________________________________________________________

## Q3 (Old): What breaks with mixed SRs?

**This section preserved for historical context. See Mode 2 above for the solution.**

### The Problem: Pixel Coordinates Encode Hz Values

Spectrograms map:

- **Y-axis (pixels)** → Frequency (Hz via mel scale)
- **X-axis (pixels)** → Time (samples via hop_length)

The maximum frequency (Nyquist) is **SR / 2**:

- 16 kHz audio → fmax = 8 kHz
- 44.1 kHz audio → fmax = 22.05 kHz
- 48 kHz audio → fmax = 24 kHz

**Same pixel coordinate = different Hz values** for different sample rates!

### Concrete Example

Given 3 files with different sample rates:

| File            | SR       | Nyquist   | n_mels=128 | Pixel y=50 means                          |
| --------------- | -------- | --------- | ---------- | ----------------------------------------- |
| earthquake.flac | 16 kHz   | 8 kHz     | 128 bins   | mel_to_hz(mel_50, fmax=8k) ≈ **3200 Hz**  |
| whale.wav       | 44.1 kHz | 22.05 kHz | 128 bins   | mel_to_hz(mel_50, fmax=22k) ≈ **8800 Hz** |
| sonar.flac      | 48 kHz   | 24 kHz    | 128 bins   | mel_to_hz(mel_50, fmax=24k) ≈ **9600 Hz** |

**Same pixel (y=50) means 3200 Hz vs 8800 Hz vs 9600 Hz!**

### What Breaks

#### 1. Frequency Bins are Inconsistent

```python
# Auto-infer bins reads hz_min/hz_max from JSON
# But pixel coordinates differ per SR!

# File A (16 kHz): bbox y=10-60 → 0-4000 Hz → "low" bin
# File B (48 kHz): bbox y=10-60 → 0-10000 Hz → "mid" bin

# Same pixel bbox → different frequency bin → training chaos!
```

#### 2. FrequencyMapper is Incorrect

```python
# FrequencyMapper created per file with different fmax
freq_mapper_A = FrequencyMapper(n_mels=128, fmax=8000, sr=16000)
freq_mapper_B = FrequencyMapper(n_mels=128, fmax=24000, sr=48000)

# Same Hz value → different pixel for each file
hz = 5000
pixel_A = freq_mapper_A.hz_to_pixel(5000)  # ~90
pixel_B = freq_mapper_B.hz_to_pixel(5000)  # ~60

# Model can't learn: 5kHz event appears at y=90 or y=60 randomly!
```

#### 3. Model Can't Learn Patterns

```python
# Ship noise at 100 Hz:
# - In 16 kHz file: appears at pixel y=5
# - In 48 kHz file: appears at pixel y=3

# Model learns: "Ship noise is at y=5"
# Prediction on 48 kHz file: looks at y=5 → finds 150 Hz → wrong!
```

### Solution: ALWAYS Set target_sr

```python
from rf_detr_finetuning import convert_audio_to_coco, SpectrogramConfig

# ✅ CORRECT: Resample ALL files to 22.05 kHz
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=22050,  # ← CRITICAL: Fixed SR!
        n_mels=128,
        hop_length=512,
    ),
    auto_infer_bins=3,
)

# Now ALL spectrograms have:
# - Nyquist = 11025 Hz (consistent)
# - Pixel y=50 → always ~4400 Hz (consistent)
# - Frequency bins work correctly
# - Model can learn patterns
```

```bash
# CLI version
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --target_sr 22050 \  # ← CRITICAL!
  --auto_infer_bins 3
```

### Choosing target_sr

| Use Case                                               | Recommended SR | Reasoning                                                          |
| ------------------------------------------------------ | -------------- | ------------------------------------------------------------------ |
| **Speech/music**                                       | 22050 Hz       | Captures up to 11 kHz (human voice ~80-3000 Hz)                    |
| **Low-frequency** (seismic, infrasound)                | 8000-16000 Hz  | Saves memory, faster, sufficient for \<4 kHz events                |
| **High-fidelity** (bats, ultrasound, music production) | 48000 Hz       | Preserves up to 24 kHz (bat echolocation ~20-120 kHz)              |
| **Underwater acoustics**                               | 16000-48000 Hz | Depends on target species (whales: 10-10k Hz, dolphins: 5-150k Hz) |
| **General audio**                                      | 22050 Hz       | Good default, balances quality and compute                         |

**Rule**: Set `target_sr` to at least **2× the highest frequency** you care about (Nyquist theorem).

### Verification: Check if Dataset has Mixed SRs

```bash
# Linux/Mac: Check sample rates of all files
for f in /path/to/audio/*.{flac,wav}; do
  soxi -r "$f" 2>/dev/null || ffprobe -v error -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$f"
done | sort -u

# Output example:
# 16000
# 44100
# 48000
# ← Multiple SRs found → MUST set target_sr!
```

```python
# Python: Check sample rates in metadata
import json
from pathlib import Path

input_dir = Path("/path/to/audio")
sample_rates = set()

for json_path in input_dir.glob("*.json"):
    with open(json_path) as f:
        metadata = json.load(f)
        sample_rates.add(metadata.get("sample_rate"))

print(f"Sample rates in dataset: {sorted(sample_rates)}")

if len(sample_rates) > 1:
    print("⚠️  WARNING: Mixed sample rates detected!")
    print("   You MUST set target_sr in SpectrogramConfig")
    print(f"   Recommended: target_sr={max(sample_rates)}")
else:
    print(f"✅ All files have SR={sample_rates.pop()} Hz")
    print("   target_sr is optional (but still recommended for consistency)")
```

### Verification: Check Converted COCO Dataset

```python
# Verify all spectrograms have consistent fmax
import json

with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

fmax_values = set()
used_srs = set()

for img in coco["images"]:
    metadata = img["audio_metadata"]
    fmax_values.add(metadata["frequency_info"]["hz_max"])
    used_srs.add(metadata.get("used_sr", metadata.get("original_sr")))

print(f"Used sample rates: {sorted(used_srs)}")
print(f"Fmax values: {sorted(fmax_values)}")

if len(fmax_values) == 1:
    print(f"✅ All spectrograms have consistent fmax={fmax_values.pop():.1f} Hz")
else:
    print(f"⚠️  ERROR: Multiple fmax values found!")
    print("   This means spectrograms have different frequency ranges.")
    print("   Check that target_sr is set correctly.")
```

### What if I Have Good Reason to Use Mixed SRs?

**Don't.** But if you absolutely must:

1. **Resample to common SR** (best practice)
2. **Train separate models** for each SR (expensive)
3. **Add SR as input feature** (complex, not recommended):
   ```python
   # Encode SR as additional feature
   sr_embedding = {
       16000: [1, 0, 0],
       44100: [0, 1, 0],
       48000: [0, 0, 1],
   }
   combined_features = concat([visual_features, sr_embedding[sr], freq_features])
   ```

**99% of the time**: Just set `target_sr` to a sensible fixed value.

______________________________________________________________________

## Summary

| Question                | Answer                                           | Recommendation                            |
| ----------------------- | ------------------------------------------------ | ----------------------------------------- |
| **Why log_hz?**         | Human perception + stability + mel compatibility | Always use `log10(Hz)` for features       |
| **How to tune n_bins?** | Grid search 2-6, monitor samples/category        | Start with 3, tune based on dataset size  |
| **Different SRs?**      | Breaks everything!                               | **ALWAYS** set `target_sr` to fixed value |

## Quick Start Checklist

```bash
# 1. Check your dataset for mixed SRs
for f in /path/to/audio/*.flac; do soxi -r "$f"; done | sort -u

# 2. Convert with fixed SR and auto-bins
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --target_sr 22050 \        # ← CRITICAL!
  --auto_infer_bins 3 \      # ← Start here, tune later
  --n_mels 128

# 3. Verify consistency
python -c "
import json
with open('data/audio_coco/train/_annotations.coco.json') as f:
    coco = json.load(f)
fmax = {img['audio_metadata']['frequency_info']['hz_max'] for img in coco['images']}
print(f'Fmax values: {fmax}')
assert len(fmax) == 1, 'Mixed fmax detected!'
print('✅ All spectrograms consistent')
"

# 4. Train and tune n_bins via grid search
for n_bins in 2 3 4 5; do
  # ... (see Q2 for full script)
done
```
