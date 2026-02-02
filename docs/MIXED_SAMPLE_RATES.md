# Handling Mixed Sample Rates

This guide explains how to handle datasets with different sample rates gracefully using **normalized frequency mode**.

## The Two Modes

### Mode 1: Resampling (Simple)

**When to use**: All files should sound the same, just at different sample rates.

```python
from rf_detr_finetuning import convert_audio_to_coco, SpectrogramConfig

# Resample everything to 22.05 kHz
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=22050,  # Fixed SR for all files
        normalize_frequency=False,  # Use absolute Hz
    ),
)
```

**Pros**:

- Simple, uses absolute Hz values
- Easier to interpret (1000 Hz is 1000 Hz)
- Consistent spectrogram size if audio durations similar

**Cons**:

- Resampling can introduce artifacts
- Loses original fidelity (especially downsampling 48kHz → 22kHz)
- Extra computation time

______________________________________________________________________

### Mode 2: Normalized Frequency (Flexible) ⭐

**When to use**: Files have legitimately different Nyquist frequencies, or you want to preserve original fidelity.

```python
# Keep original sample rates, use normalized frequency
convert_audio_to_coco(
    input_dir="/path/to/audio",
    output_dir="data/audio_coco",
    spec_config=SpectrogramConfig(
        target_sr=None,  # Keep original SR
        normalize_frequency=True,  # Use 0-1 scale
    ),
)
```

**Pros**:

- ✅ No resampling artifacts
- ✅ Preserves original audio fidelity
- ✅ Works with any SR mix (16kHz, 44.1kHz, 48kHz, etc.)
- ✅ SR-independent features for model training

**Cons**:

- Features are normalized (0-1), not absolute Hz
- Need to store SR in metadata for interpretation

______________________________________________________________________

## How Normalized Frequency Works

Instead of absolute Hz values, frequencies are expressed **relative to Nyquist** (SR/2):

$$\\text{normalized_freq} = \\frac{\\text{Hz}}{\\text{Nyquist}} = \\frac{\\text{Hz}}{\\text{SR}/2}$$

### Example

| File            | SR       | Nyquist   | Event at 1000 Hz | Normalized Freq        |
| --------------- | -------- | --------- | ---------------- | ---------------------- |
| earthquake.flac | 16 kHz   | 8 kHz     | 1000 Hz          | 1000/8000 = **0.125**  |
| whale.wav       | 44.1 kHz | 22.05 kHz | 1000 Hz          | 1000/22050 = **0.045** |
| sonar.flac      | 48 kHz   | 24 kHz    | 1000 Hz          | 1000/24000 = **0.042** |

**Same Hz value → different normalized frequencies!** This is intentional:

- 1000 Hz in an 8 kHz Nyquist file is "mid-band" (0.125)
- 1000 Hz in a 24 kHz Nyquist file is "low-band" (0.042)

The **perceptual position** in the spectrum differs!

______________________________________________________________________

## COCO Metadata with Both Representations

When using `normalize_frequency=True`, the COCO JSON includes both:

```json
{
  "images": [{
    "audio_metadata": {
      "sample_rate": 16000,
      "nyquist_hz": 8000.0,
      "normalize_frequency": true,
      "frequency_info": {
        "hz_min": 500.0,
        "hz_max": 3000.0,
        "hz_center": 1750.0,
        "normalized_freq_min": 0.0625,
        "normalized_freq_max": 0.375,
        "normalized_freq_center": 0.21875,
        "sample_rate": 16000
      }
    }
  }]
}
```

______________________________________________________________________

## Model Training with Normalized Frequencies

### Extract SR-Independent Features

```python
import json
import math

with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

for ann in coco["annotations"]:
    image = next(img for img in coco["images"] if img["id"] == ann["image_id"])
    freq_info = image["audio_metadata"]["frequency_info"]
    sr = image["audio_metadata"]["sample_rate"]

    # Option 1: Use normalized frequencies (SR-independent)
    features_normalized = {
        "norm_freq_center": freq_info["normalized_freq_center"],
        "norm_freq_bandwidth": freq_info["normalized_freq_max"]
        - freq_info["normalized_freq_min"],
        "log_norm_freq": math.log10(max(freq_info["normalized_freq_center"], 0.001)),
    }

    # Option 2: Use absolute Hz (if you want) + SR as context
    features_absolute = {
        "log_hz_center": math.log10(freq_info["hz_center"]),
        "log_sr": math.log10(sr),  # Include SR for context
    }

    # Option 3: Hybrid - normalized freq + SR embedding
    sr_category = "low_sr" if sr < 20000 else "mid_sr" if sr < 40000 else "high_sr"
    features_hybrid = {
        "norm_freq_center": freq_info["normalized_freq_center"],
        "sr_category": sr_category,
    }
```

### Conditioning Model on Sample Rate

```python
# In your RF-DETR training code
class FrequencyAwareClassifier(nn.Module):
    def __init__(self, visual_dim=256, freq_dim=16, sr_embed_dim=8, num_classes=10):
        super().__init__()

        # Sample rate embedding
        self.sr_embedding = nn.Embedding(
            num_embeddings=10, embedding_dim=sr_embed_dim  # Discretize SR into bins
        )

        # Frequency feature encoder
        self.freq_encoder = nn.Linear(2, freq_dim)  # [norm_freq_center, norm_bandwidth]

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + freq_dim + sr_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, visual_features, norm_freq_center, norm_bandwidth, sample_rate):
        # Discretize SR into bins
        sr_bins = torch.clamp(
            (sample_rate // 4000).long(), 0, 9  # 0-4k→0, 4-8k→1, 8-12k→2, etc.
        )
        sr_embed = self.sr_embedding(sr_bins)

        # Encode frequency features
        freq_features = torch.stack([norm_freq_center, norm_bandwidth], dim=-1)
        freq_embed = self.freq_encoder(freq_features)

        # Combine and classify
        combined = torch.cat([visual_features, freq_embed, sr_embed], dim=-1)
        return self.classifier(combined)
```

______________________________________________________________________

## Frequency Bin Auto-Inference with Normalized Mode

When using `normalize_frequency=True`, auto-inferred bins work across different SRs:

```bash
# Files with mixed SRs: 16kHz, 44.1kHz, 48kHz
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --auto_infer_bins 3

# Output (using normalized frequencies):
# Inferred 3 frequency bins from 50 samples:
#   low: 0.02-0.15 (normalized)
#   mid: 0.15-0.40 (normalized)
#   high: 0.40-0.95 (normalized)
```

These bins are **SR-independent** - an event at normalized_freq=0.3 is always "mid" regardless of SR.

______________________________________________________________________

## Choosing the Right Mode

| Dataset Characteristic                                                       | Recommended Mode                     | Config                                       |
| ---------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| **All files same SR**                                                        | Either (resampling slightly simpler) | `target_sr=SR, normalize_frequency=False`    |
| **Mixed SRs, same content type** (e.g., all speech)                          | Resampling to common SR              | `target_sr=22050, normalize_frequency=False` |
| **Mixed SRs, different content** (e.g., earthquakes at 8kHz, music at 48kHz) | Normalized frequency                 | `target_sr=None, normalize_frequency=True`   |
| **High-fidelity preservation important**                                     | Normalized frequency                 | `target_sr=None, normalize_frequency=True`   |
| **Simplicity important**                                                     | Resampling                           | `target_sr=22050, normalize_frequency=False` |

______________________________________________________________________

## Example: Real-World Dataset

### Dataset Structure

```
audio/
├── earthquake_16khz.flac  (SR=16000, contains 10-200 Hz events)
├── whale_44khz.wav        (SR=44100, contains 50-15000 Hz calls)
├── music_48khz.flac       (SR=48000, full spectrum)
```

### Resampling Mode (Forces Everything to 22.05 kHz)

```python
# ⚠️ Problem: Earthquake loses detail (200 Hz Nyquist → 11025 Hz Nyquist)
#            Music loses high freq (24 kHz → 11 kHz)

config = SpectrogramConfig(target_sr=22050, normalize_frequency=False)
```

### Normalized Mode (Preserves Original Fidelity) ⭐

```python
# ✅ Each file keeps its appropriate SR and resolution
config = SpectrogramConfig(target_sr=None, normalize_frequency=True)

# earthquake_16khz.flac:
#   - Nyquist = 8 kHz (appropriate for 10-200 Hz events)
#   - 100 Hz event → normalized_freq = 100/8000 = 0.0125 ("very low")

# whale_44khz.wav:
#   - Nyquist = 22.05 kHz (captures up to 15 kHz calls)
#   - 5000 Hz call → normalized_freq = 5000/22050 = 0.227 ("low-mid")

# music_48khz.flac:
#   - Nyquist = 24 kHz (full fidelity)
#   - 12000 Hz cymbal → normalized_freq = 12000/24000 = 0.5 ("mid")
```

______________________________________________________________________

## Verification Script

```python
import json
from collections import Counter

# Check SR distribution in converted dataset
with open("data/audio_coco/train/_annotations.coco.json") as f:
    coco = json.load(f)

srs = Counter(img["audio_metadata"]["sample_rate"] for img in coco["images"])
print("Sample rate distribution:")
for sr, count in sorted(srs.items()):
    print(f"  {sr} Hz: {count} files")

# Check if normalize_frequency is enabled
normalize_mode = coco["images"][0]["audio_metadata"].get("normalize_frequency", False)
print(f"\nNormalized frequency mode: {normalize_mode}")

if normalize_mode:
    print("\n✅ Using SR-independent normalized frequencies")
    print("   Features: normalized_freq_min/max/center (0-1 scale)")
else:
    print("\n⚠️ Using absolute Hz values")
    print("   Ensure all files have same SR or features will be inconsistent!")
```

______________________________________________________________________

## Summary

| Mode              | target_sr         | normalize_frequency | Use Case                                              |
| ----------------- | ----------------- | ------------------- | ----------------------------------------------------- |
| **Resampling**    | Set (e.g., 22050) | False               | Same content type, prefer simplicity                  |
| **Normalized** ⭐ | None              | True                | Mixed SRs, preserve fidelity, SR-independent features |

**Recommendation**: Use **normalized mode** for maximum flexibility. It handles any SR mix gracefully and preserves original audio quality.
