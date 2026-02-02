# Quick Reference: Handling Mixed Sample Rates

## TL;DR

**Use normalized frequency mode** to handle datasets with mixed sample rates (16kHz, 44.1kHz, 48kHz, etc.) without any issues:

```bash
# Default mode (recommended) - handles any SR mix!
uv run -m rf_detr_finetuning convert audio-to-coco \
  --input_dir /path/to/audio \
  --output_dir data/audio_coco \
  --auto_infer_bins 3

# This uses:
# - target_sr=None (keep original SRs)
# - normalize_frequency=True (use 0-1 scale)
```

## Two Modes Comparison

| Feature                 | Resampling Mode                 | Normalized Mode ⭐             |
| ----------------------- | ------------------------------- | ------------------------------ |
| **target_sr**           | Set (e.g., 22050)               | None                           |
| **normalize_frequency** | False                           | True                           |
| **Frequency values**    | Absolute Hz                     | 0-1 (relative to Nyquist)      |
| **Mixed SRs**           | ⚠️ Must resample all to same SR | ✅ Works with any SR mix       |
| **Audio quality**       | ⚠️ Resampling artifacts         | ✅ Preserves original fidelity |
| **Simplicity**          | ✅ Simpler (absolute Hz)        | Slightly more complex          |
| **Feature robustness**  | ⚠️ SR-dependent                 | ✅ SR-independent              |

## When to Use Which

### Use Resampling Mode If:

- All files are same content type (e.g., all speech)
- You prefer working with absolute Hz values
- Simplicity is more important than preserving original fidelity

```python
spec_config = SpectrogramConfig(
    target_sr=22050,
    normalize_frequency=False,
)
```

### Use Normalized Mode If: ⭐

- Files have different sample rates
- Preserving original audio quality is important
- You want SR-independent features for robust model training
- You're not sure (this is the safe default!)

```python
spec_config = SpectrogramConfig(
    target_sr=None,
    normalize_frequency=True,
)
```

## Example: Same Event, Different SRs

### Dataset

```
audio/
├── earthquake_16khz.flac  (100 Hz tone, SR=16000)
└── earthquake_48khz.flac  (100 Hz tone, SR=48000)
```

### Resampling Mode (Forces to 22.05 kHz)

```python
# Both files → 22.05 kHz
# 100 Hz → 100 Hz (absolute)
# Same pixel coordinate, same meaning ✅
# But: resampling artifacts, lost fidelity ⚠️
```

### Normalized Mode (Keeps Original SRs) ⭐

```python
# earthquake_16khz.flac:
#   Nyquist = 8 kHz
#   100 Hz → normalized_freq = 100/8000 = 0.0125

# earthquake_48khz.flac:
#   Nyquist = 24 kHz
#   100 Hz → normalized_freq = 100/24000 = 0.0042

# Different normalized values, but model learns:
# "100 Hz is very low frequency relative to Nyquist"
# Features are SR-independent ✅
# Original fidelity preserved ✅
```

## Metadata in COCO JSON

Both modes store all necessary information:

```json
{
  "audio_metadata": {
    "original_sr": 44100,
    "sample_rate": 44100,
    "nyquist_hz": 22050.0,
    "normalize_frequency": true,
    "frequency_info": {
      "hz_min": 1000.0,
      "hz_max": 5000.0,
      "hz_center": 3000.0,
      "normalized_freq_min": 0.0454,
      "normalized_freq_max": 0.2268,
      "normalized_freq_center": 0.1361
    }
  }
}
```

## Model Training

### Extract Features for Normalized Mode

```python
freq_info = metadata["frequency_info"]

# Use normalized frequencies (SR-independent)
features = {
    "norm_freq_center": freq_info["normalized_freq_center"],  # 0-1
    "norm_bandwidth": freq_info["normalized_freq_max"]
    - freq_info["normalized_freq_min"],
    "log_norm_freq": math.log10(max(freq_info["normalized_freq_center"], 0.001)),
}

# Optional: add SR as context
features["log_sr"] = math.log10(metadata["sample_rate"])
```

## Default Behavior

As of the latest version, **normalized mode is the default**:

```python
# Default config
SpectrogramConfig()
# → target_sr=None, normalize_frequency=True

# Handles mixed SRs automatically!
```

## See Also

- [MIXED_SAMPLE_RATES.md](MIXED_SAMPLE_RATES.md) - Detailed explanation
- [FAQ_FREQUENCY_FEATURES.md](FAQ_FREQUENCY_FEATURES.md#q3) - FAQ entry
- [AUDIO_TO_COCO_GUIDE.md](AUDIO_TO_COCO_GUIDE.md) - Main guide
