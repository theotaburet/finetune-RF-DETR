# Audio Dataset to RF-DETR: Comprehensive Guide

This guide explains how to convert audio datasets with JSON metadata into COCO-format spectrogram datasets for training RF-DETR object detection models.

## Overview

The pipeline converts:

- **Audio files** (`.flac`, `.wav`, `.mp3`, `.ogg`) → **Mel spectrogram images** (`.png`)
- **JSON metadata** (frequency, annotation, duration) → **COCO bounding box annotations**

**Built on ezakodio** - All audio processing (loading, spectrogram generation, frequency conversions) uses the ezakodio library directly.

### Key Concept: Audio Events as Visual Objects

On a spectrogram:

- **X-axis** = Time (left to right)
- **Y-axis** = Frequency (low at bottom, high at top for mel scale)

Audio events become rectangular regions (bounding boxes) that RF-DETR can learn to detect.

## Quick Start

```bash
# Convert audio dataset to COCO format
uv run -m rf_detr_finetuning convert audio-to-coco \
  /path/to/audio/dataset \
  data/audio_coco \
  --n-mels 128 \
  --split-ratios "0.7,0.2,0.1"

# Train RF-DETR on the converted dataset
uv run -m rf_detr_finetuning train \
  --config config/audio_detection.yaml \
  --dataset data/audio_coco
```

## Input Format

Your audio dataset should have this structure:

```
input_directory/
├── 32e15fad-12be-45f9-997a-a9ac3f455905.flac
├── 32e15fad-12be-45f9-997a-a9ac3f455905.json
├── 7f2f01dc-f617-43a0-b18f-926b31953885.flac
├── 7f2f01dc-f617-43a0-b18f-926b31953885.json
└── ...
```

Each JSON metadata file should contain:

```json
{
  "uuid": "7f2f01dc-f617-43a0-b18f-926b31953885",
  "label_hierarchy": "geoacoustic",
  "annotation": "Earthquake",
  "duration": 15827,
  "hz_min": 0.0,
  "hz_max": 16205.0,
  "sample_rate": 44100,
  "channels": 2,
  "confidence": 0.29,
  "other_labels": [],
  "custom": {}
}
```

### Required Fields

| Field         | Type   | Description                     |
| ------------- | ------ | ------------------------------- |
| `uuid`        | string | Unique identifier               |
| `annotation`  | string | Class label for the audio event |
| `duration`    | float  | Duration in milliseconds        |
| `hz_min`      | float  | Minimum frequency of event (Hz) |
| `hz_max`      | float  | Maximum frequency of event (Hz) |
| `sample_rate` | int    | Audio sample rate               |

### Optional Fields

| Field             | Type   | Description               |
| ----------------- | ------ | ------------------------- |
| `label_hierarchy` | string | Category grouping         |
| `confidence`      | float  | Labeling confidence (0-1) |
| `channels`        | int    | Number of audio channels  |
| `other_labels`    | list   | Additional labels         |
| `custom`          | dict   | Custom metadata           |

## Output Format

The converter produces a COCO-format dataset:

```
output_directory/
├── train/
│   ├── 32e15fad-12be-45f9-997a-a9ac3f455905.png
│   ├── 7f2f01dc-f617-43a0-b18f-926b31953885.png
│   └── _annotations.coco.json
├── valid/
│   ├── ...
│   └── _annotations.coco.json
└── test/
    ├── ...
    └── _annotations.coco.json
```

### COCO Annotation Structure

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "7f2f01dc-f617-43a0-b18f-926b31953885.png",
      "width": 619,
      "height": 128,
      "audio_metadata": {
        "original_sr": 44100,
        "n_frames": 619,
        "n_mels": 128,
        "uuid": "7f2f01dc-f617-43a0-b18f-926b31953885"
      }
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [0, 10, 619, 100],
      "area": 61900,
      "attributes": {
        "hz_min": 0.0,
        "hz_max": 16205.0,
        "time_start_ms": 0.0,
        "time_end_ms": 15827.0
      }
    }
  ],
  "categories": [
    {"id": 0, "name": "Earthquake", "supercategory": "audio_event"}
  ]
}
```

## Spectrogram Configuration

### Parameters

| Parameter    | Default | Description                                                                  |
| ------------ | ------- | ---------------------------------------------------------------------------- |
| `n_fft`      | 2048    | FFT window size. Larger = better frequency resolution, worse time resolution |
| `hop_length` | 512     | Samples between frames. Smaller = more frames, larger image width            |
| `n_mels`     | 128     | Number of mel bins. This is the image height                                 |
| `fmin`       | 0.0     | Minimum frequency (Hz)                                                       |
| `fmax`       | sr/2    | Maximum frequency (Hz). Nyquist by default                                   |
| `target_sr`  | None    | Resample audio to this rate                                                  |

### Resolution Trade-offs

```
Image Width  = audio_duration_samples / hop_length
Image Height = n_mels

Time Resolution = hop_length / sample_rate (seconds per pixel)
Freq Resolution ≈ (fmax - fmin) / n_mels (Hz per pixel, approximate for mel scale)
```

**Example**: 15 seconds of audio at 44100 Hz with hop_length=512:

- Width = (15 * 44100) / 512 ≈ 1292 pixels

## Frequency-Aware Categories

A key feature is **frequency-aware category splitting**. The same annotation can become different categories based on frequency range:

```bash
uv run -m rf_detr_finetuning convert audio-to-coco \
  input_dir output_dir \
  --frequency-bins "0,500,low_freq;500,5000,mid_freq;5000,22050,high_freq"
```

This creates categories like:

- `ship_noise_low_freq` — horizontal line at bottom of spectrogram
- `sonar_mid_freq` — horizontal line in middle
- `whale_call_high_freq` — pattern in upper region

### Why This Matters

Different acoustic phenomena occur at different frequency bands:

| Frequency Range | Typical Sources                    |
| --------------- | ---------------------------------- |
| 0-500 Hz        | Ship noise, machinery, earthquakes |
| 500-5000 Hz     | Marine mammals, some sonars        |
| 5000-22050 Hz   | High-frequency sonars, rain, ice   |

By encoding frequency information into categories, the model learns frequency-specific patterns.

## Python API

### Basic Usage

```python
from rf_detr_finetuning.audio_to_coco import (
    convert_audio_to_coco,
    SpectrogramConfig,
)

# Simple conversion
convert_audio_to_coco(
    input_dir="/mnt/d/SoundBase/Sounds/SDK/geoacoustic",
    output_dir="data/geoacoustic_coco",
)

# With custom spectrogram settings
config = SpectrogramConfig(
    n_fft=4096,  # Higher frequency resolution
    hop_length=256,  # Higher time resolution
    n_mels=256,  # Taller images
    fmin=20.0,  # Skip very low frequencies
    fmax=8000.0,  # Limit to 8kHz
)

convert_audio_to_coco(
    input_dir="/mnt/d/SoundBase/Sounds/SDK/geoacoustic",
    output_dir="data/geoacoustic_coco",
    spec_config=config,
    frequency_bins=[
        (0, 500, "low"),
        (500, 2000, "mid"),
        (2000, 8000, "high"),
    ],
)
```

### Processing Single Files

```python
from rf_detr_finetuning.audio_to_coco import (
    AudioMetadata,
    SpectrogramConfig,
    CategoryRegistry,
    process_audio_file,
)
from pathlib import Path

# Load metadata
metadata = AudioMetadata.from_json("path/to/audio.json")

# Process
config = SpectrogramConfig()
registry = CategoryRegistry()

img, bbox, extra = process_audio_file(
    Path("path/to/audio.flac"),
    metadata,
    config,
    registry,
)

# Save image
img.save("output/spectrogram.png")

# Get bbox info
print(f"Category: {bbox.category_name}")
print(f"Frequency range: {bbox.hz_min} - {bbox.hz_max} Hz")
```

### Custom Frequency Mapping

```python
from rf_detr_finetuning.audio_to_coco import FrequencyMapper

# Uses ezakodio's hz_to_mel and mel_to_hz internally

# Create mapper for a 128-mel spectrogram, 0-22050 Hz range
mapper = FrequencyMapper(
    n_mels=128,
    fmin=0.0,
    fmax=22050.0,
    sample_rate=44100,
)

# Convert frequency to pixel row
pixel_row = mapper.hz_to_pixel(1000.0)  # 1kHz -> pixel Y
print(f"1000 Hz is at pixel row {pixel_row:.1f}")

# Convert pixel back to frequency
freq = mapper.pixel_to_hz(64)  # Middle row -> Hz
print(f"Pixel row 64 corresponds to {freq:.1f} Hz")
```

### Direct ezakodio Usage

You can also use ezakodio directly for audio processing:

```python
from ezakodio.io import load_audio
from ezakodio.dsp import mel_spectrogram
from ezakodio import hz_to_mel, mel_to_hz

# Load audio
audio, sr = load_audio("audio.flac", mono=True, device="cpu")

# Compute spectrogram
spec = mel_spectrogram(
    audio,
    sample_rate=sr,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    log_scale=True,
)

# Frequency conversions
mel_val = hz_to_mel(1000.0)  # 1kHz in mel scale
hz_val = mel_to_hz(mel_val)  # Back to Hz
```

## Extending for Segment-Level Annotations

The current implementation treats each audio file as having one annotation covering the full duration. For segment-level annotations (multiple events in one file), extend the `process_audio_file` function:

```python
from rf_detr_finetuning.audio_to_coco import (
    BoundingBox,
    FrequencyMapper,
    TimeMapper,
)


def process_segments(
    audio_path: Path,
    segments: list[dict],  # List of {start_ms, end_ms, hz_min, hz_max, label}
    mel_spec: np.ndarray,
    sample_rate: int,
    spec_config: SpectrogramConfig,
    category_registry: CategoryRegistry,
) -> list[BoundingBox]:
    """Process multiple segments from one audio file."""

    n_mels, n_frames = mel_spec.shape
    fmax = spec_config.fmax or sample_rate / 2

    freq_mapper = FrequencyMapper(n_mels, spec_config.fmin, fmax, sample_rate)
    total_duration_ms = (n_frames * spec_config.hop_length / sample_rate) * 1000
    time_mapper = TimeMapper(
        total_duration_ms, n_frames, sample_rate, spec_config.hop_length
    )

    boxes = []
    for seg in segments:
        # Time -> X pixels
        x_left = time_mapper.ms_to_pixel(seg["start_ms"])
        x_right = time_mapper.ms_to_pixel(seg["end_ms"])

        # Frequency -> Y pixels (inverted: high freq = top = low Y)
        y_top = freq_mapper.hz_to_pixel(seg["hz_max"])
        y_bottom = freq_mapper.hz_to_pixel(seg["hz_min"])

        cat_id, cat_name = category_registry.get_category_with_frequency(
            seg["label"], seg["hz_min"], seg["hz_max"]
        )

        boxes.append(
            BoundingBox(
                x=x_left,
                y=y_top,
                width=x_right - x_left,
                height=y_bottom - y_top,
                category_id=cat_id,
                category_name=cat_name,
                hz_min=seg["hz_min"],
                hz_max=seg["hz_max"],
                time_start_ms=seg["start_ms"],
                time_end_ms=seg["end_ms"],
            )
        )

    return boxes
```

## Training Configuration

Create a config file for audio spectrogram training:

```yaml
# config/audio_detection.yaml
epochs: 50
lr: 1e-4
batch_size: 8
grad_accum_steps: 2

# For spectrograms, you might want different augmentations
# than typical image datasets
```

## Best Practices

### 1. Choose Appropriate Spectrogram Resolution

- **Detection of long events** (ship passages): Use larger `hop_length` (1024+)
- **Detection of short events** (clicks, pulses): Use smaller `hop_length` (128-256)
- **Broadband events**: Use fewer `n_mels` (64-128)
- **Narrowband events** (tones, calls): Use more `n_mels` (256+)

### 2. Normalize Sample Rates

Your dataset can have mixed sample rates! Two modes are supported:

**Mode 1: Resampling (simple)**

```python
config = SpectrogramConfig(
    target_sr=22050,  # Resample all files to 22.05 kHz
    normalize_frequency=False,  # Use absolute Hz values
)
```

**Mode 2: Normalized Frequency (flexible, recommended)** ⭐

```python
config = SpectrogramConfig(
    target_sr=None,  # Keep original SRs
    normalize_frequency=True,  # Use freq normalized to Nyquist (0-1)
)
```

Normalized mode lets you handle files with different sample rates (16kHz, 44.1kHz, 48kHz) gracefully without resampling artifacts. See [MIXED_SAMPLE_RATES.md](MIXED_SAMPLE_RATES.md) for details.

### 3. Use Frequency Bins for Underwater Acoustics

```python
# Common underwater acoustic sources
frequency_bins = [
    (0, 100, "very_low"),  # Seismic, large ships
    (100, 1000, "low"),  # Small vessels, machinery
    (1000, 10000, "mid"),  # Dolphins, small odontocetes
    (10000, 48000, "high"),  # Porpoises, echosounders
]
```

### 4. Monitor Spectrogram Quality

After conversion, visually inspect some spectrograms:

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("data/audio_coco/train/sample.png")
plt.imshow(img)
plt.xlabel("Time (frames)")
plt.ylabel("Mel frequency bin")
plt.title("Spectrogram Preview")
plt.show()
```

## Troubleshooting

### Import Error: ezakodio not found

```bash
uv add git+https://github.com/ezako/ezakodio.git
```

### Memory Issues with Large Audio Files

For very long audio files, use ezakodio's streaming mode:

```python
from ezakodio.dsp import streaming_mel_spectrogram_from_file, StreamingConfig

config = StreamingConfig(
    chunk_duration_s=60.0,  # Process 60 seconds at a time
    device="cpu",
)

result = streaming_mel_spectrogram_from_file(
    "very_long_recording.wav",
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    config=config,
)
spec = result.spectrogram
```

### Bounding Boxes Too Small/Large

Adjust `n_mels` and `hop_length`:

- Larger `n_mels` = taller bounding boxes (more pixels in Y)
- Smaller `hop_length` = wider bounding boxes (more pixels in X)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     audio_to_coco.py                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ AudioMetadata   │    │ SpectrogramConfig│                   │
│  │ (from JSON)     │    │ (n_fft, n_mels..)│                   │
│  └────────┬────────┘    └────────┬─────────┘                   │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────┐                   │
│  │        process_audio_file()             │                   │
│  │                                         │                   │
│  │  Uses ezakodio directly:                │                   │
│  │  1. ezakodio.io.load_audio()            │                   │
│  │  2. ezakodio.dsp.mel_spectrogram()      │                   │
│  │  3. spectrogram_to_image()              │                   │
│  │  4. FrequencyMapper (uses hz_to_mel)    │                   │
│  │  5. Create BoundingBox                  │                   │
│  └─────────────────────────────────────────┘                   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │         COCODataset                     │                   │
│  │  - add_image()                          │                   │
│  │  - add_annotation()                     │                   │
│  │  - save() -> _annotations.coco.json     │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                           ┌──────────────────────┐
                           │      ezakodio        │
                           ├──────────────────────┤
                           │ io.load_audio()      │
                           │ dsp.mel_spectrogram()│
                           │ hz_to_mel()          │
                           │ mel_to_hz()          │
                           └──────────────────────┘
```

## Next Steps

1. **Convert your dataset**:

   ```bash
   uv run -m rf_detr_finetuning convert audio-to-coco \
     /mnt/d/SoundBase/Sounds/SDK/geoacoustic \
     data/geoacoustic_coco
   ```

2. **Inspect the output** to verify spectrograms look correct

3. **Create a training config** for your specific detection task

4. **Train RF-DETR** on the converted dataset

5. **Evaluate** and iterate on spectrogram parameters
