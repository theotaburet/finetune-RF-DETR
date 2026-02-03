# Audio Chunking System Documentation

## Overview

The audio chunking system converts variable-length audio files (e.g., 52 seconds of whale calls, 50ms gunshots) into fixed-size spectrograms (640×640 pixels) for training RF-DETR object detection models. It handles temporal event bounding boxes, frequency-based categories, and padding strategies to ensure high-quality training data.

## Key Features

- ✅ **Time-based FFT**: Sample-rate independent configuration
- ✅ **Sliding windows**: Configurable overlap to capture events at boundaries
- ✅ **Mel-scale mapping**: Frequency coordinates aligned to mel spectrogram
- ✅ **Padding awareness**: Bboxes constrained to actual content (no black rectangles)
- ✅ **Smart dropping**: Automatically drops chunks with excessive padding
- ✅ **Random augmentation**: Randomizes padding position for very short audio
- ✅ **Visual debugging**: Draws bboxes on spectrograms for verification

## Architecture

### Core Components

```
AudioChunker (main orchestrator)
    ↓
    ├── compute_chunk_boundaries() → [(start_ms, end_ms), ...]
    ├── extract_audio_chunk() → padded audio segment
    ├── _compute_spectrogram() → mel spectrogram
    ├── align_bbox_to_chunk() → bbox pixel coordinates
    ├── resize_spectrogram() → 640×640 with padding
    └── AudioChunk (output) → {spectrogram, bboxes, metadata}
```

### Data Classes

**TimeBasedFFTConfig**: Sample-rate independent FFT parameters

```python
@dataclass
class TimeBasedFFTConfig:
    fft_ms: float = 25.0  # FFT window duration (ms)
    hop_ms: float = 10.0  # Time per pixel (ms)
    n_mels: int = 128  # Mel bands (height before resize)
```

**ChunkConfig**: Chunking behavior

```python
@dataclass
class ChunkConfig:
    window_duration_ms: float = 6400.0  # Auto-computed from target_size × hop_ms
    overlap_ms: float = 1280.0  # Sliding window overlap
    min_overlap_with_event_ratio: float = 0.3  # Bbox inclusion threshold
    target_width: int = 640  # Target width (auto-computed)
    target_height: int = 640  # Target height (auto-computed)
    padding_mode: str = "zero"  # "zero", "repeat", or "reflect"
    min_chunk_content_ratio: float = 0.5  # Drop threshold for padded chunks
    random_pad_position: bool = True  # Randomize short audio position
```

**AudioChunk**: Processed chunk

```python
@dataclass
class AudioChunk:
    chunk_index: int  # Chunk number (0-indexed)
    start_ms: float  # Start time in original audio
    end_ms: float  # End time in original audio
    spectrogram: np.ndarray  # uint8 array (H, W) or (H, W, 3)
    bboxes: list[ChunkBbox]  # Aligned bboxes
    source_uuid: str  # Source file identifier
    sample_rate: int  # Audio sample rate
    is_padded: bool  # Whether audio was padded
    padding_amount_ms: float  # Amount of padding added
```

**ChunkBbox**: Aligned bounding box

```python
@dataclass
class ChunkBbox:
    x: float  # Left edge (pixels, COCO format)
    y: float  # Top edge (pixels, COCO format)
    width: float  # Width (pixels)
    height: float  # Height (pixels)
    category: str  # Label (e.g., "odontoceti")
    category_id: int  # Numeric category ID
    overlap_ratio: float  # Event overlap with chunk (0.0-1.0)
    hz_min: float  # Original frequency min (Hz)
    hz_max: float  # Original frequency max (Hz)
    original_time_start_ms: float  # Original start time
    original_time_end_ms: float  # Original end time
```

## Configuration

### Simplified YAML Format

```yaml
# config/audio_chunking.yaml
fft:
  hop_ms: 10.0   # Time per pixel (ms)
  fft_ms: 25.0   # FFT window (ms)
  n_mels: 128    # Mel bands

chunking:
  target_size: 640  # Square size → auto-computes width, height, window_duration_ms
  overlap_ms: 1280.0  # 20% overlap
  min_overlap_with_event_ratio: 0.3  # Include bbox if ≥30% in chunk
  padding_mode: zero
  min_chunk_content_ratio: 0.5  # Drop chunks with <50% content
  random_pad_position: true  # Randomize padding for augmentation

spectrogram:
  freq_scale: mel
  fmin: 0.0
  fmax: null  # Use Nyquist frequency
```

### Auto-Computed Values

When you set `target_size: 640`:

- `target_width = 640`
- `target_height = 640`
- `window_duration_ms = 640 × 10.0 = 6400ms`

**Why this matters**: This ensures spectrograms have exactly 640 pixels width with **no padding**, eliminating black rectangles on the right side.

## How It Works

### 1. Chunk Boundary Calculation

```python
def compute_chunk_boundaries(
    total_duration_ms: float, config: ChunkConfig
) -> list[tuple[float, float]]:
    """Generate sliding window boundaries.

    Special cases:
    - Audio < window → single chunk (0, total_duration)
    - Drops chunks with content_ratio < min_chunk_content_ratio
    """
```

**Example**: 52-second audio with 6.4s windows, 1.28s overlap

```
Chunk 0: [0, 6400ms]
Chunk 1: [5120, 11520ms]  # Start = 6400 - 1280
Chunk 2: [10240, 16640ms]
...
Chunk 9: [46080, 52085ms]  # Last chunk (81.7% content, kept)
```

### 2. Audio Extraction with Padding

```python
def extract_audio_chunk(
    audio: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    padding_mode: str = "zero",
    random_pad_position: bool = False,
) -> tuple[np.ndarray, bool, float]:
    """Extract audio segment with optional padding.

    Returns:
        (chunk_audio, is_padded, padding_ms)
    """
```

**Padding modes**:

- `zero`: Pad with silence (default)
- `repeat`: Loop audio to fill gap
- `reflect`: Mirror audio at boundaries

**Random positioning**: For very short audio (e.g., 50ms gunshot in 6400ms window), randomizes where it sits in the padded chunk for robustness.

### 3. Spectrogram Computation

```python
from ezakodio.dsp import mel_spectrogram

spec = mel_spectrogram(
    audio_tensor,
    sample_rate=sr,
    n_mels=128,
    n_fft=n_fft,
    hop_length=hop_length,
    f_min=0.0,
    f_max=sr / 2,
)

# Flip vertically: high frequencies at top (standard visualization)
spec = np.flipud(spec)
```

**Result**: (128, 640) array where:

- Height: 128 mel bands (before resize to 640)
- Width: 640 pixels (6400ms / 10ms per pixel)

### 4. Bbox Alignment

```python
def align_bbox_to_chunk(
    event_start_ms: float,
    event_end_ms: float,
    hz_min: float,
    hz_max: float,
    chunk_start_ms: float,
    chunk_end_ms: float,
    chunk_width_px: int,
    chunk_height_px: int,
    freq_min: float,
    freq_max: float,
    category: str,
    category_id: int,
    actual_width_px: int | None = None,  # Content width before padding
    use_mel_scale: bool = True,  # Use mel-scale mapping
) -> ChunkBbox | None:
    """Transform event coordinates to chunk-relative pixels."""
```

**Time mapping** (X-axis):

```python
# Constrain to actual content width (no padding)
width_px = actual_width_px if actual_width_px else chunk_width_px

overlap_start = max(event_start_ms, chunk_start_ms)
overlap_end = min(event_end_ms, chunk_end_ms)

rel_start = overlap_start - chunk_start_ms
rel_end = overlap_end - chunk_start_ms

x = (rel_start / chunk_duration) * width_px
x_end = (rel_end / chunk_duration) * width_px
width = x_end - x
```

**Frequency mapping** (Y-axis with mel scale):

```python
# Convert Hz to mel: mel = 2595 * log10(1 + hz/700)
mel_min = 2595.0 * np.log10(1.0 + freq_min / 700.0)
mel_max = 2595.0 * np.log10(1.0 + freq_max / 700.0)
event_mel_min = 2595.0 * np.log10(1.0 + hz_min / 700.0)
event_mel_max = 2595.0 * np.log10(1.0 + hz_max / 700.0)

# Map to pixel space
y_top = ((event_mel_max - mel_min) / (mel_max - mel_min)) * chunk_height_px
y_bottom = ((event_mel_min - mel_min) / (mel_max - mel_min)) * chunk_height_px

# CRITICAL: Flip Y-axis because spectrogram was flipped with np.flipud()
y_top = chunk_height_px - y_top
y_bottom = chunk_height_px - y_bottom

y = min(y_top, y_bottom)
height = abs(y_bottom - y_top)
```

**Why flip?** The spectrogram is flipped with `np.flipud()` so high frequencies appear at the top (y=0). Bbox coordinates must account for this inversion.

### 5. Resizing Strategy

```python
def resize_spectrogram(
    spec: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize with padding-aware strategy.

    - HEIGHT: Resized (frequency scaling is OK)
    - WIDTH: Padded (preserves time resolution)
    """
```

**Example**: Original (128, 640) → Target (640, 640)

1. Resize height: (128, 640) → (640, 640) via bilinear interpolation
2. Pad width: Not needed (already 640 pixels)

For shorter chunks:

1. Resize height: (128, 408) → (640, 408)
2. Pad width: (640, 408) → (640, 640) with zeros on right

**Result**: Bboxes only span [0, 408] on X-axis (no black rectangles).

## Usage

### CLI Command

```bash
python -m rf_detr_finetuning chunk-audio \
  --audio-dir data/audio/ \
  --metadata-dir data/metadata/ \
  --output-dir data/chunks/ \
  --config config/audio_chunking.yaml
```

### Python API

```python
from pathlib import Path
from rf_detr_finetuning import (
    AudioChunker,
    TimeBasedFFTConfig,
    ChunkConfig,
    load_chunking_config_from_yaml,
)

# Load from config file
fft_config, chunk_config = load_chunking_config_from_yaml(
    Path("config/audio_chunking.yaml")
)

# Or create manually
fft_config = TimeBasedFFTConfig(fft_ms=25, hop_ms=10, n_mels=128)
chunk_config = ChunkConfig(
    window_duration_ms=6400,  # Or use target_size in YAML
    overlap_ms=1280,
    target_width=640,
    target_height=640,
    min_chunk_content_ratio=0.5,
)

# Create chunker
chunker = AudioChunker(fft_config=fft_config, chunk_config=chunk_config)

# Chunk a file
chunks = chunker.chunk_audio_file(
    audio_path=Path("audio.flac"),
    metadata_path=Path("audio.json"),
)

# Access chunk data
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.start_ms:.0f}-{chunk.end_ms:.0f}ms")
    print(f"  Shape: {chunk.spectrogram.shape}")
    print(f"  Bboxes: {len(chunk.bboxes)}")

    for bbox in chunk.bboxes:
        print(
            f"    - {bbox.category}: [{bbox.x:.0f}, {bbox.y:.0f}, "
            f"{bbox.width:.0f}, {bbox.height:.0f}]"
        )
```

## Visual Debugging

### Test with Real Files

```bash
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s
```

**Output**: `output/test_chunks_debug/*.png` with bboxes drawn in green

**Example output**:

```
======================================================================
CHUNKED AUDIO VISUALIZATION
======================================================================
Audio file: ddd4a606-c07c-4c48-879e-e93d729d79c4.flac
JSON file: ddd4a606-c07c-4c48-879e-e93d729d79c4.json
Config: window=6400.0ms, overlap=1280.0ms, size=640x640
        min_content_ratio=0.5, random_pad=True
Total chunks: 10
Output directory: /path/to/output/test_chunks_debug
======================================================================

Chunk 00:      0-  6400ms | 1 bbox(es) | normal | ..._chunk0000_debug.png
  └─ Bbox 0: [   0,   77,  640,  430] | label=odontoceti | overlap=0.12
...
```

### Drawing Bboxes

```python
from rf_detr_finetuning import draw_bboxes_on_spectrogram

img = draw_bboxes_on_spectrogram(
    spectrogram=chunk.spectrogram,
    bboxes=chunk.bboxes,
    output_path=Path("debug.png"),  # Optional
)
```

Uses `supervision` library for professional bbox rendering with labels.

## Common Issues & Solutions

### Issue 1: Black Rectangles on Right Side

**Symptom**: Spectrograms have black padding, bboxes extend into it

**Solution**: Set `target_size` properly in config

```yaml
chunking:
  target_size: 640  # Auto-computes window_duration_ms = 640 × hop_ms
```

Formula: `window_duration_ms = target_size × hop_ms`

### Issue 2: Upside-Down Bboxes

**Symptom**: Bboxes appear at wrong vertical position

**Solution**: Ensure Y-coordinates account for `np.flipud()` flip:

```python
# After mel-scale mapping
y_top = chunk_height_px - y_top
y_bottom = chunk_height_px - y_bottom
```

### Issue 3: Missing File-Level Annotations

**Symptom**: File-level bboxes not appearing in chunks

**Solution**: Set `is_file_level=True` in event dict:

```python
event = {
    "time_start_ms": 0,
    "time_end_ms": 52085,
    "hz_min": 1000,
    "hz_max": 8000,
    "category": "odontoceti",
    "category_id": 0,
    "is_file_level": True,  # Bypass overlap threshold
}
```

### Issue 4: Black Blobs at End

**Symptom**: Last chunk is mostly black padding

**Solution**: Increase `min_chunk_content_ratio`:

```yaml
chunking:
  min_chunk_content_ratio: 0.5  # Drop chunks with <50% content
```

### Issue 5: Very Short Audio Dropped

**Symptom**: 50ms gunshots are being dropped

**Solution**: Use random padding for robustness:

```yaml
chunking:
  min_chunk_content_ratio: 0.0  # Keep all chunks
  random_pad_position: true      # Randomize short audio position
```

Special case: Audio shorter than window creates a single chunk.

## Testing

### Run All Tests

```bash
pytest tests/test_audio_chunking.py -v
```

**Expected**: 40 tests passing

### Test Coverage

- `TimeBasedFFTConfig`: Sample rate conversions, time/freq resolution
- `ChunkConfig`: Validation, defaults
- `compute_chunk_boundaries()`: Single/multiple chunks, exact fit, min_content_ratio
- `compute_bbox_overlap()`: Full/partial/no overlap, zero duration
- `align_bbox_to_chunk()`: Full/partial event, no overlap filtering
- `pad_audio()`: Zero/repeat/reflect modes
- `extract_audio_chunk()`: Middle extraction, padding scenarios
- `spectrogram_to_image_array()`: Normalization, dtype handling
- `resize_spectrogram()`: 2D/3D arrays, padding
- `ChunkBbox`: COCO/xyxy conversions
- `AudioChunk`: Duration calculation, ID generation
- `AudioChunker`: Synthetic/real audio, short audio, random padding
- `draw_bboxes_on_spectrogram()`: Grayscale/RGB, empty bboxes
- `load_chunking_config_from_yaml()`: Config parsing, defaults

### Visual Verification

```bash
# Generate debug images
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

# View images
eog output/test_chunks_debug/*.png
```

## Performance Considerations

### Memory Usage

Each chunk stores:

- Spectrogram: 640 × 640 × 3 bytes = ~1.2 MB
- Bboxes: ~100 bytes per bbox
- Metadata: ~200 bytes

For 10 chunks: ~12 MB per audio file

### Optimization Tips

1. **Batch processing**: Process multiple files in parallel
2. **Streaming**: Use generators for large datasets
3. **Lazy loading**: Only load spectrograms when needed
4. **GPU**: mel_spectrogram uses CPU by default, can use GPU

## Advanced Topics

### Custom Frequency Scales

Override `freq_scale` parameter:

```python
chunker = AudioChunker(
    fft_config=fft_config,
    chunk_config=chunk_config,
    freq_scale="linear",  # or "log"
)
```

**Note**: Linear scale doesn't use mel conversion in bbox mapping.

### Label Hierarchy Extraction

For nested labels (e.g., "whistles + clicks + odontoceti"):

```python
# Extract last part only
label = event.get("label_hierarchy", "").split(" + ")[-1]
# Result: "odontoceti"
```

### Custom Padding Strategies

Implement custom padding in `pad_audio()`:

```python
def pad_audio(audio, target_samples, mode="custom"):
    if mode == "custom":
        # Your padding logic here
        pass
```

## References

- **ezakodio**: https://github.com/earthspecies/ezakodio
- **supervision**: https://supervision.roboflow.com/
- **RF-DETR**: https://github.com/roboflow/rf-detr
- **Mel scale**: https://en.wikipedia.org/wiki/Mel_scale

## Contributing

When modifying the chunking system:

1. ✅ Run all tests: `pytest tests/test_audio_chunking.py -v`
2. ✅ Visual check: `pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s`
3. ✅ Update docstrings and this documentation
4. ✅ Maintain backward compatibility with old config format
5. ✅ Never add librosa dependency (use numpy mel conversion)
