# GitHub Copilot Instructions for RF-DETR Fine-Tuning Pipeline

## Project Context

This is an RF-DETR (Real-time DEtection TRansformer) fine-tuning pipeline with specialized support for **audio event detection**. The project converts variable-length audio files into fixed-size spectrograms and trains object detection models to detect temporal events.

## Core Technologies

- **RF-DETR**: Roboflow's object detection transformer
- **ezakodio**: Audio I/O and mel spectrogram generation
- **supervision**: Bbox visualization and utilities
- **Rich**: Progress bars and terminal formatting
- **PyYAML**: Configuration management
- **pytest**: Testing framework

## Code Style Rules

### Formatting

- **NO long separator comments** (e.g., `# ============================`)
- Use **Rich** for progress bars, **never tqdm**
- Use `logging` with `RichHandler` for console output
- Avoid `print()` statements for debugging
- Full type hints for all functions
- Docstrings in Google style

### Error Handling

- Wrap I/O operations in try/except
- Return `None` or raise specific exceptions
- Never silently fail
- Add memory cleanup in `finally` blocks

### Comments

- Add comments only when code is not self-explanatory
- Explain *why*, not *what*
- Document non-obvious decisions and edge cases

## Audio Chunking System

### Critical Constraints

1. **NEVER use librosa** - Use numpy-based mel conversion:

   ```python
   mel = 2595.0 * np.log10(1.0 + hz / 700.0)
   ```

2. **Perfect pixel alignment formula**:

   ```python
   window_duration_ms = target_width * hop_ms
   ```

   Example: 640px × 10ms = 6400ms (no padding!)

3. **Y-axis is flipped**: After `np.flipud()`, high frequencies are at top (y=0)

   - Bbox Y-coordinates must account for this flip
   - Use: `y = chunk_height_px - y_mel_based`

4. **Bbox width constraint**: Use `actual_width_px` (before padding), not `target_width`

   - Prevents bboxes from extending into black padding regions

5. **Mel-scale frequency mapping**: Default to mel scale for frequency coordinates

   - Linear Hz mapping is a fallback option
   - Pass `use_mel_scale=True` to `align_bbox_to_chunk()`

### Configuration Pattern

```yaml
# Simplified config - values are auto-computed
fft:
  hop_ms: 10.0
  fft_ms: 25.0
  n_mels: 128

chunking:
  target_size: 640  # Auto-computes: width=640, height=640, window_duration_ms=6400
  overlap_ms: 1280.0
  min_chunk_content_ratio: 0.5  # Drop chunks with <50% content
  random_pad_position: true  # Randomize padding for augmentation
```

### Key Classes

**TimeBasedFFTConfig**: Sample-rate independent FFT parameters

- Methods: `get_n_fft(sr)`, `get_hop_length(sr)`, `time_per_pixel()`

**ChunkConfig**: Chunking behavior

- `target_size` → auto-computes `target_width`, `target_height`, `window_duration_ms`
- `min_chunk_content_ratio`: Drop threshold for padded chunks
- `random_pad_position`: Randomize short audio position

**AudioChunk**: Processed chunk with aligned bboxes

- `spectrogram`: Flipped (np.flipud) for standard visualization
- `bboxes`: List of ChunkBbox with chunk-relative coordinates
- `is_padded`, `padding_amount_ms`: Padding metadata

**ChunkBbox**: Bbox with overlap information

- Coordinates: COCO format (x, y, width, height)
- `overlap_ratio`: Event overlap with chunk
- `is_file_level`: Bypasses overlap threshold if True

### Common Functions

**`align_bbox_to_chunk()`**: Transform event coordinates to chunk pixels

- Parameters: `actual_width_px`, `use_mel_scale` (both important!)
- Returns: ChunkBbox or None if no overlap

**`resize_spectrogram()`**: Resize with padding-aware strategy

- HEIGHT: Resized (frequency scaling is OK)
- WIDTH: Padded (preserves time resolution)

**`compute_chunk_boundaries()`**: Generate sliding windows

- Special case: audio < window → single chunk (0, total_duration)
- Drops chunks with content ratio < `min_chunk_content_ratio`

**`extract_audio_chunk()`**: Extract with optional padding

- `random_pad_position`: Randomize pad_before vs pad_after
- Returns: (audio, is_padded, padding_ms)

### Testing Guidelines

- Visual debugging: `pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s`
- Output: `output/test_chunks_debug/*.png`
- All tests must pass: `pytest tests/test_audio_chunking.py -v`
- Current: 40 tests passing

### Common Pitfalls to Avoid

1. **Black padding blobs**: Set `min_chunk_content_ratio > 0.0`
2. **Upside-down bboxes**: Y-coords must flip after `np.flipud()`
3. **Bboxes in padding**: Always pass `actual_width_px`
4. **Missing file-level bboxes**: Set `is_file_level=True` in event
5. **Using librosa**: Use numpy mel conversion instead
6. **Hardcoded dimensions**: Use `target_size` in config

### File Structure

```
src/rf_detr_finetuning/
├── audio_chunking.py      # ~950 lines - main chunking logic
├── cli.py                 # CLI commands (chunk-audio)
├── data.py                # Dataset utilities
├── finetune.py            # Training logic
└── predict.py             # Inference

config/
└── audio_chunking.yaml    # Chunking configuration

tests/
└── test_audio_chunking.py # 40 unit tests
```

### Label Hierarchy Handling

When extracting labels from metadata:

- Use **last part** of `label_hierarchy` (e.g., "odontoceti" not "whistles + clicks")
- Fallback to `annotation` field if `label_hierarchy` missing
- Always extract as: `event.get("label_hierarchy", "").split(" + ")[-1]`

### Visualization

Use `supervision` library for drawing bboxes:

```python
import supervision as sv

detections = sv.Detections(xyxy=xyxy, class_id=class_ids)
annotator = sv.BoxAnnotator(thickness=2)
annotated = annotator.annotate(image, detections)
```

Fallback manual drawing if supervision not available (see `_draw_bboxes_manual`).

## CLI Design

### Command Structure

```bash
# Audio chunking
python -m rf_detr_finetuning chunk-audio \
  --audio-dir data/audio/ \
  --metadata-dir data/metadata/ \
  --output-dir data/chunks/ \
  --config config/audio_chunking.yaml

# Training (existing)
python -m rf_detr_finetuning train --config config/train.yaml

# Prediction (existing)
python -m rf_detr_finetuning predict --image path/to/image.jpg
```

### Configuration Loading

- YAML config file is preferred
- CLI args override config values
- Auto-compute derived values (window_duration_ms from target_size)
- Backward compatible with old format (explicit dimensions)

## Dependencies

### Core

- numpy, torch, PIL, PyYAML
- ezakodio (audio), supervision (viz), Rich (UI)

### Optional

- pytest, pytest-cov (testing)
- ruff, pre-commit (linting)

### **NEVER** Add

- librosa (use numpy mel conversion)
- tqdm (use Rich progress)

## Documentation References

Read these for domain context:

- `docs/AUDIO_TO_COCO_GUIDE.md` - Audio → COCO conversion
- `docs/AUDIO_CHUNKING.md` - Chunking system details
- `docs/FREQUENCY_AWARE_CLASSIFICATION.md` - Frequency-based categories
- `docs/MIXED_SAMPLE_RATES.md` - Sample rate handling
- `.github/agents/AGENTS.md` - Agent configuration

## When Suggesting Code

1. Check if functionality exists in `audio_chunking.py` before adding
2. Maintain type hints and docstrings
3. Add tests for new features
4. Update config examples in docstrings
5. Follow existing patterns (e.g., TimeBasedFFTConfig style)
6. Use Rich for any progress/UI elements
7. Never break the 40 passing tests

## Special Notes

- **Spectrograms are flipped**: Always account for `np.flipud()` in bbox calculations
- **Padding is strategic**: Width padded (time), height resized (frequency)
- **Config is smart**: One value (`target_size`) computes all dimensions
- **Testing is visual**: Generated images help verify bbox alignment
- **No black blobs**: `min_chunk_content_ratio` ensures quality chunks
