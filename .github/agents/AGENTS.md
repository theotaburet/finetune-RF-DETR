# Agent HQ Configuration for RF-DETR Fine-Tuning (Roboflow Stack)

## 🧠 Agents

### Engineer

**Role**: RF-DETR fine-tuning and deployment specialist
**Tools**: Python, Roboflow SDK, `rfdetr`, `supervision`, Weights & Biases
**Behavior**:

- Review PRs modifying `train.py`, `config.yaml`, or `inference.py`
- Validate dataset loading via Roboflow API and class mappings
- Ensure correct use of `optimize_for_inference()` and model export
- Check reproducibility: fixed seeds, versioned datasets, consistent configs

### Doc-Scribe

**Role**: Documentation and reproducibility assistant
**Tools**: Markdown, GitHub Wiki, W&B Reports
**Behavior**:

- Maintain README with setup, training, and inference instructions
- Auto-generate model cards from W&B runs and config metadata
- Ensure Roboflow dataset links and model IDs are documented
- Track changes to `config.yaml` and document rationale

### Mentor-Bot

**Role**: Communication and feedback facilitator
**Tools**: GitHub Issues, Discussions, Email Drafting
**Behavior**:

- Draft follow-ups after demo sessions or PR merges
- Summarize feedback from reviewers and suggest next steps
- Track mentorship trial progress and flag missing responses
- Help onboard new contributors with Roboflow-specific guides

## 🔐 Permissions

| Agent      | Branch Access  | PR Review | Issue Commenting |
| ---------- | -------------- | --------- | ---------------- |
| engineer   | `main`, `dev`  | ✅        | ✅               |
| doc-scribe | `docs`, `main` | ✅        | ✅               |
| mentor-bot | `main`         | ❌        | ✅               |

## 📚 Context

Agents may read and reference:

- `README.md`, `config.yaml`, `train.py`, `inference.py`
- `rfdetr/`, `supervision/`, `notebooks/`
- W&B run metadata and Roboflow project/version strings

## 🧭 Mission Rules

- Never commit `.env` or API keys
- PRs touching `train.py`, `config.yaml`, or `inference.py` must be reviewed by `engineer`
- All training runs must log to W&B with project/run name matching the dataset
- Dataset usage must include version pinning (e.g., `project/version` in Roboflow)
- Inference scripts must use `get_model()` or `RFDETR*` classes from `rfdetr`

## 🧪 Protocols

### Fine-Tuning Validation

- Confirm `ROBOFLOW_API_KEY` is loaded securely
- Validate dataset pull via `rf.load()` or CLI
- Ensure correct resolution and class count in `config.yaml`
- Check for `model.optimize_for_inference()` before export

### Documentation Update

- Update README if CLI, config, or training logic changes
- Include example usage:
  ```bash
  python train.py --config config.yaml
  python inference.py --image path/to/image.jpg --model rf-model/1
  ```

## 📋 Best Practices

### Code Style

**Formatting Rules:**

- **NO long separator comments** like `# ============================` or `# ----------------------------`
- Keep code clean and let structure speak for itself
- Use docstrings for functions/classes, not banner comments
- Follow DRY (Don't Repeat Yourself) and KISS (Keep It Simple, Stupid) principles

**Progress & Logging:**

- Use **Rich** library for progress bars (not tqdm)
- Use `logging` with `RichHandler` for console output
- Avoid print() statements for debugging

**Error Handling:**

- Always wrap I/O operations in try/except
- Return `None` or raise specific exceptions, never silently fail
- Add memory cleanup in `finally` blocks for large objects

### Code Comments

When writing or modifying code, add comments if the code is not self-explanatory.
This improves readability, maintainability, and helps other contributors understand complex logic or non-obvious decisions.

## 📖 Project Documentation

Agents should reference these docs for domain-specific guidance:

| Document                                 | Purpose                                       |
| ---------------------------------------- | --------------------------------------------- |
| `docs/AUDIO_TO_COCO_GUIDE.md`            | Audio dataset conversion to COCO format       |
| `docs/FREQUENCY_AWARE_CLASSIFICATION.md` | Frequency-based category splitting            |
| `docs/MIXED_SAMPLE_RATES.md`             | Handling datasets with different sample rates |
| `docs/FAQ_FREQUENCY_FEATURES.md`         | FAQ for frequency features                    |
| `docs/ROAST_FIXES_SUMMARY.md`            | Code quality improvements and patterns        |
| `docs/AUDIO_CHUNKING.md`                 | Audio chunking system for RF-DETR training    |

## 🎵 Audio Processing Pipeline

### Audio Chunking System

The project includes a sophisticated audio chunking module (`src/rf_detr_finetuning/audio_chunking.py`) that converts variable-length audio files into fixed-size spectrograms for object detection training.

**Key Features:**

- **Time-based FFT**: Sample-rate independent configuration (fft_ms, hop_ms)
- **Sliding windows**: Configurable overlap for capturing events at chunk boundaries
- **Mel-scale bboxes**: Frequency coordinates mapped to mel scale for better alignment
- **Padding awareness**: Bboxes don't extend into black padding regions
- **Smart dropping**: `min_chunk_content_ratio` drops chunks with excessive padding
- **Random augmentation**: `random_pad_position` for very short audio (e.g., 50ms gunshots)

**Configuration Pattern:**

```yaml
# config/chunking.yaml
fft:
  hop_ms: 10.0  # Time resolution (ms per pixel)
  fft_ms: 25.0
  n_mels: 128

chunking:
  target_size: 640  # Auto-computes: width=height=640, window_duration_ms=6400
  overlap_ms: 1280.0  # 20% overlap
  min_chunk_content_ratio: 0.5  # Drop chunks with <50% content
  random_pad_position: true  # Randomize padding for robustness
```

**Critical Rules:**

1. **Never use librosa** - Use numpy-based mel conversion: `2595 * log10(1 + hz/700)`
2. **window_duration_ms = target_size × hop_ms** - Ensures no width padding
3. **Y-axis is flipped** - After `np.flipud()`, high frequencies at top (y=0)
4. **Bbox coordinates**:
   - Width: Constrained to `actual_width_px` (before padding)
   - Height: Mel-scale mapped, then flipped for visualization
5. **File-level annotations**: Use `is_file_level=True` to bypass overlap threshold

**Testing:**

- 40 unit tests in `tests/test_audio_chunking.py`
- Visual debugging: `pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s`
- Output: `output/test_chunks_debug/*.png` with bboxes drawn

### Audio Data Classes

**TimeBasedFFTConfig**: SR-independent FFT parameters

- `fft_ms`: FFT window duration
- `hop_ms`: Time per spectrogram pixel
- `n_mels`: Mel bands (height before resize)

**ChunkConfig**: Chunking behavior

- `target_size`: Square image size (auto-computes dimensions)
- `window_duration_ms`: Auto-computed (target_size × hop_ms)
- `overlap_ms`: Sliding window overlap
- `min_overlap_with_event_ratio`: Bbox inclusion threshold (default 0.3)
- `min_chunk_content_ratio`: Drop threshold for padded chunks (default 0.5)
- `padding_mode`: "zero", "repeat", or "reflect"
- `random_pad_position`: Randomize short audio position (default True)

**AudioChunk**: Processed chunk with metadata

- `spectrogram`: uint8 array (H, W) or (H, W, 3)
- `bboxes`: List of ChunkBbox with aligned coordinates
- `start_ms`, `end_ms`: Temporal boundaries
- `is_padded`: Whether audio was shorter than window
- `padding_amount_ms`: How much padding was added

**ChunkBbox**: Bbox aligned to chunk coordinates

- `x, y, width, height`: Pixel coordinates (COCO format)
- `category`, `category_id`: Label information
- `overlap_ratio`: How much of event is in chunk
- `hz_min`, `hz_max`: Original frequency bounds
- `original_time_start_ms`, `original_time_end_ms`: Original timestamps

### Common Pitfalls

1. **Black padding blobs**: Set `min_chunk_content_ratio > 0.0` to drop them
2. **Upside-down bboxes**: Ensure Y-coordinates account for `np.flipud()`
3. **Bboxes in padding**: Pass `actual_width_px` to `align_bbox_to_chunk()`
4. **Wrong frequency mapping**: Use mel-scale, not linear Hz
5. **Missing file-level bboxes**: Set `is_file_level=True` in event dict

### CLI Commands

```bash
# Chunk audio files with YAML config
python -m rf_detr_finetuning chunk-audio \
  --audio-dir data/audio/ \
  --metadata-dir data/metadata/ \
  --output-dir data/chunks/ \
  --config config/chunking.yaml
```
