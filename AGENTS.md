# Agent Guidelines for RF-DETR Fine-Tuning Pipeline

## Build/Lint/Test Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio_chunking.py -v

# Run single test
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

# Run with real file markers
pytest tests/ -m "real_files" -v

# Linting and formatting
pre-commit run --all-files

# Or run individual tools
ruff check . --fix
ruff format .
codespell --write-changes
docformatter --in-place --recursive
```

## Code Style Guidelines

### Formatting

- **Line length**: 120 characters (configured in pyproject.toml)
- **No long separator comments** like `# ============================` or `# ----------------------------`
- Use docstrings for functions/classes, not banner comments
- Full type hints for all function signatures
- Docstrings in **Google style** (not NumPy or reST)

### Imports

- Use `isort` (via ruff) for import sorting
- Standard library imports first, then third-party, then local
- Absolute imports preferred over relative
- Example:
  ```python
  import logging
  from pathlib import Path
  from typing import Optional

  import numpy as np
  from PIL import Image

  from rf_detr_finetuning.audio_chunking import AudioChunker
  ```

### Naming Conventions

- **Classes**: PascalCase (e.g., `AudioChunker`, `TimeBasedFFTConfig`)
- **Functions/variables**: snake_case (e.g., `compute_mel_spectrogram`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_HOP_MS`)
- **Private methods**: prefix with underscore (e.g., `_internal_helper()`)

### Error Handling

- Wrap I/O operations in try/except blocks
- Return `None` or raise specific exceptions, never silently fail
- Add memory cleanup in `finally` blocks for large objects
- Use standard exceptions: `ValueError`, `FileNotFoundError`, `RuntimeError`

### Logging & Progress

- Use **Rich** library for progress bars (`rich.progress`), **never tqdm**
- Use `logging` with `RichHandler` for console output
- Avoid `print()` statements for debugging

### Comments

- Add comments only when code is not self-explanatory
- Explain **why**, not **what**
- Document non-obvious decisions and edge cases

## Audio Processing Rules

### Critical Constraints

1. **NEVER use librosa** - Use numpy-based mel conversion:

   ```python
   mel = 2595.0 * np.log10(1.0 + hz / 700.0)
   ```

2. **Perfect pixel alignment**:

   ```python
   window_duration_ms = target_width * hop_ms
   # Example: 640px × 10ms = 6400ms (no padding!)
   ```

3. **Y-axis is flipped**: After `np.flipud()`, high frequencies are at top (y=0)

   - Bbox Y-coordinates must account for this flip
   - Use: `y = chunk_height_px - y_mel_based`

4. **Bbox width constraint**: Use `actual_width_px` (before padding), not `target_width`

   - Prevents bboxes from extending into black padding regions

5. **Mel-scale frequency mapping**: Default to mel scale for frequency coordinates

### Testing Audio Code

```bash
# Visual debugging with real files
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

# Output images help verify bbox alignment
# Check: output/test_chunks_debug/*.png
```

## Project Structure

```
src/rf_detr_finetuning/
├── dataprocessor/      # Audio I/O, spectrogram features, chunking
├── dataloader/         # Dataset classes, collate functions
├── trainer/            # Training loop, checkpointing
├── predictor/          # Inference on images/audio
├── eventprocessor/     # Convert window detections to full-audio events
├── audio_chunking.py   # Legacy chunking (backward compat)
└── audio_to_coco.py    # Legacy COCO conversion (backward compat)
```

## Configuration Patterns

Use YAML configs with auto-computed values:

```yaml
fft:
  hop_ms: 10.0
  fft_ms: 25.0
  n_mels: 128

chunking:
  target_size: 640  # Auto-computes width=640, height=640
  overlap_ms: 1280.0
  min_chunk_content_ratio: 0.5
  random_pad_position: true
```

## Pre-commit Hooks

This project uses pre-commit for automated checks:

- end-of-file-fixer
- trailing-whitespace
- check-yaml, check-json, check-toml
- codespell (spell checking)
- docformatter (docstring formatting)
- ruff (linting and formatting)
- mdformat (markdown formatting)

## Security & Best Practices

- Never commit `.env` or API keys
- PRs touching training/inference logic must be reviewed
- All training runs must log to W&B with project/run name matching the dataset
- Dataset usage must include version pinning (e.g., `project/version` in Roboflow)
- Inference scripts must use `get_model()` or `RFDETR*` classes from `rfdetr`

## Documentation References

- `docs/AUDIO_TO_COCO_GUIDE.md` - Audio → COCO conversion
- `docs/AUDIO_CHUNKING.md` - Chunking system details
- `docs/FREQUENCY_AWARE_CLASSIFICATION.md` - Frequency-based categories
- `docs/MIXED_SAMPLE_RATES.md` - Sample rate handling
- `docs/FAQ_FREQUENCY_FEATURES.md` - FAQ for frequency features
- `docs/ROAST_FIXES_SUMMARY.md` - Code quality improvements
