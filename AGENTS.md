# Agent Guidelines for RF-DETR Fine-Tuning Pipeline

## Project Overview

End-to-end pipeline for **marine acoustic event detection** using RF-DETR object detection on spectrograms. Audio files are converted to mel spectrograms, chunked into 640x640 images, annotated in COCO format, used to fine-tune RF-DETR, and post-processed back into temporal audio events.

**Pipeline flow:** Download (EKB API) -> Preprocess (audio -> spectrogram chunks) -> Split (train/val/test) -> Train (RF-DETR fine-tune) -> Evaluate -> Optimize Merging -> Infer

**Current test configuration:** Single class (`mysticeti`), small model size.

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

# Run the full pipeline
python run_pipeline.py --config config/pipeline.yaml

# Run individual steps
python run_dataprocessing.py --config config/chunking.yaml --audio-dir data/downloaded
python run_training.py --dataset-dir data/split_dataset --model-size small
python run_inference_audio.py --weights output/checkpoint_best.pth --audio-dir data/inference
```

## Project Structure

```
finetune-RF-DETR/
├── run_pipeline.py              # Master orchestrator (all 7 steps, toggleable)
├── run_download_data.py         # Step 1: Download from EKB API
├── run_dataprocessing.py        # Step 2: Audio -> spectrogram chunks
├── run_training.py              # Step 4: RF-DETR fine-tuning
├── run_inference.py             # Step 6: Image/audio inference
├── run_inference_audio.py       # Full audio inference pipeline
├── run_merging.py               # Event merging CLI
├── run_optimize_merging.py      # Step 5: Merge parameter optimization
│
├── config/
│   ├── pipeline.yaml            # Master config (all steps)
│   ├── chunking.yaml            # FFT, chunking, preprocessing params
│   ├── download.yaml            # EKB API config, split config
│   ├── merging.yaml             # Class-wise merge parameters
│   └── examples/                # Example configs
│
├── src/rf_detr_finetuning/
│   ├── __main__.py              # CLI entry: python -m rf_detr_finetuning
│   ├── cli.py                   # jsonargparse CLI (download, convert, train, predict)
│   ├── utils.py                 # load_class_names()
│   │
│   ├── dataprocessor/           # Audio processing core
│   │   ├── io.py                # Audio file loading (via ezakodio)
│   │   ├── features.py          # Mel spectrogram computation, flip
│   │   ├── preprocessing.py     # AGC, detrend, preemphasis, dynamic range
│   │   ├── chunking.py          # Data classes, chunk boundaries, bbox alignment
│   │   ├── chunker.py           # AudioChunker orchestrator class
│   │   ├── normalization.py     # Spectrogram normalization, resize, RGB conversion
│   │   ├── augmentation.py      # Noise injection, spec augment, time shift
│   │   └── visualization.py     # Debug bbox overlay on spectrograms
│   │
│   ├── dataloader/              # Dataset management
│   │   ├── dataset.py           # COCOAudioDataset, AudioChunkDataset, InMemoryChunkDataset
│   │   ├── collate.py           # Collate functions for detection training
│   │   ├── splitter.py          # Train/val/test splitting with stratification
│   │   ├── sampler.py           # BalancedClassSampler, DeterministicSampler
│   │   └── coco_export.py       # COCO format export, CategoryRegistry
│   │
│   ├── trainer/                 # Training infrastructure
│   │   ├── rfdetr_wrapper.py    # RFDETRTrainer, model size selection
│   │   ├── config.py            # TrainerConfig, OptimizerConfig, etc.
│   │   ├── loop.py              # TrainingState
│   │   ├── checkpoint.py        # CheckpointManager
│   │   ├── logger.py            # TrainingLogger, MetricsTracker
│   │   └── metrics.py           # compute_coco_metrics()
│   │
│   ├── predictor/               # Inference
│   │   ├── inference.py         # RFDETRPredictor, Detection, PredictionResult
│   │   ├── batch.py             # BatchPredictor, predict_directory()
│   │   └── audio.py             # AudioPredictor, windows_to_events conversion
│   │
│   └── eventprocessor/          # Post-processing detections -> audio events
│       ├── event.py             # AudioEvent, EventList data classes
│       ├── postprocessor.py     # EventPostProcessor, windows_to_events()
│       ├── merger.py            # EventMerger, ClassWiseMerger, NMS, temporal merge
│       ├── evaluator.py         # evaluate_events(), match_events(), ClassMetrics
│       └── optimizer.py         # Grid search for merge parameters
│
├── data/
│   ├── downloaded/              # Raw audio + sidecar JSON (train/valid/test splits)
│   └── processed/               # Chunked spectrogram PNGs + COCO annotations
│
├── tests/
│   ├── conftest.py
│   ├── test_audio_chunking.py, test_data.py, test_pipeline_dry_run.py, ...
│   ├── unit/                    # Unit tests per module
│   └── regression/              # Baseline snapshots
│
├── scripts/
│   └── sync_project.sh          # rsync to remote GPU server
│
└── docs/                        # Design docs (hierarchy, class IDs, merging, etc.)
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

  from rf_detr_finetuning.dataprocessor.chunker import AudioChunker
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
   # Example: 640px * 5ms = 3200ms (no padding!)
   ```

3. **Y-axis convention**: `ezakodio.viz.spectrogram_to_image` flips vertically internally — high frequencies end up at top (y=0) in the output image. Do NOT call `np.flipud()` before passing a spectrogram to it (double-flip = upside down).

   - Bbox Y-coordinates must still account for this flip
   - Use: `y = chunk_height_px - y_mel_based`

4. **Bbox width constraint**: Use `actual_width_px` (before padding), not `target_width`

   - Prevents bboxes from extending into black padding regions
   - **Known issue**: Clamping at `chunking.py:363-366` uses `chunk_width_px` instead of `actual_width_px`

5. **Mel-scale frequency mapping**: Default to mel scale for frequency coordinates

6. **Spectrogram rendering**: Use `ezakodio.viz.spectrogram_to_image` with `cmap="jet"`

   - Outputs RGB uint8 arrays with shape `(H, W, 3)`
   - Downstream code should use `shape[:2]` for height/width
   - Only call `grayscale_to_rgb()` when input is 2D

7. **Flip + torch conversion**: `np.flipud()` creates negative strides

   - Call `.copy()` before `torch.from_numpy()` when using flipped arrays as tensors
   - `spectrogram_to_image` handles its own flip internally — do not pre-flip before calling it

8. **Optional spectrograms**: `AudioChunk.spectrogram` can be `None`

   - Always guard before reading shapes or normalizing

### Known Chunking Issues (Under Investigation)

These issues were identified in the chunking pipeline and should be tracked:

1. **`random_pad_position` is effectively dead code** (`chunking.py:457-461`, `chunker.py:181`): `compute_chunk_boundaries` clamps `end_ms` to `total_duration_ms`, so `extract_audio_chunk` never needs to pad audio. The config option has no effect in the standard pipeline. Visual padding (from `resize_spectrogram`) is what actually fills shorter chunks.

2. **Bbox clamping uses wrong width** (`chunking.py:363-366`): Bboxes are clamped to `chunk_width_px` (target=640) instead of `actual_width_px`. This allows bboxes to theoretically extend into the zero-padded region of resized spectrograms.

3. **Silent event loss at audio tail** (`chunking.py:226-235`): Events in trailing audio shorter than `min_chunk_content_ratio * window_duration_ms` are silently dropped with no warning logged.

4. **Dataset chunk count overestimation** (`dataset.py:262-286`): `AudioChunkDataset._precompute_chunk_counts` does not account for `min_chunk_content_ratio` filtering, causing the last chunk per file to be duplicated during training via the fallback at `dataset.py:328-329`.

### Dataset Split Rules

- **Class-balanced validation**: Validation sets must be balanced by class for metrics
- **Audio-stem integrity**: Keep stems together in the same split (no leakage)

### Event Grouping Rules

- **Grouped events**: Overlapping/grouped events must be merged into a single annotation
- **Sidecar JSON**: Store merged segments with grouped labels metadata

### Testing Audio Code

```bash
# Visual debugging with real files
pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

# Output images help verify bbox alignment
# Check: output/test_chunks_debug/*.png
```

### Debug Visualization

- Enable via `preprocess.debug_visualize: true` in `config/pipeline.yaml`
- Output location: `data/processed/debug_viz/`

## Configuration Architecture

### Key Config Files

| File                   | Purpose                        | Key params                                                |
| ---------------------- | ------------------------------ | --------------------------------------------------------- |
| `config/pipeline.yaml` | Master pipeline (all steps)    | `class_names`, `model_size`, step `enabled` flags         |
| `config/chunking.yaml` | FFT + chunking + preprocessing | `fft_ms=50`, `hop_ms=5`, `n_mels=256`, `target_size=640`  |
| `config/download.yaml` | EKB API + split config         | API URL/token, `test_source_files`, `confidence_min=0.20` |
| `config/merging.yaml`  | Post-detection event merging   | Per-class `delta_time`, `delta_freq`, `score_threshold`   |

### Critical Config Relationships

- **Class ID = index in `class_names` list** (0-based). Must match `merging.yaml` integer keys.
- **`window_duration_ms`** is auto-computed as `target_size * hop_ms` (640 * 5 = 3200ms).
- **RF-DETR convention**: `num_classes = len(class_names) + 1` (background class).
- **Model sizes**: `nano`, `small`, `base`, `medium`, `large` (from `rfdetr` package).

### Current Test Config

```yaml
# pipeline.yaml
class_names: ["mysticeti"]    # Single-class for testing
train:
  model_size: small           # Fast iteration
  epochs: 50
  batch_size: 4
```

## Key Dependencies

- **rfdetr**: RF-DETR model classes (RFDETRNano through RFDETRLarge)
- **ezakodio**: Audio I/O and spectrogram visualization (git dependency from ezako)
- **supervision**: Detection visualization
- **jsonargparse**: CLI argument parsing with YAML config support
- **scipy**: Spectrogram resizing via `scipy.ndimage.zoom`

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

- `docs/HIERARCHY_AWARE_CLASSIFICATION.md` - Label hierarchy system
- `docs/DOWNLOAD_DATA.md` - Data download guide
- `docs/CLASS_ID_WIRING.md` - Class ID mapping
- `docs/EVENT_MERGING.md` - Event merging details
- `docs/DEAD_CODE_ANALYSIS.md` - Dead code audit
