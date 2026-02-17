# Dead Code Analysis Report

## Summary

This document identifies dead code in the RF-DETR fine-tuning repository. Dead code refers to code that is defined but never used, called, or executed.

**Generated:** 2026-02-17

______________________________________________________________________

## Categories of Dead Code

### 1. Deprecated/Legacy Functions (Scheduled for Removal)

These functions are marked as deprecated and should be removed in v0.3.0:

| File                                 | Function/Class           | Status         | Reason                                                    |
| ------------------------------------ | ------------------------ | -------------- | --------------------------------------------------------- |
| `src/rf_detr_finetuning/finetune.py` | `finetune_model()`       | **DEPRECATED** | Legacy training wrapper, replaced by `RFDETRTrainer`      |
| `src/rf_detr_finetuning/predict.py`  | `prediction()`           | **DEPRECATED** | Legacy prediction function, replaced by `RFDETRPredictor` |
| `src/rf_detr_finetuning/data.py`     | `convert_yolo_to_coco()` | **DEPRECATED** | Only used in tests, legacy data conversion                |

**Recommendation:** Remove these in v0.3.0 release.

______________________________________________________________________

### 2. Potentially Unused Functions

These functions are defined but may have no callers in the main codebase:

#### In `src/rf_detr_finetuning/eventprocessor/merger.py`:

- **`soft_nms()`** (line 252-319): Defined but never called in the codebase
  - This is an alternative NMS implementation
  - Currently only `EventMerger.merge()` and `ClassWiseMerger.merge()` are used
  - **Action:** Either document as available alternative or remove

#### In `src/rf_detr_finetuning/dataprocessor/augmentation.py`:

- **`mixup_audio()`** (line 171-221): Exported but never used in training pipeline
  - Mixup augmentation is defined but not integrated into training
  - **Action:** Integrate into training or remove

#### In `src/rf_detr_finetuning/dataprocessor/io.py`:

- **`get_audio_duration_ms()`** (line 106-124): Exported but minimal usage
  - May be redundant with other audio loading functions
  - **Action:** Review usage in chunking code

#### In `src/rf_detr_finetuning/dataprocessor/normalization.py`:

- **`spectrogram_to_image_array()`** (line 109-145): Used only in experiments

  - Used in `experiments/preprocessing_explorer.py`, `experiments/augmentation_explorer.py`, `experiments/agc_tuner.py`
  - Not used in production pipeline
  - **Action:** Move to experiments utils or document as debugging utility

- **`apply_colormap()`** (line 163-196): Never called

  - Defined but no usage found
  - **Action:** Remove or integrate into visualization pipeline

______________________________________________________________________

### 3. Unused Parameters

#### In `run_pipeline.py`:

- **`test_indices`** variable in split step (line 624)
  - Comment says: "# test_indices not used - splitting handled by image_id mapping"
  - The variable is computed but never used
  - **Action:** Remove the unused variable

______________________________________________________________________

### 4. Commented-Out Code

No significant blocks of commented-out code were found. Some inline comments reference removed legacy code:

#### In `src/rf_detr_finetuning/__init__.py`:

```python
# Legacy modules have been removed. Use the new modular API instead:
# - from rf_detr_finetuning.dataprocessor import AudioChunker, ChunkConfig, TimeBasedFFTConfig
# - from rf_detr_finetuning.dataloader import COCODatasetBuilder, CategoryRegistry
```

These are documentation comments, not dead code.

______________________________________________________________________

### 5. Dead Code in Experiment Scripts

The following experiment scripts contain code that is only used for testing/exploration:

| Script                                   | Purpose                | Status                              |
| ---------------------------------------- | ---------------------- | ----------------------------------- |
| `experiments/preprocessing_explorer.py`  | Visualization tool     | **Active** - Keep for debugging     |
| `experiments/augmentation_explorer.py`   | Augmentation testing   | **Active** - Keep for debugging     |
| `experiments/agc_tuner.py`               | AGC parameter tuning   | **Active** - Keep for tuning        |
| `experiments/batch_experiments.py`       | Batch processing tests | **Review** - Check if still needed  |
| `experiments/visualize_chunk_merging.py` | Merge visualization    | **Active** - Used for documentation |
| `experiments/generate_test_events.py`    | Test data generator    | **Active** - Used for testing merge |

**Recommendation:** These are acceptable as they're in the `experiments/` directory.

______________________________________________________________________

### 6. Incomplete/Placeholder Code

#### In `src/rf_detr_finetuning/audio_preprocessing.py`:

```python
def __repr__(self) -> str:
    """Return a concise string representation of the instance."""  # FIXME
```

- The docstring says FIXME, indicating incomplete implementation
- **Action:** Complete the implementation or remove

______________________________________________________________________

### 7. Unused Imports

The following imports are not directly used in the files (though `annotations` is special):

#### False Positives (Not Actually Dead):

- `from __future__ import annotations` - Used for postponed evaluation of type hints
- Type hint imports like `Optional`, `Union`, etc. - Used in type annotations

#### In test files:

- Test fixtures may have imports only used for type checking
- These are acceptable

______________________________________________________________________

## Recommendations

### High Priority (Remove Soon)

1. **Remove deprecated functions in v0.3.0:**

   - `finetune_model()`
   - `prediction()`
   - `convert_yolo_to_coco()`

2. **Remove unused variable:**

   - `test_indices` in `run_pipeline.py`

### Medium Priority (Review and Decide)

3. **Review and decide on:**
   - `soft_nms()` - Document or remove
   - `mixup_audio()` - Integrate or remove
   - `apply_colormap()` - Integrate or remove
   - `__repr__()` FIXME in audio_preprocessing.py

### Low Priority (Keep for Now)

4. **Experiment scripts** - Keep in experiments directory
5. **Visualization utilities** - Keep for debugging

______________________________________________________________________

## How to Verify

To verify dead code findings:

```bash
# Check if a function is used
grep -r "function_name" --include="*.py" . | grep -v "^.*def " | grep -v "^.*#"

# Example for soft_nms:
grep -r "soft_nms" --include="*.py" . | grep -v "^.*def " | grep -v "^.*#"
# Result: Only appears in definition and exports, no callers

# Example for apply_colormap:
grep -r "apply_colormap" --include="*.py" . | grep -v "^.*def " | grep -v "^.*#"
# Result: Only appears in definition, no callers
```

______________________________________________________________________

## Maintenance Notes

- This analysis was performed using grep and manual code review
- Some "dead" code may be kept intentionally for:
  - Public API compatibility
  - Future features
  - Documentation examples
  - Testing utilities
- Always verify before removing code
- Consider marking dead code with `# TODO: Remove in v0.x.x` comments

______________________________________________________________________

## Files to Review

Based on this analysis, the following files should be reviewed:

1. `src/rf_detr_finetuning/finetune.py` - Contains deprecated `finetune_model()`
2. `src/rf_detr_finetuning/predict.py` - Contains deprecated `prediction()`
3. `src/rf_detr_finetuning/data.py` - Contains deprecated `convert_yolo_to_coco()`
4. `src/rf_detr_finetuning/eventprocessor/merger.py` - Contains unused `soft_nms()`
5. `src/rf_detr_finetuning/dataprocessor/normalization.py` - Contains unused `apply_colormap()`
6. `src/rf_detr_finetuning/dataprocessor/augmentation.py` - Contains unused `mixup_audio()`
7. `run_pipeline.py` - Contains unused `test_indices` variable
