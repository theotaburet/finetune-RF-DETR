# Visual Preprocessing Parameter Testing

## Overview

The `test_preprocessing_visual.py` script helps you find the optimal preprocessing parameters by generating side-by-side comparisons of spectrograms with different settings.

## Quick Start

```bash
# Test on a single audio file
python scripts/test_preprocessing_visual.py \
  --audio data_tests/example.wav \
  --output output/preprocessing_test \
  --duration 10.0

# Test on all files in a directory
python scripts/test_preprocessing_visual.py \
  --audio data_tests/ \
  --output output/preprocessing_test \
  --duration 10.0
```

## What It Tests

### AGC (Automatic Gain Control) Settings

- **No preprocessing**: Raw spectrogram
- **AGC -30dB**: Very quiet (conservative)
- **AGC -25dB**: Moderate (recommended default)
- **AGC -20dB**: Loud
- **AGC -25dB + top_db=60**: More compressed dynamic range
- **AGC -25dB + clip_percentile=95**: Aggressive clipping of outliers

### Colormaps

- **Grayscale**: Standard (what RF-DETR sees with default RGB duplication)
- **Magma**: Perceptually uniform, good for identifying intensity levels
- **Viridis**: Another perceptually uniform option
- **Inferno**: High contrast
- **Plasma**: Vibrant colors

## Output

The script generates:

1. **Comparison grid**: `preprocessing_comparison_<filename>.png`

   - All configurations × all colormaps in one image
   - Good for quick overview

2. **Individual high-res images**: `<config>_<colormap>_<filename>.png`

   - One image per configuration/colormap combination
   - Good for detailed inspection

## How to Find the Sweet Spot

### 1. Check for "Burned" Spectrograms

Burned spectrograms have large white/bright areas (clipped values). Look for:

- Too much white = reduce `target_db` or increase `clip_percentile`
- Example: If AGC -20dB looks burned, try AGC -25dB

### 2. Check for Too Dark Spectrograms

Too dark = hard to see features. Look for:

- Mostly dark = increase `target_db` or reduce `top_db`
- Example: If AGC -30dB looks too dark, try AGC -25dB

### 3. Compare Colormaps

- **Grayscale**: Good baseline, but loses subtle intensity differences
- **Magma/Viridis**: Better for seeing intensity gradations
- **Inferno/Plasma**: Higher contrast, but may be distracting

### 4. Iterate

Edit `config/audio_chunking.yaml` and re-run the test:

```yaml
preprocessing:
  agc:
    enabled: true
    target_db: -25.0  # Adjust this
    clip_percentile: 99.0  # Or this
  dynamic_range:
    top_db: 80.0  # Or this
    clip_percentile: 99.0
```

## Key Parameters

### AGC

- **target_db**: Target RMS level

  - -30 to -25 dB: Quiet (conservative, prevents burning)
  - -25 to -20 dB: Moderate
  - -20 to -15 dB: Loud (good for quiet sources)

- **clip_percentile**: Clip values above this percentile

  - 99.0 = clip top 1% (conservative)
  - 95.0 = clip top 5% (aggressive)
  - null = no clipping

### Dynamic Range

- **top_db**: Maximum dynamic range

  - 80 dB: Standard
  - 60 dB: More compressed (shows quieter details)
  - 100 dB: Full range (may appear dark)

- **clip_percentile**: Additional clipping after AGC

  - Same as AGC clip_percentile above

## Tips

1. **Start conservative**: AGC -25dB with clip_percentile=99.0
2. **Test on multiple files**: Different audio types may need different settings
3. **Check both loud and quiet regions**: Make sure both are visible
4. **Consider colormap**: Magma/Viridis can help model distinguish intensities better
5. **Preprocessing is on full file now**: AGC/detrend works on entire 10-minute file, not 6-second chunks

## Example Workflow

```bash
# 1. Test current settings
python scripts/test_preprocessing_visual.py --audio data_tests/example.wav --output output/test1

# 2. Check output/test1/preprocessing_comparison_example.png
# Notice: AGC -20dB looks burned

# 3. Edit config/audio_chunking.yaml, change target_db: -25.0

# 4. Re-test
python scripts/test_preprocessing_visual.py --audio data_tests/example.wav --output output/test2

# 5. Compare output/test1 vs output/test2
# Better! But maybe magma colormap would help...

# 6. Note to self: Consider adding colormap to dataset generation
```

## Integration with Training

Once you find good parameters:

1. Update `config/audio_chunking.yaml`
2. Re-generate your dataset: `python scripts/chunk_audio_dataset.py ...`
3. Train with the new spectrograms

## Colormap Support (Future)

Currently, the training pipeline converts grayscale to RGB by duplicating channels. To use colormaps like magma:

1. Modify `src/rf_detr_finetuning/audio_to_coco.py` to use `apply_colormap()` instead of grayscale duplication
2. Update `scripts/chunk_audio_dataset.py` similarly
3. Re-generate the dataset

This may improve model performance by providing more distinguishable intensity levels.
