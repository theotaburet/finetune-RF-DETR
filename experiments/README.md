# Audio Processing Experiments Toolkit

This directory contains interactive tools for exploring and tuning audio processing parameters. Use these scripts to experiment with preprocessing, augmentation, and AGC settings before applying them to your training pipeline.

## Overview

The experiment toolkit provides:

- **Visual exploration** of parameter effects on audio/spectrograms
- **Batch processing** across multiple files
- **Configuration export** for use in training
- **Comparison tools** for side-by-side parameter evaluation

## Quick Start

```bash
# Explore preprocessing parameters
python experiments/preprocessing_explorer.py --audio data/sample.flac

# Tune AGC parameters
python experiments/agc_tuner.py --audio data/sample.flac --save-config config/agc.yaml

# Explore augmentations
python experiments/augmentation_explorer.py --audio data/sample.flac

# Run batch experiments on multiple files
python experiments/batch_experiments.py --input data/audio/ --type preprocessing
```

## Scripts

### 1. Preprocessing Explorer (`preprocessing_explorer.py`)

Visual exploration of preprocessing parameters (AGC, detrend, preemphasis).

**Features:**

- Compare multiple preprocessing combinations
- Visualize waveform and spectrogram changes
- Export configurations for training
- Compute RMS and amplitude statistics

**Usage:**

```bash
# Basic usage
python experiments/preprocessing_explorer.py --audio data/sample.flac

# With custom output directory
python experiments/preprocessing_explorer.py \
  --audio data/sample.flac \
  --output experiments/preprocessing_results/

# Use existing config as base
python experiments/preprocessing_explorer.py \
  --audio data/sample.flac \
  --config config/preprocessing.yaml
```

**Output:**

- `{audio_name}_comparison.png` - Side-by-side comparison of all variations
- `{audio_name}_configs.yaml` - All tested configurations
- `spectrograms/` - Individual spectrogram images

### 2. AGC Tuner (`agc_tuner.py`)

Interactive tuning of Automatic Gain Control parameters with recommendations.

**Features:**

- Test multiple target dB levels
- Evaluate max gain settings
- Compare frame sizes
- Automatic recommendations based on results
- Export optimal config for training

**Usage:**

```bash
# Run with defaults
python experiments/agc_tuner.py --audio data/sample.flac

# Custom parameter ranges
python experiments/agc_tuner.py \
  --audio data/sample.flac \
  --target-dbs -30,-25,-20 \
  --max-gains 20,30,40

# Save recommended config for training
python experiments/agc_tuner.py \
  --audio data/sample.flac \
  --save-config config/agc_recommended.yaml
```

**Output:**

- `{audio_name}_agc_spectrograms.png` - Spectrogram comparison
- `{audio_name}_agc_waveforms.png` - Waveform comparison
- `{audio_name}_agc_analysis.yaml` - Detailed analysis
- `agc_recommended.yaml` (if --save-config provided)

**Recommendations:**
The script automatically recommends settings based on:

- Avoiding clipping (max amplitude < 0.99)
- Targeting around -25 dB RMS for consistent levels
- Balancing gain range

### 3. Augmentation Explorer (`augmentation_explorer.py`)

Explore audio and spectrogram augmentation effects.

**Features:**

- Test noise levels with SNR calculation
- Time shift variations
- Gain variations
- SpecAugment (time/frequency masking)
- Multiple random variations per parameter

**Usage:**

```bash
# Explore all augmentation types
python experiments/augmentation_explorer.py --audio data/sample.flac

# Only specific types
python experiments/augmentation_explorer.py \
  --audio data/sample.flac \
  --types noise,gain

# More variations per parameter
python experiments/augmentation_explorer.py \
  --audio data/sample.flac \
  --variations 5
```

**Augmentation Types:**

- `noise` - Gaussian noise at different levels
- `time_shift` - Circular time shifts
- `gain` - Random amplitude scaling
- `spec` - SpecAugment (time/freq masking)

**Output:**

- `{audio_name}_audio_aug.png` - Audio augmentation comparison
- `{audio_name}_spec_aug.png` - Spectrogram augmentation comparison
- `{audio_name}_*_aug.yaml` - Configuration files
- `spectrograms/` - Individual examples

### 4. Batch Experiments (`batch_experiments.py`)

Run experiments across multiple audio files and aggregate statistics.

**Features:**

- Process entire directories
- Statistical aggregation (mean, std)
- Clipping detection
- Success rate tracking
- Rich table output

**Usage:**

```bash
# Preprocessing experiments
python experiments/batch_experiments.py \
  --input data/audio/ \
  --type preprocessing \
  --output output/batch_preprocessing/

# Augmentation experiments
python experiments/batch_experiments.py \
  --input data/audio/ \
  --type augmentation \
  --output output/batch_augmentation/
```

**Output:**

- `*_results.json` - Detailed results with statistics
- Console table with summary

## Experiment Workflow

### 1. Initial Exploration

Start with a representative audio file:

```bash
# Explore preprocessing effects
python experiments/preprocessing_explorer.py \
  --audio data/representative_sample.flac \
  --output experiments/initial_exploration/
```

Review the comparison plot to understand how different preprocessing combinations affect the audio.

### 2. AGC Tuning

Find optimal AGC settings for your dataset:

```bash
python experiments/agc_tuner.py \
  --audio data/representative_sample.flac \
  --target-dbs -30,-25,-20,-15 \
  --save-config config/my_agc.yaml
```

Check the recommendations and adjust based on your dataset characteristics.

### 3. Batch Validation

Validate settings across multiple files:

```bash
python experiments/batch_experiments.py \
  --input data/audio/ \
  --type preprocessing \
  --output experiments/batch_validation/
```

Look for:

- Consistent RMS levels across files
- Low clipping percentage
- Appropriate standard deviation

### 4. Augmentation Selection

Choose augmentation parameters:

```bash
python experiments/augmentation_explorer.py \
  --audio data/sample.flac \
  --variations 5 \
  --output experiments/augmentation_selection/
```

Select parameters that provide good diversity without over-distorting the signal.

### 5. Export to Training Config

Combine all tuned parameters into your training configuration:

```yaml
# config/training.yaml
preprocessing:
  agc:
    enabled: true
    target_db: -25.0  # From agc_tuner.py
    max_gain_db: 30.0
    min_gain_db: -20.0
    frame_size_s: 0.1
  apply_detrend: true
  apply_preemphasis: true

augmentation:
  training:
    noise_level: 0.005  # From augmentation_explorer.py
    time_shift_max: 0.1
    gain_min: 0.8
    gain_max: 1.2
    spec_time_mask: 0.1
    spec_freq_mask: 0.1
```

## Parameter Guidelines

### AGC (Automatic Gain Control)

**Target dB:**

- `-30 dB`: Very conservative, good for quiet recordings
- `-25 dB`: Balanced (recommended starting point)
- `-20 dB`: Aggressive, good for loud environments
- `-15 dB`: Very aggressive, may cause clipping

**Max Gain:**

- `20 dB`: Conservative, less risk of noise amplification
- `30 dB`: Balanced (recommended)
- `40+ dB`: Aggressive, may amplify background noise

**Frame Size:**

- `0.05s`: Fast adaptation, may be unstable
- `0.1s`: Balanced (recommended)
- `0.2s`: Slow adaptation, more stable

### Noise Augmentation

**Noise Level (relative to signal RMS):**

- `0.001`: Subtle (SNR ~60 dB)
- `0.005`: Moderate (SNR ~46 dB) - recommended
- `0.01`: Noticeable (SNR ~40 dB)
- `0.02`: Strong (SNR ~34 dB)

### Time Shift

**Max Shift Ratio:**

- `0.05`: Small shifts (5% of duration)
- `0.1`: Moderate (10%) - recommended
- `0.2`: Large shifts (20%)

### Gain Augmentation

**Gain Range:**

- `0.9 - 1.1`: Subtle variations
- `0.8 - 1.2`: Moderate - recommended
- `0.7 - 1.3`: Strong variations

## Tips

1. **Use representative samples**: Choose audio files that represent your dataset diversity
2. **Check extremes**: Test parameters on both quiet and loud files
3. **Visualize**: Always review the comparison plots before applying settings
4. **Batch validate**: Run batch experiments to ensure consistency
5. **Start conservative**: Begin with milder augmentations and increase gradually
6. **Monitor clipping**: Keep clipping percentage below 1% for AGC
7. **Consider SNR**: For noise augmentation, maintain SNR above 30 dB for training

## Output Directory Structure

```
output/
├── preprocessing_experiments/
│   ├── sample_comparison.png
│   ├── sample_configs.yaml
│   └── spectrograms/
│       ├── sample_none.png
│       ├── sample_agc_-25db.png
│       └── ...
├── agc_tuning/
│   ├── sample_agc_spectrograms.png
│   ├── sample_agc_waveforms.png
│   └── sample_agc_analysis.yaml
├── augmentation_experiments/
│   ├── sample_audio_aug.png
│   ├── sample_spec_aug.png
│   └── spectrograms/
└── batch_experiments/
    └── preprocessing_results.json
```

## Integration with Training

After running experiments, export your optimized configuration:

```bash
# Tune AGC
python experiments/agc_tuner.py \
  --audio data/sample.flac \
  --save-config config/my_experiment_agc.yaml

# Use in training
python run_pipeline.py train --config config/my_experiment_agc.yaml
```

The saved configuration can be used directly with the training pipeline.

## Troubleshooting

**No audio files found:**

- Check file extensions match (.flac, .wav, .mp3)
- Verify input directory path

**All variations look similar:**

- Check that parameters have sufficient range
- Verify audio file has sufficient dynamic range
- Try a more diverse sample

**Too much clipping with AGC:**

- Lower max_gain_db
- Increase target_db (less negative)
- Check if original audio already has high amplitude

**Augmentations too subtle:**

- Increase noise_level for noise augmentation
- Increase max_shift_ratio for time shift
- Widen gain range for gain augmentation
