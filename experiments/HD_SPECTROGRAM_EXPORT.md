# HD Spectrogram Export Configuration

All experiment scripts now export spectrograms as **HD PNG images (300 DPI)** with a consistent **viridis colormap**.

## Export Specifications

- **Format**: PNG
- **Resolution**: 300 DPI (High Definition)
- **Colormap**: Viridis (consistent across all scripts)
- **Size**: 12x8 inches for individual spectrograms
- **Normalization**: Applied for consistent visualization
- **Orientation**: High frequencies at top (flipped spectrogram)

## Scripts Updated

### 1. preprocessing_explorer.py

Exports individual spectrograms for each preprocessing variation:

```python
# HD export with viridis colormap
plt.figure(figsize=(12, 8))
plt.imshow(spec_img, aspect="auto", origin="upper", cmap="viridis")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
```

**Output:**

- `output/preprocessing_experiments/spectrograms/sample_agc_-25db.png`
- Comparison plot: `sample_comparison.png` (300 DPI)

### 2. agc_tuner.py

Exports spectrogram comparison grid:

```python
# HD spectrogram grid with viridis colormap
plt.savefig(output_path, dpi=300, bbox_inches="tight")
```

**Output:**

- `output/agc_tuning/sample_agc_spectrograms.png` (300 DPI)
- Waveform comparison: `sample_agc_waveforms.png` (300 DPI)

### 3. augmentation_explorer.py

Exports augmentation comparison plots:

```python
# HD export with viridis colormap
plt.savefig(output_path, dpi=300, bbox_inches="tight")
```

**Output:**

- `output/augmentation_experiments/sample_audio_aug.png` (300 DPI)
- `output/augmentation_experiments/sample_spec_aug.png` (300 DPI)
- Individual examples: `spectrograms/sample_noise.png` (300 DPI)

## Visual Consistency

All exported spectrograms share:

1. **Same colormap** - Viridis (perceptually uniform, good for publications)
2. **Same resolution** - 300 DPI (print quality)
3. **Same orientation** - High frequencies at top
4. **Same normalization** - Consistent intensity scaling
5. **Same bounding box** - `bbox_inches='tight'` for clean edges

## File Sizes

With 300 DPI, expect:

- Individual spectrograms: ~500KB - 2MB each
- Comparison grids: ~2MB - 5MB each
- Batch exports: Scales with number of variations

## Usage for Publications

The 300 DPI resolution is suitable for:

- Conference papers
- Journal publications
- Posters
- Technical reports

For web usage or quick viewing, you can convert to lower DPI:

```bash
# Convert to 150 DPI for web
convert -density 150 input.png output_web.png

# Convert to 72 DPI for thumbnails
convert -density 72 input.png output_thumb.png
```

## Customization

To change the colormap, edit any of the scripts:

```python
# Change from 'viridis' to another colormap
plt.imshow(spec_img, aspect="auto", origin="upper", cmap="plasma")

# Available colormaps: viridis, plasma, inferno, magma, cividis
# Grayscale: 'gray' or 'Greys'
```

To change DPI:

```python
# Edit the dpi parameter in savefig calls
plt.savefig(output_path, dpi=600)  # Ultra HD
plt.savefig(output_path, dpi=150)  # Web quality
```

## Example Output Structure

```
output/
├── preprocessing_experiments/
│   ├── sample_comparison.png (300 DPI)
│   └── spectrograms/
│       ├── sample_none.png (300 DPI, viridis)
│       ├── sample_agc_-30db.png (300 DPI, viridis)
│       ├── sample_agc_-25db.png (300 DPI, viridis)
│       └── ...
├── agc_tuning/
│   ├── sample_agc_spectrograms.png (300 DPI)
│   └── sample_agc_waveforms.png (300 DPI)
└── augmentation_experiments/
    ├── sample_audio_aug.png (300 DPI)
    ├── sample_spec_aug.png (300 DPI)
    └── spectrograms/
        ├── sample_noise.png (300 DPI, viridis)
        └── ...
```

All spectrograms are exported with consistent quality for reliable comparison and publication-ready visuals.
