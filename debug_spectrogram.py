#!/usr/bin/env python3
"""Debug spectrogram visualization: shows exactly what the network sees.

Reads the same YAML config and runs the same preprocessing pipeline as training/inference,
then displays the spectrogram at each stage so you can verify the processing is correct.

Usage:
    python debug_spectrogram.py path/to/audio.wav
    python debug_spectrogram.py path/to/audio.wav --config config/audio_chunking.yaml
    python debug_spectrogram.py path/to/audio.wav --chunk-index 2
    python debug_spectrogram.py path/to/audio.wav --save output.png

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/audio_chunking.yaml")


def load_config(config_path: Path) -> dict:
    """Load the full YAML config and return all sections.

    Args:
        config_path: Path to audio_chunking.yaml

    Returns:
        Dict with keys: fft_config, chunk_config, preproc_config, spectrogram_section, raw_yaml

    """
    import yaml

    from rf_detr_finetuning.dataprocessor.chunker import load_chunking_config_from_yaml

    fft_config, chunk_config, preproc_config, spec_config = load_chunking_config_from_yaml(config_path)

    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)

    spectrogram_section = raw_yaml.get("spectrogram", {})

    return {
        "fft_config": fft_config,
        "chunk_config": chunk_config,
        "preproc_config": preproc_config,
        "spectrogram_section": spectrogram_section,
        "raw_yaml": raw_yaml,
    }


def run_pipeline_stages(
    audio_path: Path,
    config: dict,
) -> dict:
    """Run the full spectrogram pipeline stage-by-stage, matching chunker.py exactly.

    Returns a dict with intermediate results at each stage for visualization.

    """
    from rf_detr_finetuning.dataprocessor.chunker import AudioChunker
    from rf_detr_finetuning.dataprocessor.features import (
        compute_mel_spectrogram,
        compute_mel_spectrogram_db,
        flip_spectrogram,
    )
    from rf_detr_finetuning.dataprocessor.io import load_audio_file
    from rf_detr_finetuning.dataprocessor.normalization import grayscale_to_rgb, normalize_to_range
    from rf_detr_finetuning.dataprocessor.preprocessing import (
        apply_dynamic_range_compression,
        preprocess_audio,
    )

    fft_config = config["fft_config"]
    chunk_config = config["chunk_config"]
    preproc_config = config["preproc_config"]

    # Load audio
    audio_raw, sr = load_audio_file(audio_path)
    n_fft = fft_config.get_n_fft(sr)
    hop_length = fft_config.get_hop_length(sr)
    fmax = sr / 2

    # Pad if needed (same as chunker.py:270)
    if len(audio_raw) < n_fft:
        audio_raw = np.pad(audio_raw, (0, n_fft - len(audio_raw)), mode="constant")

    logger.info(
        f"Audio: {audio_path.name}, sr={sr}, duration={len(audio_raw) / sr:.2f}s, n_fft={n_fft}, hop={hop_length}"
    )

    stages = {}

    # Stage 1: Raw power spectrogram (no preprocessing, no log)
    stages["1_raw_power"] = compute_mel_spectrogram(
        audio_raw,
        sr,
        n_mels=fft_config.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=fmax,
    )

    # Stage 2: Audio preprocessing (detrend + preemphasis + AGC)
    if preproc_config is not None:
        audio_processed, preproc_meta = preprocess_audio(audio_raw, sr, preproc_config)
        logger.info(f"Preprocessing: {preproc_meta}")
    else:
        audio_processed = audio_raw.copy()
        preproc_meta = {}
    stages["preproc_meta"] = preproc_meta

    # Stage 3: Mel spectrogram in dB (matches chunker._compute_spectrogram)
    spec_db = compute_mel_spectrogram_db(
        audio_processed,
        sr,
        n_mels=fft_config.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=fmax,
        min_db=-80.0,
    )
    stages["3_spec_db"] = spec_db
    logger.info(f"Spec dB range: [{spec_db.min():.1f}, {spec_db.max():.1f}]")

    # Stage 4: Spectral whitening (matches chunker.py:135-137)
    if preproc_config is not None and preproc_config.spectral_whitening:
        median_profile = np.median(spec_db, axis=1, keepdims=True)
        spec_whitened = spec_db - median_profile
        logger.info(f"Whitened range: [{spec_whitened.min():.1f}, {spec_whitened.max():.1f}]")
    else:
        spec_whitened = spec_db.copy()
    stages["4_whitened"] = spec_whitened

    # Stage 5: MAD normalization (matches chunker.py:140-143)
    if preproc_config is not None and preproc_config.mad_normalization:
        mad = np.median(
            np.abs(spec_whitened - np.median(spec_whitened, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )
        mad = np.maximum(mad, 1e-6)
        spec_mad = spec_whitened / mad
        logger.info(f"MAD-normalized range: [{spec_mad.min():.1f}, {spec_mad.max():.1f}]")
    else:
        spec_mad = spec_whitened
    stages["5_mad"] = spec_mad

    # Stage 6: Flip (high freq at top, matches chunker.py:194)
    spec_flipped = flip_spectrogram(spec_mad)
    stages["6_flipped"] = spec_flipped

    # Stage 7: Dynamic range compression + normalize to uint8 (matches chunker.py:198)
    if preproc_config is not None:
        spec_drc = apply_dynamic_range_compression(spec_flipped, preproc_config.dynamic_range)
        spec_uint8 = normalize_to_range(spec_drc, 0.0, 255.0).astype(np.uint8)
    else:
        spec_uint8 = normalize_to_range(spec_flipped, 0.0, 255.0).astype(np.uint8)
    stages["7_final_uint8"] = spec_uint8
    logger.info(f"Final uint8 range: [{spec_uint8.min()}, {spec_uint8.max()}]")

    # Stage 8: Convert to RGB (matches run_pipeline.py:769)
    spec_rgb = grayscale_to_rgb(spec_uint8)
    stages["8_rgb"] = spec_rgb

    # Also run the actual chunker to get the real chunks with resizing
    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preproc_config,
    )
    chunks = chunker.chunk_audio_file(audio_path)
    stages["chunks"] = chunks

    stages["sample_rate"] = sr
    stages["audio_raw"] = audio_raw
    stages["audio_processed"] = audio_processed

    return stages


def plot_stages(stages: dict, audio_path: Path, config: dict, chunk_index: int | None = None) -> plt.Figure:
    """Create a multi-panel figure showing each pipeline stage.

    Args:
        stages: Dict from run_pipeline_stages()
        audio_path: Path to audio file (for title)
        config: Config dict (for annotation)
        chunk_index: If set, also show the specific chunk as the network sees it

    Returns:
        matplotlib Figure

    """
    fft_config = config["fft_config"]
    preproc_config = config["preproc_config"]
    chunks = stages["chunks"]

    # Determine which stages to show
    panels = [
        ("1. Raw power spectrogram", stages["1_raw_power"], "viridis", False),
        ("3. After dB conversion (+ preprocess)", stages["3_spec_db"], "inferno", False),
    ]

    if preproc_config is not None and preproc_config.spectral_whitening:
        panels.append(("4. After spectral whitening", stages["4_whitened"], "inferno", False))

    if preproc_config is not None and preproc_config.mad_normalization:
        panels.append(("5. After MAD normalization", stages["5_mad"], "inferno", False))

    panels.append(("6. Flipped (high freq at top)", stages["6_flipped"], "inferno", False))
    panels.append(("7. Final uint8 [0-255] (what is saved to PNG)", stages["7_final_uint8"], "gray", True))

    # Add specific chunk view if requested
    show_chunk = None
    if chunk_index is not None and chunk_index < len(chunks):
        show_chunk = chunks[chunk_index]
    elif chunks:
        show_chunk = chunks[0]
        chunk_index = 0

    if show_chunk is not None:
        panels.append(
            (
                f"8. Chunk {chunk_index} ({show_chunk.start_ms:.0f}-{show_chunk.end_ms:.0f}ms) - NETWORK INPUT",
                show_chunk.spectrogram,
                "gray",
                True,
            )
        )

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    sr = stages["sample_rate"]

    for ax, (title, data, cmap, is_uint8) in zip(axes, panels):
        if data.ndim == 3:
            # RGB image - show as-is
            ax.imshow(data)
        else:
            im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

        vmin, vmax = float(data.min()), float(data.max())
        mean_val = float(data.mean())
        std_val = float(data.std())
        shape_str = f"{data.shape[0]}x{data.shape[1]}"

        stats = f"range=[{vmin:.1f}, {vmax:.1f}]  mean={mean_val:.1f}  std={std_val:.1f}  shape={shape_str}"
        ax.set_title(f"{title}\n{stats}", fontsize=10, loc="left")
        ax.set_ylabel("Freq bin")
        ax.set_xlabel("Time frame")

    # Add config summary as suptitle
    config_summary = (
        f"{audio_path.name}  |  sr={sr}  |  "
        f"fft={fft_config.fft_ms}ms  hop={fft_config.hop_ms}ms  n_mels={fft_config.n_mels}  |  "
        f"whitening={'ON' if preproc_config and preproc_config.spectral_whitening else 'OFF'}  "
        f"MAD={'ON' if preproc_config and preproc_config.mad_normalization else 'OFF'}  "
        f"AGC={'ON' if preproc_config and preproc_config.agc.enabled else 'OFF'}  "
        f"preemphasis={preproc_config.preemphasis if preproc_config else 0}"
    )
    fig.suptitle(config_summary, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    return fig


def plot_histogram(stages: dict) -> plt.Figure:
    """Plot value distribution histograms at key stages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    hist_data = [
        ("Raw power", stages["1_raw_power"].ravel()),
        ("After dB conversion", stages["3_spec_db"].ravel()),
        ("After whitening", stages["4_whitened"].ravel()),
        ("Final uint8", stages["7_final_uint8"].ravel().astype(float)),
    ]

    for ax, (label, data) in zip(axes.ravel(), hist_data):
        # Remove infinities and NaNs for histogram
        clean = data[np.isfinite(data)]
        if len(clean) == 0:
            ax.set_title(f"{label} - NO FINITE VALUES", fontsize=10)
            continue

        ax.hist(clean, bins=200, color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_title(f"{label}  (min={clean.min():.2f}, max={clean.max():.2f}, std={clean.std():.2f})", fontsize=10)
        ax.set_ylabel("Count")
        ax.axvline(np.median(clean), color="red", linestyle="--", alpha=0.7, label=f"median={np.median(clean):.2f}")
        ax.legend(fontsize=8)

    fig.suptitle("Value distributions at each pipeline stage", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_chunks_grid(stages: dict, max_chunks: int = 12) -> plt.Figure | None:
    """Show all chunks in a grid, exactly as the network would see them."""
    chunks = stages["chunks"]
    if not chunks:
        return None

    n = min(len(chunks), max_chunks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            chunk = chunks[i]
            spec = chunk.spectrogram
            if spec.ndim == 3:
                ax.imshow(spec)
            else:
                ax.imshow(spec, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"Chunk {i}: {chunk.start_ms:.0f}-{chunk.end_ms:.0f}ms\nshape={spec.shape}, bboxes={len(chunk.bboxes)}",
                fontsize=9,
            )
        ax.axis("off")

    fig.suptitle(f"All chunks as network input ({len(chunks)} total)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    """Main entry point for spectrogram debugging CLI.

    Parses command-line arguments and launches visualization of the audio spectrogram.

    Args:
        None

    Returns:
        None

    """
    parser = argparse.ArgumentParser(
        description="Debug spectrogram visualization: shows exactly what the network sees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to audio file (.wav, .flac, .mp3)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to audio_chunking.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Show a specific chunk index in detail (default: first chunk)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to file instead of displaying (e.g. output.png)",
    )
    parser.add_argument("--no-histograms", action="store_true", help="Skip histogram plots")
    parser.add_argument("--no-chunks-grid", action="store_true", help="Skip chunks grid plot")
    args = parser.parse_args()

    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    logger.info(f"Config: {args.config}")
    config = load_config(args.config)

    # Run pipeline
    logger.info(f"Processing: {args.audio}")
    stages = run_pipeline_stages(args.audio, config)

    # Print summary
    chunks = stages["chunks"]
    logger.info(f"Generated {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        logger.info(
            f"  Chunk {i}: {c.start_ms:.0f}-{c.end_ms:.0f}ms, "
            f"shape={c.spectrogram.shape}, bboxes={len(c.bboxes)}, "
            f"padded={c.is_padded}"
        )

    # Generate plots
    save_dir = args.save.parent if args.save else Path(".")
    stem = args.save.stem if args.save else args.audio.stem

    fig_stages = plot_stages(stages, args.audio, config, chunk_index=args.chunk_index)
    stages_path = save_dir / f"{stem}_stages.png"
    fig_stages.savefig(stages_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved pipeline stages: {stages_path}")
    plt.close(fig_stages)

    if not args.no_histograms:
        fig_hist = plot_histogram(stages)
        hist_path = save_dir / f"{stem}_histograms.png"
        fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved histograms: {hist_path}")
        plt.close(fig_hist)

    if not args.no_chunks_grid:
        fig_grid = plot_chunks_grid(stages)
        if fig_grid is not None:
            grid_path = save_dir / f"{stem}_chunks.png"
            fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved chunks grid: {grid_path}")
            plt.close(fig_grid)

    logger.info("Done.")


if __name__ == "__main__":
    main()
