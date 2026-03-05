#!/usr/bin/env python3
"""Debug visualization for the audio chunking pipeline.

Picks a representative sound (preferring grouped/multi-event, falling back to
multi-chunk), then produces a multi-panel figure showing every stage of the
spectrogram pipeline alongside the final chunks with bounding boxes.

Usage:
    python debug_chunking_viz.py                       # auto-pick a good example
    python debug_chunking_viz.py --audio path/to.wav   # specific file
    python debug_chunking_viz.py --output-dir debug_output

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_best_example(data_dir: Path) -> tuple[Path, Path] | None:
    """Find a good example sound for visualization.

    Priority:
    1. A grouped sound with multiple distinct labels
    2. A sound long enough to produce multiple chunks (>3.2 s at hop=5ms, target=640)
    3. Any sound with a sidecar JSON

    Args:
        data_dir: Root of downloaded data (e.g. data/downloaded).

    Returns:
        (audio_path, metadata_path) or None.

    """
    candidates_multi_label: list[tuple[Path, Path, int]] = []
    candidates_multi_chunk: list[tuple[Path, Path, float]] = []
    candidates_any: list[tuple[Path, Path]] = []

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            wav_path = json_path.with_suffix(".wav")
            if not wav_path.exists():
                continue

            try:
                meta = json.load(open(json_path))
            except Exception:
                continue

            events = meta.get("events", [])
            grouped_labels = []
            for evt in events:
                grouped_labels.extend(evt.get("grouped_labels", []))

            # Priority 1: multiple distinct labels
            unique_labels = set(grouped_labels)
            if len(unique_labels) > 1:
                candidates_multi_label.append((wav_path, json_path, len(unique_labels)))

            # Priority 2: long enough for multiple chunks
            try:
                info = sf.info(str(wav_path))
                dur = info.duration
            except Exception:
                dur = 0.0

            if dur > 4.0:
                candidates_multi_chunk.append((wav_path, json_path, dur))

            candidates_any.append((wav_path, json_path))

    if candidates_multi_label:
        candidates_multi_label.sort(key=lambda x: -x[2])
        best = candidates_multi_label[0]
        logger.info(f"Found multi-label sound with {best[2]} labels: {best[0].name}")
        return best[0], best[1]

    if candidates_multi_chunk:
        # Pick one around 5-15 s for a nice 2-3 chunk example
        candidates_multi_chunk.sort(key=lambda x: abs(x[2] - 8.0))
        best = candidates_multi_chunk[0]
        logger.info(f"Found multi-chunk sound ({best[2]:.1f}s): {best[0].name}")
        return best[0], best[1]

    if candidates_any:
        return candidates_any[0][0], candidates_any[0][1]

    return None


def run_pipeline_stages(
    audio_path: Path,
    metadata_path: Path | None,
    chunking_config_path: Path,
) -> dict:
    """Run the full preprocessing pipeline and capture every intermediate stage.

    Returns a dict with keys:
        raw_audio, preprocessed_audio, sample_rate,
        raw_spec_power, raw_spec_db, preprocessed_spec_db,
        whitened_spec, drc_spec, final_spec,
        chunks (list of AudioChunk), events, metadata

    """
    # Load config
    from rf_detr_finetuning.dataprocessor.chunker import AudioChunker, load_chunking_config_from_yaml
    from rf_detr_finetuning.dataprocessor.features import (
        compute_mel_spectrogram,
        compute_mel_spectrogram_db,
        flip_spectrogram,
    )
    from rf_detr_finetuning.dataprocessor.io import load_audio_file
    from rf_detr_finetuning.dataprocessor.normalization import normalize_to_range
    from rf_detr_finetuning.dataprocessor.preprocessing import (
        apply_dynamic_range_compression,
        preprocess_audio,
    )

    fft_config, chunk_config, preproc_config = load_chunking_config_from_yaml(chunking_config_path)

    # Load audio
    audio_raw, sr = load_audio_file(audio_path)
    n_fft = fft_config.get_n_fft(sr)
    hop_length = fft_config.get_hop_length(sr)
    if len(audio_raw) < n_fft:
        audio_raw = np.pad(audio_raw, (0, n_fft - len(audio_raw)), mode="constant")

    fmax = sr / 2

    # Stage 1: raw power spectrogram (before any preprocessing, no log)
    raw_spec_power = compute_mel_spectrogram(
        audio_raw, sr, n_mels=fft_config.n_mels, n_fft=n_fft, hop_length=hop_length, f_min=0.0, f_max=fmax
    )

    # Stage 2: raw spectrogram in dB (still before audio preprocessing)
    raw_spec_db = compute_mel_spectrogram_db(
        audio_raw,
        sr,
        n_mels=fft_config.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=fmax,
        min_db=-80.0,
    )

    # Stage 3: preprocess audio (detrend + preemphasis + AGC)
    if preproc_config is not None:
        audio_preprocessed, preproc_meta = preprocess_audio(audio_raw, sr, preproc_config)
    else:
        audio_preprocessed = audio_raw.copy()
        preproc_meta = {}

    # Stage 4: spectrogram of preprocessed audio in dB
    preprocessed_spec_db = compute_mel_spectrogram_db(
        audio_preprocessed,
        sr,
        n_mels=fft_config.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=fmax,
        min_db=-80.0,
    )

    # Stage 5: flip (high freq at top)
    flipped_spec = flip_spectrogram(preprocessed_spec_db)

    # Stage 6: spectral whitening (per-frequency median subtraction)
    if preproc_config is not None and preproc_config.spectral_whitening:
        median_profile = np.median(flipped_spec, axis=1, keepdims=True)
        whitened_spec = flipped_spec - median_profile
    else:
        whitened_spec = flipped_spec.copy()

    # Stage 7: MAD normalization
    if preproc_config is not None and preproc_config.mad_normalization:
        mad = np.median(
            np.abs(whitened_spec - np.median(whitened_spec, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )
        mad = np.maximum(mad, 1e-6)
        mad_spec = whitened_spec / mad
    else:
        mad_spec = whitened_spec

    # Stage 8: dynamic range compression
    if preproc_config is not None:
        drc_spec = apply_dynamic_range_compression(mad_spec, preproc_config.dynamic_range)
    else:
        drc_spec = mad_spec.copy()

    # Stage 9: normalize to 0-255
    final_spec = normalize_to_range(drc_spec, 0.0, 255.0).astype(np.uint8)

    # Parse metadata
    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.load(open(metadata_path))

    # Now run the full chunker to get actual chunks with bboxes
    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preproc_config,
    )
    chunks = chunker.chunk_audio_file(audio_path, metadata_path)

    return {
        "audio_path": audio_path,
        "raw_audio": audio_raw,
        "preprocessed_audio": audio_preprocessed,
        "sample_rate": sr,
        "preproc_meta": preproc_meta,
        "raw_spec_power": raw_spec_power,
        "raw_spec_db": raw_spec_db,
        "preprocessed_spec_db": preprocessed_spec_db,
        "flipped_spec": flipped_spec,
        "whitened_spec": whitened_spec,
        "mad_spec": mad_spec,
        "drc_spec": drc_spec,
        "final_spec": final_spec,
        "chunks": chunks,
        "metadata": metadata,
        "fft_config": fft_config,
        "chunk_config": chunk_config,
        "preproc_config": preproc_config,
    }


def plot_pipeline_stages(stages: dict, output_dir: Path) -> None:
    """Create a multi-panel figure showing every pipeline stage."""
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = stages["audio_path"]
    sr = stages["sample_rate"]
    dur_s = len(stages["raw_audio"]) / sr

    # Figure 1: Full pipeline stages (6 panels, vertical)
    preproc_config = stages.get("preproc_config")
    whitening_on = preproc_config is not None and preproc_config.spectral_whitening
    mad_on = preproc_config is not None and preproc_config.mad_normalization

    stage_data = [
        ("1. Raw mel spectrogram (power, no log)", stages["raw_spec_power"]),
        ("2. Raw mel spectrogram (dB scale: 10*log10)", stages["raw_spec_db"]),
        ("3. After audio preproc (detrend+preemph+AGC) -> dB", stages["preprocessed_spec_db"]),
        (
            f"4. Flipped + spectral whitening {'[ON]' if whitening_on else '[OFF]'}",
            stages["whitened_spec"],
        ),
        (
            f"5. MAD normalization {'[ON]' if mad_on else '[OFF]'} + DRC",
            stages["drc_spec"],
        ),
        ("6. Final normalized (0-255, uint8)", stages["final_spec"]),
    ]

    n_panels = len(stage_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels))
    fig.suptitle(
        f"Pipeline stages: {audio_path.name}\n"
        f"Duration: {dur_s:.2f}s | SR: {sr} Hz | "
        f"FFT: {stages['fft_config'].fft_ms}ms | Hop: {stages['fft_config'].hop_ms}ms | "
        f"Mels: {stages['fft_config'].n_mels}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (title, spec) in zip(axes, stage_data):
        im = ax.imshow(
            spec if spec.shape[0] < spec.shape[1] else spec,
            aspect="auto",
            origin="upper",
            cmap="magma",
        )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    axes[-1].set_xlabel("Time frame")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"{audio_path.stem}_pipeline_stages.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved pipeline stages: {path}")

    # Figure 2: Chunks with bounding boxes
    chunks = stages["chunks"]
    if not chunks:
        logger.warning("No chunks produced — skipping chunk figure.")
        return

    n_chunks = len(chunks)
    fig, axes = plt.subplots(1, n_chunks, figsize=(6 * n_chunks, 7), squeeze=False)
    axes = axes[0]

    metadata = stages["metadata"]
    events_info = metadata.get("events", [])
    grouped_labels = []
    for evt in events_info:
        grouped_labels.extend(evt.get("grouped_labels", []))
    unique_labels = sorted(set(grouped_labels)) if grouped_labels else ["(no labels)"]

    fig.suptitle(
        f"Chunks: {audio_path.name}\n"
        f"{n_chunks} chunk(s) | Window: {stages['chunk_config'].window_duration_ms:.0f}ms | "
        f"Overlap: {stages['chunk_config'].overlap_ratio:.0%} | "
        f"Labels: {', '.join(unique_labels)}",
        fontsize=12,
        fontweight="bold",
    )

    # Color map for categories
    colors = plt.cm.Set1(np.linspace(0, 1, max(10, len(unique_labels))))

    for i, (ax, chunk) in enumerate(zip(axes, chunks)):
        spec = chunk.spectrogram
        if spec is None:
            ax.set_title(f"Chunk {i}: NO SPECTROGRAM")
            continue

        # Display spectrogram
        if spec.ndim == 3:
            ax.imshow(spec, aspect="auto", origin="upper")
        else:
            ax.imshow(spec, aspect="auto", origin="upper", cmap="magma")

        # Draw bboxes
        for bbox in chunk.bboxes:
            x1, y1, x2, y2 = bbox.to_xyxy()
            cat_idx = hash(bbox.category) % len(colors)
            color = colors[cat_idx]
            rect = mpatches.FancyBboxPatch(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                boxstyle="round,pad=0",
            )
            ax.add_patch(rect)
            ax.text(
                x1 + 2,
                y1 - 4,
                f"{bbox.category} ({bbox.overlap_ratio:.0%})",
                fontsize=8,
                color="white",
                backgroundcolor=(float(color[0]), float(color[1]), float(color[2]), 0.6),
            )

        padded_str = f" [padded {chunk.padding_amount_ms:.0f}ms]" if chunk.is_padded else ""
        ax.set_title(
            f"Chunk {i}: {chunk.start_ms:.0f}-{chunk.end_ms:.0f}ms\n{len(chunk.bboxes)} bbox(es){padded_str}",
            fontsize=10,
        )
        ax.set_xlabel("Time (px)")
        ax.set_ylabel("Frequency (px)")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = output_dir / f"{audio_path.stem}_chunks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chunk visualization: {path}")

    # Figure 3: Preprocessing statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Preprocessing diagnostics", fontsize=13, fontweight="bold")

    # Panel 1: Raw vs preprocessed waveform
    ax = axes[0, 0]
    raw = stages["raw_audio"]
    pre = stages["preprocessed_audio"]
    t = np.arange(len(raw)) / sr
    ax.plot(t, raw, alpha=0.5, label="Raw", linewidth=0.5)
    ax.plot(t, pre, alpha=0.7, label="Preprocessed", linewidth=0.5)
    ax.set_title("Waveform: raw vs preprocessed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

    # Panel 2: Histogram of spectrogram values at each stage
    ax = axes[0, 1]
    for label, spec in [
        ("Raw power", stages["raw_spec_power"]),
        ("dB scale", stages["raw_spec_db"]),
        ("After whitening", stages["whitened_spec"]),
        ("After DRC", stages["drc_spec"]),
        ("Final uint8", stages["final_spec"]),
    ]:
        vals = spec.flatten()
        # Subsample for speed
        if len(vals) > 50000:
            vals = np.random.default_rng(42).choice(vals, 50000, replace=False)
        ax.hist(vals, bins=100, alpha=0.5, label=label, density=True)
    ax.set_title("Value distributions across stages")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Panel 3: Mean spectrum (frequency profile)
    ax = axes[1, 0]
    final = stages["final_spec"].astype(float)
    mean_profile = final.mean(axis=1)
    ax.plot(mean_profile)
    ax.set_title("Mean intensity per mel bin (final spectrogram)")
    ax.set_xlabel("Mel bin (0 = high freq, top of image)")
    ax.set_ylabel("Mean pixel value")
    ax.axhline(
        y=mean_profile.mean(), color="r", linestyle="--", alpha=0.5, label=f"Global mean: {mean_profile.mean():.1f}"
    )
    ax.legend(fontsize=8)

    # Panel 4: Preprocessing metadata
    ax = axes[1, 1]
    ax.axis("off")
    preproc_meta = stages.get("preproc_meta", {})
    cfg = stages.get("preproc_config")
    lines = [
        f"File: {audio_path.name}",
        f"Duration: {dur_s:.2f}s",
        f"Sample rate: {sr} Hz",
        "",
        "Preprocessing config:",
        f"  Detrend: {cfg.detrend if cfg else 'N/A'}",
        f"  Preemphasis: {cfg.preemphasis if cfg else 'N/A'}",
        f"  Spectral whitening: {cfg.spectral_whitening if cfg else 'N/A'}",
        f"  MAD normalization: {cfg.mad_normalization if cfg else 'N/A'}",
        f"  AGC enabled: {cfg.agc.enabled if cfg else 'N/A'}",
        f"  AGC target_db: {cfg.agc.target_db if cfg else 'N/A'}",
        f"  AGC max_gain: {cfg.agc.max_gain_db if cfg else 'N/A'}dB",
        f"  DRC top_db: {cfg.dynamic_range.top_db if cfg else 'N/A'}",
        f"  DRC clip_percentile: {cfg.dynamic_range.clip_percentile if cfg else 'N/A'}",
        "",
        "Spectrogram conversion: dB scale (10*log10)",
        "",
        "Preprocessing results:",
        f"  Original RMS: {preproc_meta.get('original_rms_db', 'N/A'):.1f} dB"
        if isinstance(preproc_meta.get("original_rms_db"), int | float)
        else "  Original RMS: N/A",
        f"  Final RMS: {preproc_meta.get('final_rms_db', 'N/A'):.1f} dB"
        if isinstance(preproc_meta.get("final_rms_db"), int | float)
        else "  Final RMS: N/A",
        f"  Gain applied: {preproc_meta.get('gain_applied_db', 'N/A'):.1f} dB"
        if isinstance(preproc_meta.get("gain_applied_db"), int | float)
        else "  Gain applied: N/A",
        "",
        f"Spectrogram shape: {stages['final_spec'].shape}",
        f"Value range: [{stages['final_spec'].min()}, {stages['final_spec'].max()}]",
        "",
        f"Chunks: {len(stages['chunks'])}",
        f"Labels: {', '.join(unique_labels)}",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / f"{audio_path.stem}_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved diagnostics: {path}")


def main() -> None:
    """Run debug visualization for the audio chunking pipeline."""
    parser = argparse.ArgumentParser(description="Debug visualization for audio chunking pipeline")
    parser.add_argument("--audio", type=Path, help="Specific audio file to visualize")
    parser.add_argument("--metadata", type=Path, help="Sidecar JSON (auto-detected if omitted)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/downloaded"), help="Downloaded data root")
    parser.add_argument("--chunking-config", type=Path, default=Path("config/audio_chunking.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/debug_viz"))
    args = parser.parse_args()

    if args.audio:
        audio_path = args.audio
        metadata_path = args.metadata or audio_path.with_suffix(".json")
        if not metadata_path.exists():
            metadata_path = None
    else:
        result = find_best_example(args.data_dir)
        if result is None:
            logger.error(f"No audio files found in {args.data_dir}")
            sys.exit(1)
        audio_path, metadata_path = result

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Audio: {audio_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Config: {args.chunking_config}")

    stages = run_pipeline_stages(audio_path, metadata_path, args.chunking_config)
    plot_pipeline_stages(stages, args.output_dir)

    logger.info(f"All debug figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
