#!/usr/bin/env python3
"""Merging logic visualization with synthetic audio events.

Generates synthetic audio containing known events at specific times/frequencies,
creates spectrogram chunks with overlapping windows, simulates detections in
each chunk, runs the ClassWiseMerger, and visualizes the full pipeline with
matplotlib.

Usage:
    python examples/merging_visualization.py
    python examples/merging_visualization.py --output output/merging_demo

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
    ClassWiseMerger,
)


@dataclass
class SyntheticEvent:
    """Definition of a synthetic audio event."""

    start_ms: float
    end_ms: float
    freq_min_hz: float
    freq_max_hz: float
    class_id: int
    class_name: str
    amplitude: float = 0.5


# Color palette for classes
CLASS_COLORS = {
    0: "#2ecc71",  # Green
    1: "#e74c3c",  # Red
    2: "#3498db",  # Blue
    3: "#f39c12",  # Orange
}
DETECTION_ALPHA = 0.25
MERGED_ALPHA = 0.35


def generate_synthetic_audio(
    duration_ms: float,
    sample_rate: int,
    events: list[SyntheticEvent],
) -> np.ndarray:
    """Generate synthetic audio with tonal events at specific times/frequencies.

    Args:
        duration_ms: Total audio duration in milliseconds.
        sample_rate: Sample rate in Hz.
        events: List of synthetic events to embed.

    Returns:
        Audio waveform as 1D numpy array.

    """
    n_samples = int(duration_ms / 1000 * sample_rate)
    audio = np.random.randn(n_samples) * 0.01  # Low-level background noise

    for event in events:
        start_sample = int(event.start_ms / 1000 * sample_rate)
        end_sample = int(event.end_ms / 1000 * sample_rate)
        end_sample = min(end_sample, n_samples)

        t = np.arange(end_sample - start_sample) / sample_rate
        bandwidth = event.freq_max_hz - event.freq_min_hz

        # Generate harmonic content spread across the frequency band
        signal = np.zeros_like(t)
        n_harmonics = max(1, int(bandwidth / 200))
        for i in range(n_harmonics):
            freq = event.freq_min_hz + bandwidth * (i + 0.5) / n_harmonics
            signal += np.sin(2 * np.pi * freq * t) * event.amplitude / n_harmonics

        # Apply fade in/out to avoid clicks
        fade_samples = min(int(0.01 * sample_rate), len(t) // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            signal[:fade_samples] *= fade_in
            signal[-fade_samples:] *= fade_out

        audio[start_sample:end_sample] += signal

    return audio


def compute_simple_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a simple STFT-based spectrogram.

    Args:
        audio: Audio waveform (1D).
        sample_rate: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        Tuple of (spectrogram_db, time_axis_ms, freq_axis_hz).

    """
    # Manual STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    window = np.hanning(n_fft)
    spec = np.zeros((n_fft // 2 + 1, n_frames))

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft] * window
        fft_result = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft_result)

    # Convert to dB
    spec_db = 20 * np.log10(spec + 1e-10)

    time_axis = np.arange(n_frames) * hop_length / sample_rate * 1000
    freq_axis = np.arange(n_fft // 2 + 1) * sample_rate / n_fft

    return spec_db, time_axis, freq_axis


def simulate_chunk_detections(
    events: list[SyntheticEvent],
    chunk_start_ms: float,
    chunk_end_ms: float,
    window_index: int,
    noise_ms: float = 50.0,
    confidence_range: tuple[float, float] = (0.6, 0.95),
) -> list[AudioEvent]:
    """Simulate detections that a model would produce for a chunk.

    Each ground truth event overlapping the chunk produces a detection with
    slight noise in the temporal boundaries and a random confidence score.

    Args:
        events: Ground truth synthetic events.
        chunk_start_ms: Chunk start time in ms.
        chunk_end_ms: Chunk end time in ms.
        window_index: Index of this chunk window.
        noise_ms: Random noise added to temporal boundaries.
        confidence_range: Range of simulated confidence scores.

    Returns:
        List of simulated AudioEvent detections.

    """
    rng = np.random.default_rng(seed=42 + window_index)
    detections = []

    for event in events:
        # Check overlap
        overlap_start = max(event.start_ms, chunk_start_ms)
        overlap_end = min(event.end_ms, chunk_end_ms)
        if overlap_end <= overlap_start:
            continue

        # Require minimum overlap
        overlap_ratio = (overlap_end - overlap_start) / (event.end_ms - event.start_ms)
        if overlap_ratio < 0.15:
            continue

        # Add noise to boundaries (simulating imprecise model)
        det_start = overlap_start + rng.uniform(-noise_ms, noise_ms)
        det_end = overlap_end + rng.uniform(-noise_ms, noise_ms)
        det_start = max(chunk_start_ms, det_start)
        det_end = min(chunk_end_ms, det_end)

        if det_end <= det_start:
            continue

        score = rng.uniform(*confidence_range)

        detections.append(
            AudioEvent(
                start_ms=det_start,
                end_ms=det_end,
                class_id=event.class_id,
                class_name=event.class_name,
                score=score,
                min_freq_hz=event.freq_min_hz + rng.uniform(-50, 50),
                max_freq_hz=event.freq_max_hz + rng.uniform(-50, 50),
                source_windows=[window_index],
            )
        )

    return detections


def plot_merging_pipeline(
    events: list[SyntheticEvent],
    spec_db: np.ndarray,
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    chunk_boundaries: list[tuple[float, float]],
    per_chunk_detections: list[list[AudioEvent]],
    merged_events: EventList,
    output_path: Path | None = None,
) -> None:
    """Create a 4-panel visualization of the merging pipeline.

    Panel 1: Full spectrogram with ground truth events
    Panel 2: Chunk boundaries overlaid on spectrogram
    Panel 3: Per-chunk detections (before merging)
    Panel 4: Merged events (after ClassWiseMerger)

    Args:
        events: Ground truth synthetic events.
        spec_db: Spectrogram in dB (freq x time).
        time_axis: Time axis in ms.
        freq_axis: Frequency axis in Hz.
        chunk_boundaries: List of (start_ms, end_ms) for each chunk.
        per_chunk_detections: Detections per chunk before merging.
        merged_events: Final merged events.
        output_path: Path to save the figure (optional).

    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    spec_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": "magma",
        "extent": [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    }

    # Panel 1: Ground truth events
    ax = axes[0]
    ax.imshow(spec_db, **spec_kwargs)
    ax.set_title("Ground Truth Events", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)")
    for event in events:
        color = CLASS_COLORS.get(event.class_id, "#ffffff")
        rect = mpatches.FancyBboxPatch(
            (event.start_ms, event.freq_min_hz),
            event.end_ms - event.start_ms,
            event.freq_max_hz - event.freq_min_hz,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.3,
            boxstyle="round,pad=0",
        )
        ax.add_patch(rect)
        ax.text(
            event.start_ms + 10,
            event.freq_max_hz + 50,
            f"{event.class_name}",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    # Panel 2: Chunk boundaries
    ax = axes[1]
    ax.imshow(spec_db, **spec_kwargs)
    ax.set_title("Overlapping Chunk Windows", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)")
    chunk_colors = plt.cm.Set2(np.linspace(0, 1, len(chunk_boundaries)))
    for i, (cs, ce) in enumerate(chunk_boundaries):
        ax.axvspan(cs, ce, alpha=0.15, color=chunk_colors[i])
        ax.axvline(cs, color=chunk_colors[i], linewidth=1, linestyle="--", alpha=0.7)
        ax.text(
            cs + 20,
            freq_axis[-1] * 0.95,
            f"W{i}",
            color=chunk_colors[i],
            fontsize=9,
            fontweight="bold",
            va="top",
        )

    # Panel 3: Per-chunk detections (before merge)
    ax = axes[2]
    ax.imshow(spec_db, **spec_kwargs)
    ax.set_title("Per-Window Detections (Before Merge)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)")
    all_dets = []
    for dets in per_chunk_detections:
        all_dets.extend(dets)
    for det in all_dets:
        color = CLASS_COLORS.get(det.class_id, "#ffffff")
        rect = mpatches.FancyBboxPatch(
            (det.start_ms, det.min_freq_hz or 0),
            det.end_ms - det.start_ms,
            (det.max_freq_hz or freq_axis[-1]) - (det.min_freq_hz or 0),
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=DETECTION_ALPHA,
            linestyle="--",
            boxstyle="round,pad=0",
        )
        ax.add_patch(rect)
    ax.text(
        time_axis[0] + 20,
        freq_axis[-1] * 0.95,
        f"{len(all_dets)} detections",
        color="white",
        fontsize=10,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.6},
    )

    # Panel 4: Merged events
    ax = axes[3]
    ax.imshow(spec_db, **spec_kwargs)
    ax.set_title("After ClassWiseMerger", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    for event in merged_events:
        color = CLASS_COLORS.get(event.class_id, "#ffffff")
        rect = mpatches.FancyBboxPatch(
            (event.start_ms, event.min_freq_hz or 0),
            event.end_ms - event.start_ms,
            (event.max_freq_hz or freq_axis[-1]) - (event.min_freq_hz or 0),
            linewidth=2.5,
            edgecolor=color,
            facecolor=color,
            alpha=MERGED_ALPHA,
            boxstyle="round,pad=0",
        )
        ax.add_patch(rect)
        label = f"{event.class_name} ({event.score:.2f})"
        merged_count = event.metadata.get("merged_count", 1)
        if merged_count > 1:
            label += f" [{merged_count} merged]"
        ax.text(
            event.start_ms + 10,
            (event.max_freq_hz or freq_axis[-1]) + 50,
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
        )
    ax.text(
        time_axis[0] + 20,
        freq_axis[-1] * 0.95,
        f"{len(merged_events)} events",
        color="white",
        fontsize=10,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.6},
    )

    # Add legend
    legend_patches = [mpatches.Patch(color=c, alpha=0.5, label=f"Class {i}") for i, c in CLASS_COLORS.items()]
    axes[0].legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.7,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


def main(output_dir: str | None = None) -> None:
    """Run the full merging visualization demo.

    Args:
        output_dir: Optional directory to save output figures.

    """
    # Configuration
    sample_rate = 16000
    duration_ms = 15000.0  # 15 seconds
    window_ms = 3200.0
    overlap_ratio = 0.2
    stride_ms = window_ms * (1 - overlap_ratio)

    # Define synthetic events
    # Two whale calls (class 0) that should be merged
    # One ship noise (class 1) spanning a wide time range
    # One click train (class 2) - short burst
    ground_truth_events = [
        SyntheticEvent(
            start_ms=1000,
            end_ms=3500,
            freq_min_hz=200,
            freq_max_hz=800,
            class_id=0,
            class_name="whale_call",
            amplitude=0.6,
        ),
        SyntheticEvent(
            start_ms=4000,
            end_ms=6500,
            freq_min_hz=200,
            freq_max_hz=900,
            class_id=0,
            class_name="whale_call",
            amplitude=0.5,
        ),
        SyntheticEvent(
            start_ms=2000,
            end_ms=10000,
            freq_min_hz=50,
            freq_max_hz=200,
            class_id=1,
            class_name="ship_noise",
            amplitude=0.4,
        ),
        SyntheticEvent(
            start_ms=8500,
            end_ms=9500,
            freq_min_hz=2000,
            freq_max_hz=5000,
            class_id=2,
            class_name="click_train",
            amplitude=0.7,
        ),
        SyntheticEvent(
            start_ms=12000,
            end_ms=14000,
            freq_min_hz=500,
            freq_max_hz=1500,
            class_id=0,
            class_name="whale_call",
            amplitude=0.55,
        ),
    ]

    print("=== Merging Visualization Demo ===")
    print(f"Duration: {duration_ms / 1000:.1f}s, Window: {window_ms:.0f}ms, Overlap: {overlap_ratio:.0%}")
    print(f"Ground truth events: {len(ground_truth_events)}")

    # Generate synthetic audio
    audio = generate_synthetic_audio(duration_ms, sample_rate, ground_truth_events)

    # Compute spectrogram for visualization
    spec_db, time_axis, freq_axis = compute_simple_spectrogram(
        audio,
        sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    # Compute overlapping chunk boundaries
    chunk_boundaries: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        chunk_boundaries.append((start, end))
        start += stride_ms
    print(f"Chunk windows: {len(chunk_boundaries)}")

    # Simulate per-chunk detections
    per_chunk_detections: list[list[AudioEvent]] = []
    all_raw_events: list[AudioEvent] = []
    for i, (cs, ce) in enumerate(chunk_boundaries):
        dets = simulate_chunk_detections(
            ground_truth_events,
            cs,
            ce,
            window_index=i,
            noise_ms=80.0,
            confidence_range=(0.55, 0.92),
        )
        per_chunk_detections.append(dets)
        all_raw_events.extend(dets)

    print(f"Total raw detections: {len(all_raw_events)}")

    # Create EventList for merging
    event_list = EventList(
        events=all_raw_events,
        duration_ms=duration_ms,
        class_names={0: "whale_call", 1: "ship_noise", 2: "click_train"},
    )

    # Configure and run ClassWiseMerger
    merge_config = ClassWiseMergeConfig(
        class_params={
            # Whale calls: merge if within 1000ms and 300Hz
            0: ClassMergeParams(delta_time_ms=1000.0, delta_freq_hz=300.0, score_strategy="max"),
            # Ship noise: merge if within 2000ms and 200Hz (long-duration, low-freq)
            1: ClassMergeParams(delta_time_ms=2000.0, delta_freq_hz=200.0, score_strategy="avg"),
            # Click trains: tight merge (within 300ms, 1000Hz)
            2: ClassMergeParams(delta_time_ms=300.0, delta_freq_hz=1000.0, score_strategy="max"),
        },
        score_threshold=0.3,
        min_duration_ms=100.0,
    )
    merger = ClassWiseMerger(merge_config)
    merged_events = merger.merge(event_list)

    print(f"After merging: {len(merged_events)} events")
    for event in merged_events:
        merged_count = event.metadata.get("merged_count", 1)
        print(
            f"  {event.class_name}: {event.start_ms:.0f}-{event.end_ms:.0f}ms "
            f"({event.duration_ms:.0f}ms) score={event.score:.2f} "
            f"freq={event.min_freq_hz:.0f}-{event.max_freq_hz:.0f}Hz "
            f"[merged from {merged_count}]"
        )

    # Plot the full pipeline
    output_path = None
    if output_dir:
        output_path = Path(output_dir) / "merging_pipeline.png"

    plot_merging_pipeline(
        events=ground_truth_events,
        spec_db=spec_db,
        time_axis=time_axis,
        freq_axis=freq_axis,
        chunk_boundaries=chunk_boundaries,
        per_chunk_detections=per_chunk_detections,
        merged_events=merged_events,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merging logic visualization with synthetic audio")
    parser.add_argument("--output", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()
    main(output_dir=args.output)
