"""Visualization of chunk detection merging strategy with class-wise parameters.

This script creates fake detection data across overlapping chunks to demonstrate
the class-wise merging algorithm using Delta_Time (temporal) and Delta_Hz
(frequency) parameters instead of IoU.

It visualizes:
- Original chunks with detections
- Merged detections after applying class-wise merging strategy
- Color-coded by detection confidence
- Frequency ranges for each detection

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Import the merger classes
from rf_detr_finetuning.eventprocessor import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
    ClassWiseMerger,
)


@dataclass
class ChunkDetection:
    """Detection within a single chunk."""

    chunk_id: int
    chunk_start_ms: float
    chunk_end_ms: float
    det_start_ms: float  # Absolute time
    det_end_ms: float
    class_id: int
    class_name: str
    score: float
    min_freq_hz: float | None = None
    max_freq_hz: float | None = None

    def to_event(self) -> AudioEvent:
        """Convert to AudioEvent."""
        return AudioEvent(
            start_ms=self.det_start_ms,
            end_ms=self.det_end_ms,
            class_id=self.class_id,
            class_name=self.class_name,
            score=self.score,
            min_freq_hz=self.min_freq_hz,
            max_freq_hz=self.max_freq_hz,
            source_windows=[self.chunk_id],
        )


def create_fake_detections() -> list[ChunkDetection]:
    """Create fake detections across overlapping chunks.

    Creates realistic scenario with different sound classes that have
    different temporal and frequency characteristics:
    - bird_call: Short duration, high frequency (2-4 kHz)
    - engine_noise: Longer duration, low frequency (100-800 Hz)
    - dog_bark: Medium duration, mid frequency (500-1500 Hz)

    """
    detections = []

    # Chunk 0: 0-3200ms
    # bird_call at 500-1500ms (partial), engine_noise at 2000-2800ms
    detections.append(
        ChunkDetection(
            chunk_id=0,
            chunk_start_ms=0,
            chunk_end_ms=3200,
            det_start_ms=500,
            det_end_ms=1500,
            class_id=0,
            class_name="bird_call",
            score=0.85,
            min_freq_hz=2000,
            max_freq_hz=4000,
        )
    )
    detections.append(
        ChunkDetection(
            chunk_id=0,
            chunk_start_ms=0,
            chunk_end_ms=3200,
            det_start_ms=2000,
            det_end_ms=2800,
            class_id=1,
            class_name="engine_noise",
            score=0.92,
            min_freq_hz=100,
            max_freq_hz=800,
        )
    )

    # Chunk 1: 2560-5760ms (20% overlap with chunk 0)
    # Same events continue in overlapping region
    detections.append(
        ChunkDetection(
            chunk_id=1,
            chunk_start_ms=2560,
            chunk_end_ms=5760,
            det_start_ms=600,
            det_end_ms=1600,
            class_id=0,
            class_name="bird_call",
            score=0.88,
            min_freq_hz=2000,
            max_freq_hz=4000,
        )
    )
    detections.append(
        ChunkDetection(
            chunk_id=1,
            chunk_start_ms=2560,
            chunk_end_ms=5760,
            det_start_ms=2100,
            det_end_ms=2900,
            class_id=1,
            class_name="engine_noise",
            score=0.89,
            min_freq_hz=100,
            max_freq_hz=800,
        )
    )

    # Chunk 2: 5120-8320ms (20% overlap with chunk 1)
    # bird_call continues but at different frequency (overlaps freq gap)
    detections.append(
        ChunkDetection(
            chunk_id=2,
            chunk_start_ms=5120,
            chunk_end_ms=8320,
            det_start_ms=700,
            det_end_ms=1400,
            class_id=0,
            class_name="bird_call",
            score=0.75,
            min_freq_hz=2000,
            max_freq_hz=4000,
        )
    )

    # Chunk 3: 7680-10880ms
    # dog_bark at different frequency band
    detections.append(
        ChunkDetection(
            chunk_id=3,
            chunk_start_ms=7680,
            chunk_end_ms=10880,
            det_start_ms=900,
            det_end_ms=2500,
            class_id=2,
            class_name="dog_bark",
            score=0.95,
            min_freq_hz=500,
            max_freq_hz=1500,
        )
    )

    return detections


def visualize_detections(
    detections: list[ChunkDetection],
    merged_events: EventList | None = None,
    class_configs: dict[int, ClassMergeParams] | None = None,
    output_path: str | None = None,
) -> None:
    """Visualize chunk detections and merged events.

    Args:
        detections: List of chunk detections.
        merged_events: Optional merged events to show.
        class_configs: Optional class-wise merge configurations to display.
        output_path: Optional path to save figure.

    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Colors for different classes
    class_colors = {
        0: "#3498db",  # Blue - bird_call
        1: "#e74c3c",  # Red - engine_noise
        2: "#2ecc71",  # Green - dog_bark
    }

    # Top plot: Original chunk detections
    ax1 = axes[0]
    ax1.set_title("Raw Detections per Chunk (Before Merge)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Chunk ID")

    # Get unique chunks
    chunks = {}
    for det in detections:
        if det.chunk_id not in chunks:
            chunks[det.chunk_id] = {"start": det.chunk_start_ms, "end": det.chunk_end_ms}

    # Draw chunks as background rectangles
    for chunk_id, chunk_info in chunks.items():
        rect = mpatches.Rectangle(
            (chunk_info["start"], chunk_id - 0.4),
            chunk_info["end"] - chunk_info["start"],
            0.8,
            linewidth=1,
            edgecolor="black",
            facecolor="#ecf0f1",
            alpha=0.5,
        )
        ax1.add_patch(rect)
        ax1.text(
            chunk_info["start"] + 50,
            chunk_id,
            f"Chunk {chunk_id}\n{chunk_info['start']:.0f}-{chunk_info['end']:.0f}ms",
            fontsize=8,
            verticalalignment="center",
        )

    # Draw detections as colored rectangles
    for det in detections:
        color = class_colors.get(det.class_id, "#95a5a6")
        # Alpha based on confidence
        alpha = 0.4 + (det.score * 0.6)

        rect = mpatches.Rectangle(
            (det.det_start_ms, det.chunk_id - 0.3),
            det.det_end_ms - det.det_start_ms,
            0.6,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax1.add_patch(rect)

        # Add score and frequency info
        mid_time = (det.det_start_ms + det.det_end_ms) / 2
        freq_info = ""
        if det.min_freq_hz is not None and det.max_freq_hz is not None:
            freq_info = f"\n{det.min_freq_hz:.0f}-{det.max_freq_hz:.0f}Hz"

        ax1.text(
            mid_time,
            det.chunk_id,
            f"{det.score:.2f}{freq_info}",
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if det.score > 0.8 else "black",
        )

    ax1.set_xlim(-200, max(d.det_end_ms for d in detections) + 500)
    ax1.set_ylim(-0.5, len(chunks) - 0.5)
    ax1.set_yticks(range(len(chunks)))
    ax1.set_yticklabels([f"Chunk {i}" for i in range(len(chunks))])
    ax1.grid(True, axis="x", alpha=0.3)

    # Legend for top plot
    legend_elements = [
        mpatches.Patch(color=class_colors[0], label="bird_call (class 0)"),
        mpatches.Patch(color=class_colors[1], label="engine_noise (class 1)"),
        mpatches.Patch(color=class_colors[2], label="dog_bark (class 2)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Bottom plot: Merged events
    ax2 = axes[1]
    if merged_events:
        title = "Merged Events (After Class-wise Merge)"
        ax2.set_title(title, fontsize=14, fontweight="bold")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Event")

        # Draw all chunks in background (lighter)
        for chunk_id, chunk_info in chunks.items():
            rect = mpatches.Rectangle(
                (chunk_info["start"], -0.5),
                chunk_info["end"] - chunk_info["start"],
                len(merged_events) + 0.5,
                linewidth=1,
                edgecolor="gray",
                facecolor="#ecf0f1",
                alpha=0.3,
                linestyle="--",
            )
            ax2.add_patch(rect)

        # Draw merged events
        for i, event in enumerate(merged_events.events):
            color = class_colors.get(event.class_id, "#95a5a6")
            alpha = 0.4 + (event.score * 0.6)

            rect = mpatches.Rectangle(
                (event.start_ms, i - 0.3),
                event.end_ms - event.start_ms,
                0.6,
                linewidth=3,
                edgecolor="black",
                facecolor=color,
                alpha=alpha,
            )
            ax2.add_patch(rect)

            # Add info label with frequency range
            mid_time = (event.start_ms + event.end_ms) / 2
            duration = event.end_ms - event.start_ms
            merged_count = event.metadata.get("merged_count", 1)
            freq_str = ""
            if event.min_freq_hz is not None and event.max_freq_hz is not None:
                freq_str = f"\n{event.min_freq_hz:.0f}-{event.max_freq_hz:.0f}Hz"

            label = (
                f"{event.class_name}\nscore:{event.score:.2f}\ndur:{duration:.0f}ms\n({merged_count} merged){freq_str}"
            )

            ax2.text(
                mid_time,
                i,
                label,
                fontsize=9,
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if event.score > 0.8 else "black",
            )

        ax2.set_xlim(-200, max(d.det_end_ms for d in detections) + 500)
        ax2.set_ylim(-0.5, len(merged_events) - 0.5)
        ax2.set_yticks(range(len(merged_events)))
        ax2.set_yticklabels([f"Event {i + 1}" for i in range(len(merged_events))])
        ax2.grid(True, axis="x", alpha=0.3)

        # Add legend for merge info
        ax2.legend(
            handles=[
                mpatches.Patch(facecolor="gray", alpha=0.3, label="Chunk boundaries"),
            ],
            loc="upper right",
        )

        # Add class-wise config info if provided
        if class_configs:
            config_text = "Class-wise Merge Config:\n"
            for class_id, params in sorted(class_configs.items()):
                config_text += f"  Class {class_id}: Δt={params.delta_time_ms}ms, Δf={params.delta_freq_hz}Hz\n"
            ax2.text(
                0.02,
                0.98,
                config_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No merged events to display",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


def print_detection_info(detections: list[ChunkDetection]) -> None:
    """Print information about detections.

    Args:
        detections: List of chunk detections.

    """
    print("\n" + "=" * 80)
    print("RAW DETECTIONS ACROSS CHUNKS")
    print("=" * 80)

    chunks = {}
    for det in detections:
        if det.chunk_id not in chunks:
            chunks[det.chunk_id] = []
        chunks[det.chunk_id].append(det)

    for chunk_id in sorted(chunks.keys()):
        chunk_dets = chunks[chunk_id]
        chunk_info = chunk_dets[0]
        print(f"\nChunk {chunk_id}: {chunk_info.chunk_start_ms:.0f}-{chunk_info.chunk_end_ms:.0f}ms")
        for det in chunk_dets:
            duration = det.det_end_ms - det.det_start_ms
            freq_str = ""
            if det.min_freq_hz is not None and det.max_freq_hz is not None:
                freq_str = f", freq={det.min_freq_hz:.0f}-{det.max_freq_hz:.0f}Hz"
            print(
                f"  - {det.class_name}: {det.det_start_ms:.0f}-{det.det_end_ms:.0f}ms "
                f"(dur={duration:.0f}ms, score={det.score:.2f}{freq_str})"
            )

    print(f"\nTotal raw detections: {len(detections)}")


def print_merged_info(events: EventList, num_raw_detections: int) -> None:
    """Print information about merged events.

    Args:
        events: Merged event list.
        num_raw_detections: Number of raw detections before merge.

    """
    print("\n" + "=" * 80)
    print("MERGED EVENTS")
    print("=" * 80)

    for i, event in enumerate(events.events, 1):
        duration = event.end_ms - event.start_ms
        merged_count = event.metadata.get("merged_count", 1)
        merge_strategy = event.metadata.get("merge_strategy", "max")
        freq_str = ""
        if event.min_freq_hz is not None and event.max_freq_hz is not None:
            freq_str = f", freq={event.min_freq_hz:.0f}-{event.max_freq_hz:.0f}Hz"

        print(f"\nEvent {i}:")
        print(f"  Class: {event.class_name} (id={event.class_id})")
        print(f"  Time: {event.start_ms:.0f}-{event.end_ms:.0f}ms (duration={duration:.0f}ms)")
        print(f"  Score: {event.score:.2f} (strategy: {merge_strategy})")
        print(f"  Merged from {merged_count} raw detection(s)")
        print(f"  Source chunks: {event.source_windows}{freq_str}")

    print(f"\nTotal merged events: {len(events)}")
    reduction = 100 * (1 - len(events) / num_raw_detections)
    print(f"Reduction: {reduction:.1f}% ({num_raw_detections} → {len(events)} events)")


def main() -> None:
    """Run the visualization demo with class-wise merging."""
    print("Class-wise Chunk Detection Merging Strategy Demo")
    print("=" * 80)
    print("Using Delta_Time (temporal) and Delta_Hz (frequency) instead of IoU")
    print()

    # Create fake detections
    detections = create_fake_detections()
    print_detection_info(detections)

    # Convert to EventList
    events = EventList(
        events=[det.to_event() for det in detections],
        audio_path="fake_audio.wav",
        duration_ms=11000,
        class_names={0: "bird_call", 1: "engine_noise", 2: "dog_bark"},
    )

    # Configure class-wise merging with different parameters per class
    # bird_call: Short events, need tight temporal merging
    # engine_noise: Long continuous sound, allow larger temporal gap
    # dog_bark: Medium duration, strict frequency matching
    class_configs = {
        0: ClassMergeParams(
            delta_time_ms=300,  # Merge if within 300ms
            delta_freq_hz=500,  # Allow 500Hz frequency difference
            score_strategy="max",
        ),
        1: ClassMergeParams(
            delta_time_ms=800,  # Merge if within 800ms (longer tolerance)
            delta_freq_hz=200,  # Strict frequency match (engine has narrow band)
            score_strategy="avg",
        ),
        2: ClassMergeParams(
            delta_time_ms=500,  # Merge if within 500ms
            delta_freq_hz=300,  # Moderate frequency tolerance
            score_strategy="weighted",
        ),
    }

    merge_config = ClassWiseMergeConfig(
        class_params=class_configs,
        default_params=ClassMergeParams(
            delta_time_ms=500,
            delta_freq_hz=500,
            score_strategy="max",
        ),
        score_threshold=0.0,
    )

    print("\n" + "=" * 80)
    print("CLASS-WISE MERGE CONFIGURATION")
    print("=" * 80)
    print("Per-class merge parameters (Delta_Time, Delta_Hz):")
    for class_id, params in sorted(class_configs.items()):
        class_name = events.class_names.get(class_id, f"class_{class_id}")
        print(f"  {class_name} (class {class_id}):")
        print(f"    Δ_time = {params.delta_time_ms}ms")
        print(f"    Δ_freq = {params.delta_freq_hz}Hz")
        print(f"    score_strategy = {params.score_strategy}")
    print()

    # Apply class-wise merging
    merger = ClassWiseMerger(merge_config)
    merged_events = merger.merge(events)

    print_merged_info(merged_events, len(detections))

    # Create visualization
    output_dir = Path("output/test_merging")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "class_wise_merging_visualization.png"

    visualize_detections(detections, merged_events, class_configs, str(output_path))

    print("\n" + "=" * 80)
    print("CLASS-WISE MERGING STRATEGY SUMMARY")
    print("=" * 80)
    print("""
Instead of IoU-based merging, we use class-specific parameters:

1. For each class, define:
   - Delta_Time (Δt): Maximum temporal distance to merge
   - Delta_Hz (Δf): Maximum frequency distance to merge
   - Score strategy: How to combine scores (max/avg/weighted)

2. Group events by class

3. For each class:
   - Sort by start time
   - Iterate through events
   - If next event is within Δt AND Δf, merge it
   - Otherwise, start new group

4. Merge groups:
   - Union of temporal boundaries
   - Union of frequency boundaries
   - Apply score strategy

Benefits over IoU-based merging:
- Class-specific behavior (short vs long events)
- Frequency-aware merging (important for multi-species audio)
- More intuitive parameters (time/frequency rather than overlap ratio)
- Better for sounds with different temporal characteristics
""")


if __name__ == "__main__":
    main()
