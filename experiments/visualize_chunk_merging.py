"""Visualization of chunk detection merging strategy for marine acoustics.

This script creates fake detection data across overlapping chunks to demonstrate
the class-wise merging algorithm for underwater acoustic analysis using
Delta_Time (temporal) and Delta_Hz (frequency) parameters.

It visualizes:
- Original chunks with detections
- Merged detections after applying class-wise merging strategy
- Color-coded by detection confidence
- Frequency ranges for each detection

Example marine species:
- Humpback whale: Complex songs with frequency sweeps
- Killer whale: Short pulsed calls and echolocation clicks
- Blue whale: Very long, low-frequency calls
- Dolphin: Rapid echolocation clicks
- Ship noise: Continuous broadband anthropogenic noise

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

    Creates realistic marine acoustic scenario with different species:
    - humpback_whale: Complex songs with frequency sweeps (200-3500 Hz)
    - killer_whale: Short pulsed calls and clicks (1-20 kHz)
    - blue_whale: Very long, low-frequency calls (10-40 Hz)
    - dolphin: Rapid echolocation clicks (up to 150 kHz)
    - ship_noise: Continuous broadband anthropogenic noise (10 Hz - 5 kHz)

    """
    detections = []

    # Chunk 0: 0-3200ms
    # Humpback whale song at 500-1500ms (partial), Ship noise at 2000-2800ms
    detections.append(
        ChunkDetection(
            chunk_id=0,
            chunk_start_ms=0,
            chunk_end_ms=3200,
            det_start_ms=500,
            det_end_ms=1500,
            class_id=0,
            class_name="humpback_whale",
            score=0.85,
            min_freq_hz=200,
            max_freq_hz=3500,
        )
    )
    detections.append(
        ChunkDetection(
            chunk_id=0,
            chunk_start_ms=0,
            chunk_end_ms=3200,
            det_start_ms=2000,
            det_end_ms=2800,
            class_id=4,
            class_name="ship_noise",
            score=0.92,
            min_freq_hz=50,
            max_freq_hz=5000,
        )
    )

    # Chunk 1: 2560-5760ms (20% overlap with chunk 0)
    # Same humpback song continues, killer whale calls at different frequency
    detections.append(
        ChunkDetection(
            chunk_id=1,
            chunk_start_ms=2560,
            chunk_end_ms=5760,
            det_start_ms=600,
            det_end_ms=1600,
            class_id=0,
            class_name="humpback_whale",
            score=0.88,
            min_freq_hz=200,
            max_freq_hz=3500,
        )
    )
    detections.append(
        ChunkDetection(
            chunk_id=1,
            chunk_start_ms=2560,
            chunk_end_ms=5760,
            det_start_ms=3000,
            det_end_ms=3800,
            class_id=1,
            class_name="killer_whale",
            score=0.89,
            min_freq_hz=2000,
            max_freq_hz=18000,
        )
    )

    # Chunk 2: 5120-8320ms (20% overlap with chunk 1)
    # Blue whale call at very low frequency
    detections.append(
        ChunkDetection(
            chunk_id=2,
            chunk_start_ms=5120,
            chunk_end_ms=8320,
            det_start_ms=700,
            det_end_ms=1400,
            class_id=0,
            class_name="humpback_whale",
            score=0.75,
            min_freq_hz=200,
            max_freq_hz=3500,
        )
    )
    detections.append(
        ChunkDetection(
            chunk_id=2,
            chunk_start_ms=5120,
            chunk_end_ms=8320,
            det_start_ms=500,
            det_end_ms=4500,
            class_id=2,
            class_name="blue_whale",
            score=0.95,
            min_freq_hz=10,
            max_freq_hz=40,
        )
    )

    # Chunk 3: 7680-10880ms
    # Dolphin clicks at high frequency
    detections.append(
        ChunkDetection(
            chunk_id=3,
            chunk_start_ms=7680,
            chunk_end_ms=10880,
            det_start_ms=900,
            det_end_ms=2500,
            class_id=3,
            class_name="dolphin",
            score=0.93,
            min_freq_hz=20000,
            max_freq_hz=120000,
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

    # Colors for different classes (marine species)
    class_colors = {
        0: "#3498db",  # Blue - humpback_whale
        1: "#e74c3c",  # Red - killer_whale
        2: "#2ecc71",  # Green - blue_whale
        3: "#f39c12",  # Orange - dolphin
        4: "#9b59b6",  # Purple - ship_noise
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
            freq_str = f"{det.min_freq_hz:.0f}-{det.max_freq_hz:.0f}Hz"
            freq_info = f"\n{freq_str}"

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
        mpatches.Patch(color=class_colors[0], label="humpback_whale (200-3500 Hz)"),
        mpatches.Patch(color=class_colors[1], label="killer_whale (2-18 kHz)"),
        mpatches.Patch(color=class_colors[2], label="blue_whale (10-40 Hz)"),
        mpatches.Patch(color=class_colors[3], label="dolphin (20-120 kHz)"),
        mpatches.Patch(color=class_colors[4], label="ship_noise (broadband)"),
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
            config_text = "Class-wise Merge Config (Marine):\n"
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
    print("RAW DETECTIONS ACROSS CHUNKS (Marine Acoustic Scenario)")
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
    print("MERGED EVENTS (Marine Acoustic Analysis)")
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
    """Run the visualization demo with marine acoustic merging."""
    print("Marine Acoustic Event Merging Strategy Demo")
    print("=" * 80)
    print("Using Delta_Time (temporal) and Delta_Hz (frequency) for underwater sounds")
    print()

    # Create fake detections
    detections = create_fake_detections()
    print_detection_info(detections)

    # Convert to EventList
    events = EventList(
        events=[det.to_event() for det in detections],
        audio_path="hydrophone_recording.flac",
        duration_ms=11000,
        class_names={
            0: "humpback_whale",
            1: "killer_whale",
            2: "blue_whale",
            3: "dolphin",
            4: "ship_noise",
        },
    )

    # Configure class-wise merging for marine species
    # Different parameters based on species' acoustic characteristics
    class_configs = {
        0: ClassMergeParams(
            delta_time_ms=3000,  # Humpback: Allow 3s gaps for continuous songs
            delta_freq_hz=400,  # Moderate tolerance for frequency sweeps
            score_strategy="max",
        ),
        1: ClassMergeParams(
            delta_time_ms=800,  # Killer whale: Tight tolerance for discrete calls
            delta_freq_hz=600,  # Broadband clicks
            score_strategy="max",
        ),
        2: ClassMergeParams(
            delta_time_ms=8000,  # Blue whale: Very long tolerance (8s)
            delta_freq_hz=30,  # Very narrow (infrasonic)
            score_strategy="avg",
        ),
        3: ClassMergeParams(
            delta_time_ms=400,  # Dolphin: Very tight for rapid clicks
            delta_freq_hz=800,  # Wide frequency range
            score_strategy="max",
        ),
        4: ClassMergeParams(
            delta_time_ms=5000,  # Ship noise: Long tolerance for continuous noise
            delta_freq_hz=100,  # Narrow band
            score_strategy="avg",
        ),
    }

    merge_config = ClassWiseMergeConfig(
        class_params=class_configs,
        default_params=ClassMergeParams(
            delta_time_ms=2000,
            delta_freq_hz=150,
            score_strategy="max",
        ),
        score_threshold=0.3,
        min_duration_ms=200.0,
    )

    print("\n" + "=" * 80)
    print("CLASS-WISE MERGE CONFIGURATION (Marine Acoustics)")
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
    output_path = output_dir / "marine_acoustic_merging_visualization.png"

    visualize_detections(detections, merged_events, class_configs, str(output_path))

    print("\n" + "=" * 80)
    print("MARINE ACOUSTIC MERGING STRATEGY SUMMARY")
    print("=" * 80)
    print("""
Class-wise merging using Delta_Time and Delta_Hz is especially important
for underwater acoustics because:

1. Different marine species occupy different frequency bands:
   - Blue whales: 10-40 Hz (infrasonic)
   - Humpback whales: 10 Hz - 4 kHz (complex songs)
   - Killer whales: 1-20 kHz (pulsed calls)
   - Dolphins: Up to 150 kHz (echolocation)

2. Temporal characteristics vary greatly:
   - Blue whale calls: 10-30 seconds
   - Humpback songs: Minutes with gaps
   - Dolphin clicks: <1 ms bursts
   - Ship noise: Continuous over minutes

3. Frequency-aware merging prevents:
   - Merging blue whale calls with ship noise
   - Combining dolphin clicks with whale songs
   - Mis-attributing sounds across species

Benefits for marine bioacoustics:
- Species-specific merge tolerances
- Frequency band isolation
- Better preservation of call structure
- Reduced false positives from overlapping chunks
""")


if __name__ == "__main__":
    main()
