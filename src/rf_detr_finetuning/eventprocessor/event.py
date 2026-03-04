"""Audio event representation.

Provides dataclasses for representing detected audio events with temporal and frequency information.

"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioEvent:
    """Detected audio event with temporal boundaries.

    Attributes:
        start_ms: Event start time in milliseconds.
        end_ms: Event end time in milliseconds.
        class_id: Predicted class index.
        class_name: Predicted class name (optional).
        score: Detection confidence score.
        min_freq_hz: Minimum frequency in Hz (optional).
        max_freq_hz: Maximum frequency in Hz (optional).
        source_windows: List of window indices that contributed to this event.
        metadata: Additional event metadata.

    """

    start_ms: float
    end_ms: float
    class_id: int
    class_name: str | None = None
    score: float = 1.0
    min_freq_hz: float | None = None
    max_freq_hz: float | None = None
    source_windows: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Event duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def center_ms(self) -> float:
        """Event center time in milliseconds."""
        return (self.start_ms + self.end_ms) / 2

    def temporal_iou(self, other: AudioEvent) -> float:
        """Compute temporal IoU with another event.

        Args:
            other: Another AudioEvent.

        Returns:
            Intersection over Union (0-1).

        """
        # Compute intersection
        inter_start = max(self.start_ms, other.start_ms)
        inter_end = min(self.end_ms, other.end_ms)
        intersection = max(0, inter_end - inter_start)

        # Compute union
        union = self.duration_ms + other.duration_ms - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def temporal_frequency_iou(self, other: AudioEvent) -> float:
        """Compute 2D IoU over both time and frequency axes.

        Uses the product of temporal and frequency extents as the "area"
        of each event, computing a true 2D intersection-over-union.
        Falls back to temporal-only IoU if either event lacks frequency info.

        Args:
            other: Another AudioEvent.

        Returns:
            2D Intersection over Union (0-1).

        """
        # Fall back to temporal IoU if frequency info is missing
        if (
            self.min_freq_hz is None
            or self.max_freq_hz is None
            or other.min_freq_hz is None
            or other.max_freq_hz is None
        ):
            return self.temporal_iou(other)

        # Temporal intersection
        t_inter_start = max(self.start_ms, other.start_ms)
        t_inter_end = min(self.end_ms, other.end_ms)
        t_intersection = max(0.0, t_inter_end - t_inter_start)

        # Frequency intersection
        f_inter_start = max(self.min_freq_hz, other.min_freq_hz)
        f_inter_end = min(self.max_freq_hz, other.max_freq_hz)
        f_intersection = max(0.0, f_inter_end - f_inter_start)

        intersection_area = t_intersection * f_intersection

        # Areas
        self_area = self.duration_ms * (self.max_freq_hz - self.min_freq_hz)
        other_area = other.duration_ms * (other.max_freq_hz - other.min_freq_hz)

        union_area = self_area + other_area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "min_freq_hz": self.min_freq_hz,
            "max_freq_hz": self.max_freq_hz,
            "source_windows": self.source_windows,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioEvent:
        """Create from dictionary."""
        return cls(
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            class_id=data["class_id"],
            class_name=data.get("class_name"),
            score=data.get("score", 1.0),
            min_freq_hz=data.get("min_freq_hz"),
            max_freq_hz=data.get("max_freq_hz"),
            source_windows=data.get("source_windows", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EventList:
    """Collection of audio events with utility methods.

    Attributes:
        events: List of AudioEvent objects.
        audio_path: Source audio file path.
        duration_ms: Total audio duration.
        class_names: Mapping of class IDs to names.

    """

    events: list[AudioEvent] = field(default_factory=list)
    audio_path: str | Path | None = None
    duration_ms: float = 0.0
    class_names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of events."""
        return len(self.events)

    def __iter__(self) -> Iterator[AudioEvent]:
        """Return iterator over events."""
        return iter(self.events)

    def __getitem__(self, idx: int) -> AudioEvent:
        """Get event by index."""
        return self.events[idx]

    def _with_events(self, events: list[AudioEvent]) -> EventList:
        """Create a new EventList sharing this list's metadata.

        Args:
            events: New events list.

        Returns:
            New EventList with the same audio_path, duration_ms, and class_names.

        """
        return EventList(
            events=events,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

    def add(self, event: AudioEvent) -> None:
        """Add an event."""
        self.events.append(event)

    def filter_by_score(self, min_score: float) -> EventList:
        """Filter events by minimum score.

        Args:
            min_score: Minimum confidence threshold.

        Returns:
            New EventList with filtered events.

        """
        filtered = [e for e in self.events if e.score >= min_score]
        return self._with_events(filtered)

    def filter_by_class(self, class_ids: list[int]) -> EventList:
        """Filter events by class.

        Args:
            class_ids: List of class IDs to keep.

        Returns:
            New EventList with filtered events.

        """
        filtered = [e for e in self.events if e.class_id in class_ids]
        return self._with_events(filtered)

    def filter_by_duration(
        self,
        min_duration_ms: float = 0.0,
        max_duration_ms: float | None = None,
    ) -> EventList:
        """Filter events by duration.

        Args:
            min_duration_ms: Minimum event duration.
            max_duration_ms: Maximum event duration (None = no limit).

        Returns:
            New EventList with filtered events.

        """
        filtered = []
        for e in self.events:
            if e.duration_ms < min_duration_ms:
                continue
            if max_duration_ms is not None and e.duration_ms > max_duration_ms:
                continue
            filtered.append(e)

        return self._with_events(filtered)

    def sort_by_time(self) -> EventList:
        """Sort events by start time.

        Returns:
            New EventList with sorted events.

        """
        sorted_events = sorted(self.events, key=lambda e: e.start_ms)
        return self._with_events(sorted_events)

    def sort_by_score(self, descending: bool = True) -> EventList:
        """Sort events by score.

        Args:
            descending: Sort high to low if True.

        Returns:
            New EventList with sorted events.

        """
        sorted_events = sorted(
            self.events,
            key=lambda e: e.score,
            reverse=descending,
        )
        return self._with_events(sorted_events)

    def get_class_counts(self) -> dict[int, int]:
        """Get count of events per class.

        Returns:
            Dictionary mapping class ID to count.

        """
        counts: dict[int, int] = {}
        for e in self.events:
            counts[e.class_id] = counts.get(e.class_id, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "duration_ms": self.duration_ms,
            "num_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "class_names": {str(k): v for k, v in self.class_names.items()},
        }

    def save(self, path: str | Path) -> None:
        """Save to JSON file.

        Args:
            path: Output file path.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EventList:
        """Load from JSON file.

        Args:
            path: Input file path.

        Returns:
            EventList instance.

        """
        with open(path) as f:
            data = json.load(f)

        events = [AudioEvent.from_dict(e) for e in data.get("events", [])]

        # JSON serialization turns int keys to strings; convert back
        raw_class_names = data.get("class_names", {})
        class_names = {int(k): v for k, v in raw_class_names.items()}

        return cls(
            events=events,
            audio_path=data.get("audio_path"),
            duration_ms=data.get("duration_ms", 0.0),
            class_names=class_names,
        )
