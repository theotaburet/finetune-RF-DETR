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
    def duration_s(self) -> float:
        """Event duration in seconds."""
        return self.duration_ms / 1000

    @property
    def center_ms(self) -> float:
        """Event center time in milliseconds."""
        return (self.start_ms + self.end_ms) / 2

    def overlaps(self, other: AudioEvent, iou_threshold: float = 0.0) -> bool:
        """Check if this event overlaps with another.

        Args:
            other: Another AudioEvent.
            iou_threshold: Minimum IoU for overlap.

        Returns:
            True if events overlap above threshold.

        """
        iou = self.temporal_iou(other)
        return iou > iou_threshold

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

    def merge(self, other: AudioEvent) -> AudioEvent:
        """Merge this event with another.

        Takes the union of temporal boundaries and averages scores.

        Args:
            other: Event to merge with.

        Returns:
            New merged AudioEvent.

        """
        return AudioEvent(
            start_ms=min(self.start_ms, other.start_ms),
            end_ms=max(self.end_ms, other.end_ms),
            class_id=self.class_id,
            class_name=self.class_name,
            score=(self.score + other.score) / 2,
            min_freq_hz=min(
                self.min_freq_hz or float("inf"),
                other.min_freq_hz or float("inf"),
            )
            if self.min_freq_hz or other.min_freq_hz
            else None,
            max_freq_hz=max(
                self.max_freq_hz or 0,
                other.max_freq_hz or 0,
            )
            if self.max_freq_hz or other.max_freq_hz
            else None,
            source_windows=list(set(self.source_windows + other.source_windows)),
        )

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
        return EventList(
            events=filtered,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

    def filter_by_class(self, class_ids: list[int]) -> EventList:
        """Filter events by class.

        Args:
            class_ids: List of class IDs to keep.

        Returns:
            New EventList with filtered events.

        """
        filtered = [e for e in self.events if e.class_id in class_ids]
        return EventList(
            events=filtered,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

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

        return EventList(
            events=filtered,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

    def sort_by_time(self) -> EventList:
        """Sort events by start time.

        Returns:
            New EventList with sorted events.

        """
        sorted_events = sorted(self.events, key=lambda e: e.start_ms)
        return EventList(
            events=sorted_events,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

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
        return EventList(
            events=sorted_events,
            audio_path=self.audio_path,
            duration_ms=self.duration_ms,
            class_names=self.class_names,
        )

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
            "class_names": self.class_names,
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

        return cls(
            events=events,
            audio_path=data.get("audio_path"),
            duration_ms=data.get("duration_ms", 0.0),
            class_names={int(k): v for k, v in data.get("class_names", {}).items()},
        )
