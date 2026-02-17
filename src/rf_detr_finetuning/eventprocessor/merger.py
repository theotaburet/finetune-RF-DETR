"""Event merging strategies.

Provides algorithms for merging overlapping detections:
- NMS-based merging
- Temporal IoU merging
- Soft-NMS

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for event merging.

    Attributes:
        iou_threshold: IoU threshold for considering overlap.
        score_threshold: Minimum score to keep events.
        merge_same_class_only: Only merge events of same class.
        merge_strategy: How to combine scores ('max', 'avg', 'sum').
        gap_tolerance_ms: Merge events within this gap (temporal).

    """

    iou_threshold: float = 0.5
    score_threshold: float = 0.0
    merge_same_class_only: bool = True
    merge_strategy: str = "max"
    gap_tolerance_ms: float = 0.0


class EventMerger:
    """Merges overlapping audio events.

    Args:
        config: Merge configuration.

    """

    def __init__(self, config: MergeConfig | None = None) -> None:
        """Initialize event merger.

        Args:
            config: Merge configuration. Uses defaults if None.

        """
        self.config = config or MergeConfig()

    def merge(self, events: EventList) -> EventList:
        """Merge overlapping events.

        Args:
            events: Input event list.

        Returns:
            Merged event list.

        """
        if len(events) == 0:
            return events

        # Sort by score (descending) for greedy NMS
        sorted_events = events.sort_by_score(descending=True).events

        merged = []
        suppressed = set()

        for i, event in enumerate(sorted_events):
            if i in suppressed:
                continue

            # Find all overlapping events
            to_merge = [event]
            for j, other in enumerate(sorted_events[i + 1 :], start=i + 1):
                if j in suppressed:
                    continue

                # Check class constraint
                if self.config.merge_same_class_only and event.class_id != other.class_id:
                    continue

                # Check IoU overlap
                iou = event.temporal_iou(other)
                if iou >= self.config.iou_threshold:
                    to_merge.append(other)
                    suppressed.add(j)

                # Check gap tolerance (for temporal merging)
                elif self.config.gap_tolerance_ms > 0:
                    gap = self._compute_gap(event, other)
                    if gap <= self.config.gap_tolerance_ms:
                        to_merge.append(other)
                        suppressed.add(j)

            # Merge the group
            merged_event = self._merge_group(to_merge)
            merged.append(merged_event)

        result = EventList(
            events=merged,
            audio_path=events.audio_path,
            duration_ms=events.duration_ms,
            class_names=events.class_names,
        )

        # Filter by score threshold
        if self.config.score_threshold > 0:
            result = result.filter_by_score(self.config.score_threshold)

        return result.sort_by_time()

    def _compute_gap(self, a: AudioEvent, b: AudioEvent) -> float:
        """Compute temporal gap between events.

        Args:
            a: First event.
            b: Second event.

        Returns:
            Gap in milliseconds (negative if overlapping).

        """
        if a.end_ms <= b.start_ms:
            return b.start_ms - a.end_ms
        elif b.end_ms <= a.start_ms:
            return a.start_ms - b.end_ms
        else:
            return 0.0  # Overlapping

    def _merge_group(self, events: list[AudioEvent]) -> AudioEvent:
        """Merge a group of overlapping events.

        Args:
            events: Events to merge.

        Returns:
            Single merged event.

        """
        if len(events) == 1:
            return events[0]

        # Compute merged boundaries
        start_ms = min(e.start_ms for e in events)
        end_ms = max(e.end_ms for e in events)

        # Compute merged score
        scores = [e.score for e in events]
        if self.config.merge_strategy == "max":
            score = max(scores)
        elif self.config.merge_strategy == "avg":
            score = sum(scores) / len(scores)
        elif self.config.merge_strategy == "sum":
            score = min(1.0, sum(scores))
        else:
            score = max(scores)

        # Take class from highest-scoring event
        best_event = max(events, key=lambda e: e.score)

        # Merge source windows
        all_windows = []
        for e in events:
            all_windows.extend(e.source_windows)

        # Merge frequency info
        min_freq = min(
            (e.min_freq_hz for e in events if e.min_freq_hz is not None),
            default=None,
        )
        max_freq = max(
            (e.max_freq_hz for e in events if e.max_freq_hz is not None),
            default=None,
        )

        return AudioEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            class_id=best_event.class_id,
            class_name=best_event.class_name,
            score=score,
            min_freq_hz=min_freq,
            max_freq_hz=max_freq,
            source_windows=list(set(all_windows)),
            metadata={"merged_count": len(events)},
        )


def nms_merge(
    events: EventList,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> EventList:
    """Apply NMS-style merging to events.

    Standard greedy NMS that suppresses overlapping lower-confidence events.

    Args:
        events: Input events.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score to keep.

    Returns:
        Merged events.

    """
    config = MergeConfig(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        merge_strategy="max",
    )
    merger = EventMerger(config)
    return merger.merge(events)


def temporal_merge(
    events: EventList,
    gap_tolerance_ms: float = 100.0,
    merge_same_class_only: bool = True,
) -> EventList:
    """Merge events that are temporally close.

    Merges events that are within gap_tolerance_ms of each other.

    Args:
        events: Input events.
        gap_tolerance_ms: Maximum gap to merge across.
        merge_same_class_only: Only merge same class.

    Returns:
        Merged events.

    """
    config = MergeConfig(
        iou_threshold=0.0,  # Don't require overlap
        gap_tolerance_ms=gap_tolerance_ms,
        merge_same_class_only=merge_same_class_only,
        merge_strategy="avg",
    )
    merger = EventMerger(config)
    return merger.merge(events)


@dataclass
class ClassMergeParams:
    """Per-class merge parameters.

    Attributes:
        delta_time_ms: Maximum temporal distance to merge (in milliseconds).
            Events within this time window will be considered for merging.
        delta_freq_hz: Maximum frequency distance to merge (in Hz).
            Events within this frequency window will be considered for merging.
        min_overlap_ratio: Minimum temporal overlap ratio (0-1).
            Alternative to delta_time_ms for overlap-based merging.
        score_strategy: How to combine scores ('max', 'avg', 'weighted').

    """

    delta_time_ms: float = 500.0
    delta_freq_hz: float = 500.0
    min_overlap_ratio: float | None = None
    score_strategy: str = "max"


@dataclass
class ClassWiseMergeConfig:
    """Configuration for class-wise event merging.

    Allows different merge parameters per class using Delta_Time and Delta_Hz
    instead of IoU-based merging.

    Attributes:
        class_params: Dictionary mapping class_id to ClassMergeParams.
        default_params: Default merge parameters for classes not in class_params.
        score_threshold: Minimum score to keep events.
        min_duration_ms: Minimum event duration after merge.
        max_duration_ms: Maximum event duration after merge.

    Example:
        >>> config = ClassWiseMergeConfig(
        ...     class_params={
        ...         0: ClassMergeParams(delta_time_ms=200, delta_freq_hz=300),
        ...         1: ClassMergeParams(delta_time_ms=1000, delta_freq_hz=100),
        ...     },
        ...     default_params=ClassMergeParams(delta_time_ms=500, delta_freq_hz=500),
        ... )

    """

    class_params: dict[int, ClassMergeParams] = field(default_factory=dict)
    default_params: ClassMergeParams = field(default_factory=ClassMergeParams)
    score_threshold: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float | None = None


class ClassWiseMerger:
    """Merges audio events with class-specific parameters.

    Uses Delta_Time (temporal distance) and Delta_Hz (frequency distance)
    instead of IoU for determining which events to merge. This allows
    class-specific merging behavior.

    Args:
        config: Class-wise merge configuration.

    Example:
        >>> config = ClassWiseMergeConfig(
        ...     class_params={
        ...         0: ClassMergeParams(delta_time_ms=200, delta_freq_hz=300),
        ...         1: ClassMergeParams(delta_time_ms=1000, delta_freq_hz=100),
        ...     },
        ... )
        >>> merger = ClassWiseMerger(config)
        >>> merged = merger.merge(events)

    """

    def __init__(self, config: ClassWiseMergeConfig | None = None) -> None:
        """Initialize class-wise merger.

        Args:
            config: Class-wise merge configuration. Uses defaults if None.

        """
        self.config = config or ClassWiseMergeConfig()

    def merge(self, events: EventList) -> EventList:
        """Merge events using class-wise parameters.

        Args:
            events: Input event list.

        Returns:
            Merged event list.

        """
        if len(events) == 0:
            return events

        # Group events by class
        events_by_class: dict[int, list[AudioEvent]] = {}
        for event in events:
            if event.class_id not in events_by_class:
                events_by_class[event.class_id] = []
            events_by_class[event.class_id].append(event)

        # Merge each class separately
        merged_events = []
        for class_id, class_events in events_by_class.items():
            params = self.config.class_params.get(class_id, self.config.default_params)
            merged = self._merge_class_events(class_events, params)
            merged_events.extend(merged)

        # Create result
        result = EventList(
            events=merged_events,
            audio_path=events.audio_path,
            duration_ms=events.duration_ms,
            class_names=events.class_names,
        )

        # Apply post-merge filters
        if self.config.score_threshold > 0:
            result = result.filter_by_score(self.config.score_threshold)

        if self.config.min_duration_ms > 0 or self.config.max_duration_ms is not None:
            result = result.filter_by_duration(
                min_duration_ms=self.config.min_duration_ms,
                max_duration_ms=self.config.max_duration_ms,
            )

        return result.sort_by_time()

    def _merge_class_events(self, events: list[AudioEvent], params: ClassMergeParams) -> list[AudioEvent]:
        """Merge events of a single class.

        Args:
            events: Events of the same class.
            params: Merge parameters for this class.

        Returns:
            Merged events.

        """
        if len(events) <= 1:
            return events

        # Sort by start time
        sorted_events = sorted(events, key=lambda e: e.start_ms)

        merged = []
        current_group = [sorted_events[0]]

        for event in sorted_events[1:]:
            # Check if this event should merge with current group
            should_merge = self._should_merge(event, current_group[-1], params)

            if should_merge:
                current_group.append(event)
            else:
                # Merge current group and start new one
                merged.append(self._merge_group(current_group, params))
                current_group = [event]

        # Don't forget the last group
        if current_group:
            merged.append(self._merge_group(current_group, params))

        return merged

    def _should_merge(self, a: AudioEvent, b: AudioEvent, params: ClassMergeParams) -> bool:
        """Check if two events should be merged.

        Args:
            a: First event.
            b: Second event.
            params: Merge parameters.

        Returns:
            True if events should be merged.

        """
        # Check temporal distance/overlap
        if params.min_overlap_ratio is not None:
            # Use overlap ratio
            iou = a.temporal_iou(b)
            if iou < params.min_overlap_ratio:
                return False
        else:
            # Use delta_time_ms
            time_gap = self._compute_time_gap(a, b)
            if time_gap > params.delta_time_ms:
                return False

        # Check frequency distance
        if a.min_freq_hz is not None and a.max_freq_hz is not None:
            if b.min_freq_hz is not None and b.max_freq_hz is not None:
                freq_gap = self._compute_freq_gap(a, b)
                if freq_gap > params.delta_freq_hz:
                    return False

        return True

    def _compute_time_gap(self, a: AudioEvent, b: AudioEvent) -> float:
        """Compute temporal gap between events.

        Args:
            a: First event.
            b: Second event.

        Returns:
            Gap in milliseconds (0 if overlapping).

        """
        if a.end_ms <= b.start_ms:
            return b.start_ms - a.end_ms
        elif b.end_ms <= a.start_ms:
            return a.start_ms - b.end_ms
        else:
            return 0.0

    def _compute_freq_gap(self, a: AudioEvent, b: AudioEvent) -> float:
        """Compute frequency gap between events.

        Args:
            a: First event.
            b: Second event.

        Returns:
            Gap in Hz (0 if overlapping).

        """
        if a.min_freq_hz is None or a.max_freq_hz is None or b.min_freq_hz is None or b.max_freq_hz is None:
            return 0.0

        if a.max_freq_hz <= b.min_freq_hz:
            return b.min_freq_hz - a.max_freq_hz
        elif b.max_freq_hz <= a.min_freq_hz:
            return a.min_freq_hz - b.max_freq_hz
        else:
            return 0.0

    def _merge_group(self, events: list[AudioEvent], params: ClassMergeParams) -> AudioEvent:
        """Merge a group of events.

        Args:
            events: Events to merge.
            params: Merge parameters for score strategy.

        Returns:
            Merged event.

        """
        if len(events) == 1:
            return events[0]

        # Compute merged boundaries
        start_ms = min(e.start_ms for e in events)
        end_ms = max(e.end_ms for e in events)

        # Compute merged score
        scores = [e.score for e in events]
        if params.score_strategy == "max":
            score = max(scores)
        elif params.score_strategy == "avg":
            score = sum(scores) / len(scores)
        elif params.score_strategy == "weighted":
            # Weight by duration
            total_duration = sum(e.duration_ms for e in events)
            if total_duration > 0:
                score = sum(e.score * (e.duration_ms / total_duration) for e in events)
            else:
                score = sum(scores) / len(scores)
        else:
            score = max(scores)

        # Take class from first event (all same class)
        representative = events[0]

        # Merge source windows
        all_windows = []
        for e in events:
            all_windows.extend(e.source_windows)

        # Merge frequency info
        min_freq = min(
            (e.min_freq_hz for e in events if e.min_freq_hz is not None),
            default=None,
        )
        max_freq = max(
            (e.max_freq_hz for e in events if e.max_freq_hz is not None),
            default=None,
        )

        return AudioEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            class_id=representative.class_id,
            class_name=representative.class_name,
            score=score,
            min_freq_hz=min_freq,
            max_freq_hz=max_freq,
            source_windows=list(set(all_windows)),
            metadata={
                "merged_count": len(events),
                "merge_strategy": params.score_strategy,
            },
        )
