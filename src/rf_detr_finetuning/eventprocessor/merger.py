"""Event merging strategies.

Provides algorithms for merging overlapping detections:
- NMS-based merging
- Temporal IoU merging
- Soft-NMS

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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


def soft_nms(
    events: EventList,
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> EventList:
    """Apply Soft-NMS to events.

    Instead of hard suppression, reduces scores of overlapping events.

    Args:
        events: Input events.
        iou_threshold: IoU threshold for score decay.
        sigma: Gaussian decay parameter.
        score_threshold: Minimum score to keep.

    Returns:
        Events with adjusted scores.

    """
    import math

    if len(events) == 0:
        return events

    # Copy events to avoid modifying originals
    result_events = [
        AudioEvent(
            start_ms=e.start_ms,
            end_ms=e.end_ms,
            class_id=e.class_id,
            class_name=e.class_name,
            score=e.score,
            min_freq_hz=e.min_freq_hz,
            max_freq_hz=e.max_freq_hz,
            source_windows=e.source_windows.copy(),
            metadata=e.metadata.copy(),
        )
        for e in events
    ]

    # Sort by score
    result_events.sort(key=lambda e: e.score, reverse=True)

    keep = []
    while result_events:
        # Take highest scoring event
        best = result_events.pop(0)
        keep.append(best)

        # Decay scores of overlapping events
        for other in result_events:
            iou = best.temporal_iou(other)
            if iou > iou_threshold:
                # Gaussian decay
                weight = math.exp(-(iou * iou) / sigma)
                other.score *= weight

        # Re-sort and filter
        result_events = [e for e in result_events if e.score >= score_threshold]
        result_events.sort(key=lambda e: e.score, reverse=True)

    return EventList(
        events=keep,
        audio_path=events.audio_path,
        duration_ms=events.duration_ms,
        class_names=events.class_names,
    ).sort_by_time()
