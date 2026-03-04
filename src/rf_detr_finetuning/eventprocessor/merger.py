"""Event merging strategies.

Provides algorithms for merging overlapping detections:
- Cluster-based merging (Union-Find) with 2D time-frequency IoU
- NMS-based merging (greedy, legacy)
- Temporal IoU merging

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList

logger = logging.getLogger(__name__)


class _UnionFind:
    """Union-Find (disjoint set) data structure for clustering events."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> dict[int, list[int]]:
        """Return mapping from root -> list of member indices."""
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


@dataclass
class MergeConfig:
    """Configuration for event merging.

    Attributes:
        iou_threshold: IoU threshold for considering overlap.
        score_threshold: Minimum score to keep events.
        merge_same_class_only: Only merge events of same class.
        merge_strategy: How to combine scores ('max', 'avg', 'sum').
        gap_tolerance_ms: Merge events within this gap (temporal).
        use_2d_iou: Use 2D time-frequency IoU instead of temporal-only.

    """

    iou_threshold: float = 0.5
    score_threshold: float = 0.0
    merge_same_class_only: bool = True
    merge_strategy: Literal["max", "avg", "sum"] = "max"
    gap_tolerance_ms: float = 0.0
    use_2d_iou: bool = True


class EventMerger:
    """Merges overlapping audio events using Union-Find clustering.

    Uses transitive clustering: if event A overlaps B and B overlaps C,
    all three are merged into a single event. This correctly handles
    long events detected across multiple overlapping windows.

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
        """Merge overlapping events using Union-Find clustering.

        Args:
            events: Input event list.

        Returns:
            Merged event list.

        """
        if len(events) == 0:
            return events

        event_list = list(events)
        n = len(event_list)
        uf = _UnionFind(n)

        # Build connectivity graph: union events that should merge
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(event_list[i], event_list[j]):
                    uf.union(i, j)

        # Merge each cluster
        merged = []
        for indices in uf.clusters().values():
            group = [event_list[i] for i in indices]
            merged.append(self._merge_group(group))

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

    def _should_merge(self, a: AudioEvent, b: AudioEvent) -> bool:
        """Determine whether two events should be merged.

        Args:
            a: First event.
            b: Second event.

        Returns:
            True if the events should be merged.

        """
        # Check class constraint
        if self.config.merge_same_class_only and a.class_id != b.class_id:
            return False

        # Check IoU overlap (2D or temporal-only)
        if self.config.use_2d_iou:
            iou = a.temporal_frequency_iou(b)
        else:
            iou = a.temporal_iou(b)

        # Use strict > 0 when threshold is 0.0 to avoid merging
        # non-overlapping events; otherwise use >= threshold.
        if self.config.iou_threshold > 0:
            if iou >= self.config.iou_threshold:
                return True
        else:
            if iou > 0:
                return True

        # Check gap tolerance (temporal proximity without overlap)
        if self.config.gap_tolerance_ms > 0:
            gap = self._compute_gap(a, b)
            if gap <= self.config.gap_tolerance_ms:
                return True

        return False

    def _compute_gap(self, a: AudioEvent, b: AudioEvent) -> float:
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
        all_windows: list[int] = []
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
            source_windows=sorted(set(all_windows)),
            metadata={"merged_count": len(events)},
        )


def nms_merge(
    events: EventList,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    use_2d_iou: bool = True,
) -> EventList:
    """Apply cluster-based merging to events.

    Unlike traditional greedy NMS, this uses Union-Find to transitively
    merge all connected events, which correctly handles long events
    detected across multiple overlapping windows.

    Args:
        events: Input events.
        iou_threshold: IoU threshold for merging.
        score_threshold: Minimum score to keep.
        use_2d_iou: Use 2D time-frequency IoU.

    Returns:
        Merged events.

    """
    config = MergeConfig(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        merge_strategy="max",
        use_2d_iou=use_2d_iou,
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
        iou_threshold=0.0,
        gap_tolerance_ms=gap_tolerance_ms,
        merge_same_class_only=merge_same_class_only,
        merge_strategy="avg",
        use_2d_iou=False,
    )
    merger = EventMerger(config)
    return merger.merge(events)


def cluster_merge(
    events: EventList,
    iou_threshold: float = 0.3,
    gap_tolerance_ms: float = 0.0,
    merge_same_class_only: bool = True,
    use_2d_iou: bool = True,
) -> EventList:
    """Merge overlapping events using transitive clustering.

    Designed for merging detections from overlapping spectrogram windows.
    Uses Union-Find to build clusters where events are connected if they
    share sufficient IoU overlap or are within a temporal gap.

    This handles the chain case correctly: if window 1 detects 1000-3200ms,
    window 2 detects 2560-5760ms, and window 3 detects 5120-6500ms, all three
    are transitively merged into a single event spanning 1000-6500ms.

    Args:
        events: Input events.
        iou_threshold: IoU threshold for connecting events.
        gap_tolerance_ms: Additional gap tolerance for bridging.
        merge_same_class_only: Only merge events of the same class.
        use_2d_iou: Use 2D time-frequency IoU (recommended).

    Returns:
        Merged events sorted by time.

    """
    config = MergeConfig(
        iou_threshold=iou_threshold,
        gap_tolerance_ms=gap_tolerance_ms,
        merge_same_class_only=merge_same_class_only,
        merge_strategy="max",
        use_2d_iou=use_2d_iou,
    )
    merger = EventMerger(config)
    return merger.merge(events)
