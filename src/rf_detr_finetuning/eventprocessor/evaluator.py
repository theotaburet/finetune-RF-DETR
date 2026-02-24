"""Audio-level temporal event evaluation.

Matches predicted AudioEvents against ground truth events using temporal IoU and computes precision, recall, F1 per
class and overall.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a single predicted event to ground truth.

    Attributes:
        pred_event: The predicted event.
        gt_event: The matched ground truth event (None if false positive).
        iou: Temporal IoU between pred and matched GT.

    """

    pred_event: AudioEvent
    gt_event: AudioEvent | None = None
    iou: float = 0.0


@dataclass
class ClassMetrics:
    """Evaluation metrics for a single class.

    Attributes:
        class_id: Class index.
        class_name: Class name.
        true_positives: Number of true positives.
        false_positives: Number of false positives.
        false_negatives: Number of false negatives (missed GT events).
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        avg_iou: Average IoU of matched pairs.

    """

    class_id: int
    class_name: str = ""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_iou: float = 0.0

    def compute(self) -> None:
        """Compute precision, recall, F1 from TP/FP/FN counts."""
        tp, fp, fn = self.true_positives, self.false_positives, self.false_negatives
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_iou": round(self.avg_iou, 4),
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result across all files and classes.

    Attributes:
        per_class: Per-class metrics.
        overall: Aggregate metrics across all classes.
        num_files: Number of audio files evaluated.
        total_predictions: Total predicted events.
        total_ground_truths: Total ground truth events.
        matches: Detailed match results (optional).

    """

    per_class: dict[int, ClassMetrics] = field(default_factory=dict)
    overall: ClassMetrics = field(default_factory=lambda: ClassMetrics(class_id=-1, class_name="overall"))
    num_files: int = 0
    total_predictions: int = 0
    total_ground_truths: int = 0
    matches: list[MatchResult] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall.to_dict(),
            "per_class": {str(k): v.to_dict() for k, v in sorted(self.per_class.items())},
            "num_files": self.num_files,
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
        }


def _compute_temporal_iou(pred: AudioEvent, gt: AudioEvent) -> float:
    """Compute temporal IoU between predicted and ground truth events.

    Args:
        pred: Predicted event.
        gt: Ground truth event.

    Returns:
        Temporal IoU in [0, 1].

    """
    intersection_start = max(pred.start_ms, gt.start_ms)
    intersection_end = min(pred.end_ms, gt.end_ms)
    intersection = max(0.0, intersection_end - intersection_start)

    pred_duration = pred.end_ms - pred.start_ms
    gt_duration = gt.end_ms - gt.start_ms
    union = pred_duration + gt_duration - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _compute_temporal_iou_with_freq(pred: AudioEvent, gt: AudioEvent) -> float:
    """Compute 2D IoU using both temporal and frequency overlap.

    Falls back to temporal-only IoU if frequency info is missing on either event.

    Args:
        pred: Predicted event.
        gt: Ground truth event.

    Returns:
        IoU in [0, 1].

    """
    has_freq = (
        pred.min_freq_hz is not None
        and pred.max_freq_hz is not None
        and gt.min_freq_hz is not None
        and gt.max_freq_hz is not None
    )

    if not has_freq:
        return _compute_temporal_iou(pred, gt)

    # Time overlap
    t_start = max(pred.start_ms, gt.start_ms)
    t_end = min(pred.end_ms, gt.end_ms)
    t_overlap = max(0.0, t_end - t_start)

    # Frequency overlap
    f_start = max(pred.min_freq_hz, gt.min_freq_hz)
    f_end = min(pred.max_freq_hz, gt.max_freq_hz)
    f_overlap = max(0.0, f_end - f_start)

    intersection = t_overlap * f_overlap

    pred_area = (pred.end_ms - pred.start_ms) * (pred.max_freq_hz - pred.min_freq_hz)
    gt_area = (gt.end_ms - gt.start_ms) * (gt.max_freq_hz - gt.min_freq_hz)
    union = pred_area + gt_area - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def match_events(
    predictions: list[AudioEvent],
    ground_truths: list[AudioEvent],
    iou_threshold: float = 0.3,
    match_class: bool = True,
    use_freq: bool = False,
) -> tuple[list[MatchResult], list[AudioEvent]]:
    """Match predicted events to ground truth using greedy temporal IoU matching.

    Each GT event can match at most one prediction (highest IoU first).
    Each prediction can match at most one GT event.

    Args:
        predictions: Predicted events.
        ground_truths: Ground truth events.
        iou_threshold: Minimum IoU for a valid match.
        match_class: Require class_id to match for a valid pair.
        use_freq: Use 2D (time+freq) IoU instead of temporal-only.

    Returns:
        Tuple of (matched_predictions, unmatched_ground_truths).
        Each MatchResult in matched_predictions has .gt_event set if it was a TP.

    """
    iou_fn = _compute_temporal_iou_with_freq if use_freq else _compute_temporal_iou

    # Compute all pairwise IoUs
    iou_pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if match_class and pred.class_id != gt.class_id:
                continue
            iou = iou_fn(pred, gt)
            if iou >= iou_threshold:
                iou_pairs.append((iou, pi, gi))

    # Greedy matching: highest IoU first
    iou_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    results: list[MatchResult] = []

    for iou, pi, gi in iou_pairs:
        if pi in matched_preds or gi in matched_gts:
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        results.append(MatchResult(pred_event=predictions[pi], gt_event=ground_truths[gi], iou=iou))

    # Add unmatched predictions as false positives
    for pi, pred in enumerate(predictions):
        if pi not in matched_preds:
            results.append(MatchResult(pred_event=pred, gt_event=None, iou=0.0))

    # Collect unmatched ground truths (false negatives)
    unmatched_gts = [gt for gi, gt in enumerate(ground_truths) if gi not in matched_gts]

    return results, unmatched_gts


def evaluate_events(
    predictions: EventList | list[AudioEvent],
    ground_truths: EventList | list[AudioEvent],
    iou_threshold: float = 0.3,
    match_class: bool = True,
    use_freq: bool = False,
    class_names: dict[int, str] | None = None,
) -> EvaluationResult:
    """Evaluate predicted events against ground truth for a single audio file.

    Args:
        predictions: Predicted events.
        ground_truths: Ground truth events.
        iou_threshold: Minimum IoU for a valid match.
        match_class: Require class_id match.
        use_freq: Use 2D IoU (time+freq).
        class_names: Optional class ID to name mapping.

    Returns:
        EvaluationResult with per-class and overall metrics.

    """
    pred_list = list(predictions) if isinstance(predictions, EventList) else predictions
    gt_list = list(ground_truths) if isinstance(ground_truths, EventList) else ground_truths
    class_names = class_names or {}

    matches, unmatched_gts = match_events(
        pred_list, gt_list, iou_threshold=iou_threshold, match_class=match_class, use_freq=use_freq
    )

    # Collect all class IDs
    all_class_ids = set()
    for e in pred_list:
        all_class_ids.add(e.class_id)
    for e in gt_list:
        all_class_ids.add(e.class_id)

    # Initialize per-class metrics
    result = EvaluationResult(
        total_predictions=len(pred_list),
        total_ground_truths=len(gt_list),
        num_files=1,
        matches=matches,
    )

    for cid in all_class_ids:
        result.per_class[cid] = ClassMetrics(
            class_id=cid,
            class_name=class_names.get(cid, f"class_{cid}"),
        )

    # Count TPs and FPs from matches
    iou_sums: dict[int, float] = {cid: 0.0 for cid in all_class_ids}
    for m in matches:
        cid = m.pred_event.class_id
        if cid not in result.per_class:
            result.per_class[cid] = ClassMetrics(class_id=cid, class_name=class_names.get(cid, f"class_{cid}"))
        if m.gt_event is not None:
            result.per_class[cid].true_positives += 1
            iou_sums[cid] = iou_sums.get(cid, 0.0) + m.iou
        else:
            result.per_class[cid].false_positives += 1

    # Count FNs from unmatched ground truths
    for gt in unmatched_gts:
        cid = gt.class_id
        if cid not in result.per_class:
            result.per_class[cid] = ClassMetrics(class_id=cid, class_name=class_names.get(cid, f"class_{cid}"))
        result.per_class[cid].false_negatives += 1

    # Compute per-class metrics
    for cid, metrics in result.per_class.items():
        metrics.compute()
        if metrics.true_positives > 0:
            metrics.avg_iou = iou_sums.get(cid, 0.0) / metrics.true_positives

    # Compute overall metrics (micro-average)
    total_tp = sum(m.true_positives for m in result.per_class.values())
    total_fp = sum(m.false_positives for m in result.per_class.values())
    total_fn = sum(m.false_negatives for m in result.per_class.values())
    total_iou = sum(iou_sums.values())

    result.overall = ClassMetrics(
        class_id=-1,
        class_name="overall",
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )
    result.overall.compute()
    if total_tp > 0:
        result.overall.avg_iou = total_iou / total_tp

    return result


def evaluate_multi_file(
    file_results: list[tuple[EventList | list[AudioEvent], EventList | list[AudioEvent]]],
    iou_threshold: float = 0.3,
    match_class: bool = True,
    use_freq: bool = False,
    class_names: dict[int, str] | None = None,
) -> EvaluationResult:
    """Evaluate across multiple audio files by aggregating TP/FP/FN counts.

    Args:
        file_results: List of (predictions, ground_truths) pairs per audio file.
        iou_threshold: Minimum IoU for a valid match.
        match_class: Require class_id match.
        use_freq: Use 2D IoU.
        class_names: Optional class name mapping.

    Returns:
        Aggregated EvaluationResult.

    """
    class_names = class_names or {}
    aggregated = EvaluationResult(num_files=len(file_results))
    iou_sums: dict[int, float] = {}

    for preds, gts in file_results:
        single = evaluate_events(
            preds, gts, iou_threshold=iou_threshold, match_class=match_class, use_freq=use_freq, class_names=class_names
        )

        aggregated.total_predictions += single.total_predictions
        aggregated.total_ground_truths += single.total_ground_truths

        for cid, metrics in single.per_class.items():
            if cid not in aggregated.per_class:
                aggregated.per_class[cid] = ClassMetrics(
                    class_id=cid,
                    class_name=class_names.get(cid, f"class_{cid}"),
                )
            aggregated.per_class[cid].true_positives += metrics.true_positives
            aggregated.per_class[cid].false_positives += metrics.false_positives
            aggregated.per_class[cid].false_negatives += metrics.false_negatives
            iou_sums[cid] = iou_sums.get(cid, 0.0) + metrics.avg_iou * metrics.true_positives

    # Compute final per-class metrics
    for cid, metrics in aggregated.per_class.items():
        metrics.compute()
        if metrics.true_positives > 0:
            metrics.avg_iou = iou_sums[cid] / metrics.true_positives

    # Compute overall
    total_tp = sum(m.true_positives for m in aggregated.per_class.values())
    total_fp = sum(m.false_positives for m in aggregated.per_class.values())
    total_fn = sum(m.false_negatives for m in aggregated.per_class.values())
    total_iou = sum(iou_sums.values())

    aggregated.overall = ClassMetrics(
        class_id=-1,
        class_name="overall",
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )
    aggregated.overall.compute()
    if total_tp > 0:
        aggregated.overall.avg_iou = total_iou / total_tp

    return aggregated
