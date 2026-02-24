"""Tests for audio-level temporal event evaluation."""

from __future__ import annotations

import pytest

from rf_detr_finetuning.eventprocessor.evaluator import (
    ClassMetrics,
    _compute_temporal_iou,
    _compute_temporal_iou_with_freq,
    evaluate_events,
    evaluate_multi_file,
    match_events,
)
from rf_detr_finetuning.eventprocessor.event import AudioEvent


def _evt(start_ms: float, end_ms: float, class_id: int = 0, **kwargs) -> AudioEvent:
    """Shorthand to build an AudioEvent."""
    return AudioEvent(start_ms=start_ms, end_ms=end_ms, class_id=class_id, **kwargs)


class TestComputeTemporalIoU:
    """Tests for _compute_temporal_iou."""

    def test_identical_events(self) -> None:
        a = _evt(100, 200)
        b = _evt(100, 200)
        assert _compute_temporal_iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = _evt(100, 200)
        b = _evt(300, 400)
        assert _compute_temporal_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = _evt(0, 100)
        b = _evt(50, 150)
        # intersection = 50, union = 100 + 100 - 50 = 150
        assert _compute_temporal_iou(a, b) == pytest.approx(50 / 150)

    def test_contained_event(self) -> None:
        a = _evt(0, 200)
        b = _evt(50, 100)
        # intersection = 50, union = 200 + 50 - 50 = 200
        assert _compute_temporal_iou(a, b) == pytest.approx(50 / 200)

    def test_adjacent_events_zero_iou(self) -> None:
        a = _evt(0, 100)
        b = _evt(100, 200)
        # intersection = 0, union = 200
        assert _compute_temporal_iou(a, b) == 0.0

    def test_zero_duration_event(self) -> None:
        a = _evt(100, 100)
        b = _evt(100, 200)
        assert _compute_temporal_iou(a, b) == 0.0


class TestComputeTemporalIoUWithFreq:
    """Tests for _compute_temporal_iou_with_freq (2D IoU)."""

    def test_falls_back_to_temporal_when_no_freq(self) -> None:
        a = _evt(0, 100)
        b = _evt(50, 150)
        assert _compute_temporal_iou_with_freq(a, b) == pytest.approx(_compute_temporal_iou(a, b))

    def test_identical_2d_events(self) -> None:
        a = _evt(0, 100, min_freq_hz=200, max_freq_hz=500)
        b = _evt(0, 100, min_freq_hz=200, max_freq_hz=500)
        assert _compute_temporal_iou_with_freq(a, b) == pytest.approx(1.0)

    def test_no_temporal_overlap_with_freq(self) -> None:
        a = _evt(0, 100, min_freq_hz=200, max_freq_hz=500)
        b = _evt(200, 300, min_freq_hz=200, max_freq_hz=500)
        assert _compute_temporal_iou_with_freq(a, b) == 0.0

    def test_no_freq_overlap(self) -> None:
        a = _evt(0, 100, min_freq_hz=100, max_freq_hz=200)
        b = _evt(0, 100, min_freq_hz=300, max_freq_hz=400)
        assert _compute_temporal_iou_with_freq(a, b) == 0.0

    def test_partial_2d_overlap(self) -> None:
        a = _evt(0, 100, min_freq_hz=0, max_freq_hz=200)
        b = _evt(50, 150, min_freq_hz=100, max_freq_hz=300)
        # t_overlap = 50, f_overlap = 100
        # intersection = 50 * 100 = 5000
        # pred_area = 100 * 200 = 20000, gt_area = 100 * 200 = 20000
        # union = 20000 + 20000 - 5000 = 35000
        assert _compute_temporal_iou_with_freq(a, b) == pytest.approx(5000 / 35000)

    def test_partial_freq_only_on_pred(self) -> None:
        """Falls back to temporal-only if one side has no freq."""
        a = _evt(0, 100, min_freq_hz=100, max_freq_hz=200)
        b = _evt(50, 150)
        assert _compute_temporal_iou_with_freq(a, b) == pytest.approx(_compute_temporal_iou(a, b))


class TestMatchEvents:
    """Tests for match_events greedy matching."""

    def test_perfect_match(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        gts = [_evt(0, 100, class_id=0)]
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0].gt_event is not None
        assert matches[0].iou == pytest.approx(1.0)
        assert len(unmatched_gts) == 0

    def test_no_match_below_threshold(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        gts = [_evt(80, 200, class_id=0)]
        # IoU = 20 / (100+120-20) = 20/200 = 0.1
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0].gt_event is None  # FP
        assert len(unmatched_gts) == 1

    def test_class_mismatch_not_matched(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        gts = [_evt(0, 100, class_id=1)]
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.5, match_class=True)
        assert matches[0].gt_event is None
        assert len(unmatched_gts) == 1

    def test_class_mismatch_matches_when_disabled(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        gts = [_evt(0, 100, class_id=1)]
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.5, match_class=False)
        assert matches[0].gt_event is not None
        assert len(unmatched_gts) == 0

    def test_greedy_one_to_one_matching(self) -> None:
        """Each GT can only match one pred and vice versa."""
        preds = [_evt(0, 100, class_id=0), _evt(10, 110, class_id=0)]
        gts = [_evt(0, 100, class_id=0)]
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.3)
        tp_count = sum(1 for m in matches if m.gt_event is not None)
        fp_count = sum(1 for m in matches if m.gt_event is None)
        assert tp_count == 1
        assert fp_count == 1
        assert len(unmatched_gts) == 0

    def test_empty_predictions(self) -> None:
        matches, unmatched_gts = match_events([], [_evt(0, 100)], iou_threshold=0.3)
        assert len(matches) == 0
        assert len(unmatched_gts) == 1

    def test_empty_ground_truths(self) -> None:
        matches, unmatched_gts = match_events([_evt(0, 100)], [], iou_threshold=0.3)
        assert len(matches) == 1
        assert matches[0].gt_event is None
        assert len(unmatched_gts) == 0

    def test_multiple_classes(self) -> None:
        """Multiple classes matched independently."""
        preds = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=1)]
        gts = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=1)]
        matches, unmatched_gts = match_events(preds, gts, iou_threshold=0.5)
        tp_count = sum(1 for m in matches if m.gt_event is not None)
        assert tp_count == 2
        assert len(unmatched_gts) == 0


class TestClassMetrics:
    """Tests for ClassMetrics.compute()."""

    def test_perfect_precision_recall(self) -> None:
        m = ClassMetrics(class_id=0, true_positives=5, false_positives=0, false_negatives=0)
        m.compute()
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_no_predictions(self) -> None:
        m = ClassMetrics(class_id=0, true_positives=0, false_positives=0, false_negatives=5)
        m.compute()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_positives(self) -> None:
        m = ClassMetrics(class_id=0, true_positives=0, false_positives=5, false_negatives=0)
        m.compute()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_mixed_tp_fp_fn(self) -> None:
        m = ClassMetrics(class_id=0, true_positives=3, false_positives=2, false_negatives=1)
        m.compute()
        assert m.precision == pytest.approx(3 / 5)
        assert m.recall == pytest.approx(3 / 4)
        expected_f1 = 2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4))
        assert m.f1 == pytest.approx(expected_f1)


class TestEvaluateEvents:
    """Tests for evaluate_events single-file evaluation."""

    def test_perfect_single_class(self) -> None:
        preds = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=0)]
        gts = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=0)]
        result = evaluate_events(preds, gts, iou_threshold=0.5)
        assert result.overall.precision == 1.0
        assert result.overall.recall == 1.0
        assert result.overall.f1 == 1.0
        assert result.overall.true_positives == 2

    def test_one_miss_one_fp(self) -> None:
        preds = [_evt(0, 100, class_id=0), _evt(500, 600, class_id=0)]
        gts = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=0)]
        result = evaluate_events(preds, gts, iou_threshold=0.5)
        assert result.overall.true_positives == 1
        assert result.overall.false_positives == 1
        assert result.overall.false_negatives == 1
        assert result.overall.precision == pytest.approx(0.5)
        assert result.overall.recall == pytest.approx(0.5)

    def test_multi_class(self) -> None:
        preds = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=1)]
        gts = [_evt(0, 100, class_id=0), _evt(200, 300, class_id=1)]
        result = evaluate_events(preds, gts, iou_threshold=0.5)
        assert result.overall.f1 == 1.0
        assert 0 in result.per_class
        assert 1 in result.per_class
        assert result.per_class[0].true_positives == 1
        assert result.per_class[1].true_positives == 1

    def test_empty_predictions(self) -> None:
        gts = [_evt(0, 100, class_id=0)]
        result = evaluate_events([], gts, iou_threshold=0.5)
        assert result.overall.precision == 0.0
        assert result.overall.recall == 0.0
        assert result.overall.false_negatives == 1

    def test_empty_ground_truths(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        result = evaluate_events(preds, [], iou_threshold=0.5)
        assert result.overall.precision == 0.0
        assert result.overall.false_positives == 1

    def test_avg_iou_computed(self) -> None:
        preds = [_evt(0, 100, class_id=0)]
        gts = [_evt(0, 100, class_id=0)]
        result = evaluate_events(preds, gts, iou_threshold=0.5)
        assert result.overall.avg_iou == pytest.approx(1.0)
        assert result.per_class[0].avg_iou == pytest.approx(1.0)


class TestEvaluateMultiFile:
    """Tests for evaluate_multi_file aggregation."""

    def test_aggregates_two_files(self) -> None:
        file1_preds = [_evt(0, 100, class_id=0)]
        file1_gts = [_evt(0, 100, class_id=0)]
        file2_preds = [_evt(0, 100, class_id=0)]
        file2_gts = [_evt(0, 100, class_id=0)]
        result = evaluate_multi_file(
            [(file1_preds, file1_gts), (file2_preds, file2_gts)],
            iou_threshold=0.5,
        )
        assert result.num_files == 2
        assert result.overall.true_positives == 2
        assert result.overall.f1 == 1.0

    def test_mixed_results_across_files(self) -> None:
        # File 1: perfect
        file1_preds = [_evt(0, 100, class_id=0)]
        file1_gts = [_evt(0, 100, class_id=0)]
        # File 2: miss
        file2_preds = []
        file2_gts = [_evt(0, 100, class_id=0)]
        result = evaluate_multi_file(
            [(file1_preds, file1_gts), (file2_preds, file2_gts)],
            iou_threshold=0.5,
        )
        assert result.overall.true_positives == 1
        assert result.overall.false_negatives == 1
        assert result.overall.precision == 1.0
        assert result.overall.recall == pytest.approx(0.5)
