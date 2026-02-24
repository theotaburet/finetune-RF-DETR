"""Tests for merge parameter optimizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
)
from rf_detr_finetuning.eventprocessor.optimizer import (
    GlobalSearchSpace,
    SearchSpace,
    _apply_merge,
    _build_config,
    optimize_default_only,
    optimize_per_class,
    save_config_yaml,
)


def _evt(start_ms: float, end_ms: float, class_id: int = 0, **kwargs) -> AudioEvent:
    """Shorthand to build an AudioEvent."""
    return AudioEvent(start_ms=start_ms, end_ms=end_ms, class_id=class_id, **kwargs)


def _make_raw_events(events: list[AudioEvent]) -> EventList:
    """Wrap events in an EventList."""
    return EventList(events=events)


class TestBuildConfig:
    """Tests for _build_config helper."""

    def test_default_only(self) -> None:
        config = _build_config(
            class_ids=[],
            class_params_values={},
            default_params_values={"delta_time_ms": 500.0, "delta_freq_hz": 200.0},
            score_threshold=0.3,
            min_duration_ms=100.0,
        )
        assert config.default_params.delta_time_ms == 500.0
        assert config.default_params.delta_freq_hz == 200.0
        assert config.score_threshold == 0.3
        assert config.min_duration_ms == 100.0
        assert config.class_params == {}

    def test_with_class_overrides(self) -> None:
        config = _build_config(
            class_ids=[0, 1],
            class_params_values={
                0: {"delta_time_ms": 100.0, "delta_freq_hz": 50.0},
                1: {"delta_time_ms": 2000.0, "delta_freq_hz": 1000.0},
            },
            default_params_values={"delta_time_ms": 500.0, "delta_freq_hz": 200.0},
            score_threshold=0.0,
            min_duration_ms=0.0,
        )
        assert len(config.class_params) == 2
        assert config.class_params[0].delta_time_ms == 100.0
        assert config.class_params[1].delta_time_ms == 2000.0


class TestApplyMerge:
    """Tests for _apply_merge."""

    def test_merges_nearby_events(self) -> None:
        events = _make_raw_events([_evt(0, 100, class_id=0), _evt(150, 250, class_id=0)])
        config = ClassWiseMergeConfig(
            default_params=ClassMergeParams(delta_time_ms=100.0, delta_freq_hz=500.0),
        )
        merged = _apply_merge([events], config)
        assert len(merged) == 1
        # 50ms gap <= 100ms delta_time => merged into one event
        assert len(merged[0]) == 1

    def test_does_not_merge_distant_events(self) -> None:
        events = _make_raw_events([_evt(0, 100, class_id=0), _evt(500, 600, class_id=0)])
        config = ClassWiseMergeConfig(
            default_params=ClassMergeParams(delta_time_ms=100.0, delta_freq_hz=500.0),
        )
        merged = _apply_merge([events], config)
        assert len(merged[0]) == 2

    def test_applies_score_filtering(self) -> None:
        events = _make_raw_events([_evt(0, 100, class_id=0, score=0.1), _evt(200, 300, class_id=0, score=0.9)])
        config = ClassWiseMergeConfig(
            default_params=ClassMergeParams(delta_time_ms=100.0, delta_freq_hz=500.0),
            score_threshold=0.5,
        )
        merged = _apply_merge([events], config)
        assert len(merged[0]) == 1
        assert merged[0][0].score >= 0.5


class TestOptimizeDefaultOnly:
    """Tests for optimize_default_only grid search."""

    def test_finds_best_params(self) -> None:
        """With detections that perfectly match GT, should find params that give F1=1."""
        raw = [_make_raw_events([_evt(0, 100, class_id=0), _evt(200, 300, class_id=0)])]
        gts = [[_evt(0, 100, class_id=0), _evt(200, 300, class_id=0)]]

        result = optimize_default_only(
            raw,
            gts,
            default_search=SearchSpace(delta_time_ms=[50.0], delta_freq_hz=[100.0]),
            global_search=GlobalSearchSpace(score_threshold=[0.0], min_duration_ms=[0.0]),
            iou_threshold=0.5,
        )

        assert result.best_f1 == pytest.approx(1.0)
        assert result.best_config is not None
        assert result.trials_evaluated == 1
        assert result.best_eval is not None
        assert result.best_eval.overall.f1 == pytest.approx(1.0)

    def test_score_threshold_filters_low_confidence(self) -> None:
        """Score threshold should filter out low-confidence FP detections."""
        raw = [
            _make_raw_events(
                [
                    _evt(0, 100, class_id=0, score=0.9),  # TP
                    _evt(500, 600, class_id=0, score=0.1),  # FP (low score)
                ]
            )
        ]
        gts = [[_evt(0, 100, class_id=0)]]

        result = optimize_default_only(
            raw,
            gts,
            default_search=SearchSpace(delta_time_ms=[50.0], delta_freq_hz=[100.0]),
            global_search=GlobalSearchSpace(score_threshold=[0.0, 0.5], min_duration_ms=[0.0]),
            iou_threshold=0.5,
        )

        # The optimizer should prefer score_threshold=0.5 which removes the FP
        assert result.best_config is not None
        assert result.best_config.score_threshold == 0.5
        assert result.best_f1 == pytest.approx(1.0)


class TestOptimizePerClass:
    """Tests for optimize_per_class two-phase optimization."""

    def test_per_class_improves_on_default(self) -> None:
        """Per-class optimization should be able to improve on defaults.

        Scenario: class 0 needs tight merging, class 1 needs loose merging.

        """
        raw = [
            _make_raw_events(
                [
                    # Class 0: two close events that should stay separate
                    _evt(0, 100, class_id=0),
                    _evt(110, 200, class_id=0),
                    # Class 1: two close events that should merge
                    _evt(300, 400, class_id=1),
                    _evt(450, 550, class_id=1),
                ]
            )
        ]

        # GT: class 0 has two separate events, class 1 has one long event
        gts = [
            [
                _evt(0, 100, class_id=0),
                _evt(110, 200, class_id=0),
                _evt(300, 550, class_id=1),
            ]
        ]

        result = optimize_per_class(
            raw,
            gts,
            class_ids=[0, 1],
            default_search=SearchSpace(delta_time_ms=[5.0, 100.0], delta_freq_hz=[500.0]),
            global_search=GlobalSearchSpace(score_threshold=[0.0], min_duration_ms=[0.0]),
            iou_threshold=0.3,
        )

        assert result.best_config is not None
        assert result.best_f1 > 0
        assert result.best_eval is not None
        assert result.best_eval.num_files == 1


class TestSaveConfigYaml:
    """Tests for save_config_yaml."""

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        import yaml

        config = ClassWiseMergeConfig(
            default_params=ClassMergeParams(delta_time_ms=500.0, delta_freq_hz=200.0, score_strategy="max"),
            class_params={0: ClassMergeParams(delta_time_ms=100.0, delta_freq_hz=50.0, score_strategy="avg")},
            score_threshold=0.3,
            min_duration_ms=100.0,
        )

        output_path = tmp_path / "merging.yaml"
        save_config_yaml(config, str(output_path), class_names={0: "whale"})

        assert output_path.exists()
        content = output_path.read_text()
        assert "delta_time_ms: 500.0" in content
        assert "whale" in content

        # Should be valid YAML (parseable)
        data = yaml.safe_load(content)
        assert data["default"]["delta_time_ms"] == 500.0
        assert data["filtering"]["score_threshold"] == 0.3

    def test_round_trip_through_from_dict(self, tmp_path: Path) -> None:
        """Config saved to YAML can be loaded back via ClassWiseMergeConfig.from_dict."""
        import yaml

        config = ClassWiseMergeConfig(
            default_params=ClassMergeParams(delta_time_ms=1000.0, delta_freq_hz=300.0, score_strategy="max"),
            class_params={
                0: ClassMergeParams(delta_time_ms=200.0, delta_freq_hz=100.0, score_strategy="avg"),
            },
            score_threshold=0.2,
            min_duration_ms=50.0,
        )

        output_path = tmp_path / "test.yaml"
        save_config_yaml(config, str(output_path))

        with open(output_path) as f:
            loaded_data = yaml.safe_load(f)

        restored = ClassWiseMergeConfig.from_dict(loaded_data)
        assert restored.default_params.delta_time_ms == 1000.0
        assert restored.default_params.delta_freq_hz == 300.0
        assert restored.score_threshold == 0.2
        assert restored.min_duration_ms == 50.0
        assert 0 in restored.class_params
        assert restored.class_params[0].delta_time_ms == 200.0
