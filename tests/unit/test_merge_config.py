"""Tests for ClassWiseMergeConfig.from_dict() and ClassMergeParams."""

from __future__ import annotations

from rf_detr_finetuning.eventprocessor.merger import ClassWiseMergeConfig


class TestClassWiseMergeConfigFromDict:
    """Tests for ClassWiseMergeConfig.from_dict()."""

    def test_empty_dict_gives_defaults(self) -> None:
        config = ClassWiseMergeConfig.from_dict({})
        assert config.default_params.delta_time_ms == 500.0
        assert config.default_params.delta_freq_hz == 500.0
        assert config.default_params.score_strategy == "max"
        assert config.class_params == {}
        assert config.score_threshold == 0.0
        assert config.min_duration_ms == 0.0
        assert config.max_duration_ms is None

    def test_custom_default_params(self) -> None:
        data = {
            "default": {
                "delta_time_ms": 200.0,
                "delta_freq_hz": 300.0,
                "score_strategy": "avg",
            }
        }
        config = ClassWiseMergeConfig.from_dict(data)
        assert config.default_params.delta_time_ms == 200.0
        assert config.default_params.delta_freq_hz == 300.0
        assert config.default_params.score_strategy == "avg"
        assert config.default_params.min_overlap_ratio is None

    def test_class_params_parsed(self) -> None:
        data = {
            "default": {"delta_time_ms": 500.0},
            "classes": {
                "0": {"delta_time_ms": 100.0, "delta_freq_hz": 200.0},
                "1": {"delta_time_ms": 1000.0},
            },
        }
        config = ClassWiseMergeConfig.from_dict(data)
        assert 0 in config.class_params
        assert 1 in config.class_params
        assert config.class_params[0].delta_time_ms == 100.0
        assert config.class_params[0].delta_freq_hz == 200.0
        assert config.class_params[1].delta_time_ms == 1000.0

    def test_class_params_inherit_from_default(self) -> None:
        """Per-class params should inherit unspecified fields from default."""
        data = {
            "default": {"delta_time_ms": 500.0, "delta_freq_hz": 700.0, "score_strategy": "weighted"},
            "classes": {
                "0": {"delta_time_ms": 100.0},
            },
        }
        config = ClassWiseMergeConfig.from_dict(data)
        # delta_time_ms was overridden
        assert config.class_params[0].delta_time_ms == 100.0
        # delta_freq_hz should fall back to default
        assert config.class_params[0].delta_freq_hz == 700.0
        # score_strategy should fall back to default
        assert config.class_params[0].score_strategy == "weighted"

    def test_string_class_ids_converted_to_int(self) -> None:
        data = {"classes": {"42": {"delta_time_ms": 100.0}}}
        config = ClassWiseMergeConfig.from_dict(data)
        assert 42 in config.class_params
        assert isinstance(list(config.class_params.keys())[0], int)

    def test_filtering_section_parsed(self) -> None:
        data = {
            "filtering": {
                "score_threshold": 0.3,
                "min_duration_ms": 100.0,
                "max_duration_ms": 60000.0,
            }
        }
        config = ClassWiseMergeConfig.from_dict(data)
        assert config.score_threshold == 0.3
        assert config.min_duration_ms == 100.0
        assert config.max_duration_ms == 60000.0

    def test_full_config(self) -> None:
        data = {
            "default": {
                "delta_time_ms": 500.0,
                "delta_freq_hz": 500.0,
                "min_overlap_ratio": 0.3,
                "score_strategy": "max",
            },
            "classes": {
                "0": {"delta_time_ms": 200.0, "delta_freq_hz": 300.0},
                "1": {"delta_time_ms": 1000.0, "delta_freq_hz": 100.0},
            },
            "filtering": {
                "score_threshold": 0.5,
                "min_duration_ms": 50.0,
                "max_duration_ms": 30000.0,
            },
        }
        config = ClassWiseMergeConfig.from_dict(data)
        assert config.default_params.min_overlap_ratio == 0.3
        assert len(config.class_params) == 2
        assert config.score_threshold == 0.5
        assert config.min_duration_ms == 50.0
        assert config.max_duration_ms == 30000.0
