"""Tests for rf_detr_finetuning.utils module."""

from __future__ import annotations

import json
from pathlib import Path

from rf_detr_finetuning.utils import load_class_names


class TestLoadClassNames:
    """Tests for load_class_names()."""

    def test_explicit_list_returned_as_is(self) -> None:
        result = load_class_names(class_names=["bird", "frog", "whale"])
        assert result == ["bird", "frog", "whale"]

    def test_explicit_list_takes_priority_over_file(self, tmp_path: Path) -> None:
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(["from_file"]))
        result = load_class_names(class_names=["from_arg"], classes_file=classes_file)
        assert result == ["from_arg"]

    def test_plain_list_json(self, tmp_path: Path) -> None:
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(["bird", "frog", "whale"]))
        result = load_class_names(classes_file=classes_file)
        assert result == ["bird", "frog", "whale"]

    def test_coco_format_json(self, tmp_path: Path) -> None:
        data = {
            "categories": [
                {"id": 2, "name": "whale"},
                {"id": 0, "name": "bird"},
                {"id": 1, "name": "frog"},
            ]
        }
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(data))
        result = load_class_names(classes_file=classes_file)
        # Should be sorted by id
        assert result == ["bird", "frog", "whale"]

    def test_dict_with_class_names_key(self, tmp_path: Path) -> None:
        data = {"class_names": ["bird", "frog"]}
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(data))
        result = load_class_names(classes_file=classes_file)
        assert result == ["bird", "frog"]

    def test_nonexistent_file_returns_none(self) -> None:
        result = load_class_names(classes_file="/nonexistent/path.json")
        assert result is None

    def test_no_args_returns_none(self) -> None:
        result = load_class_names()
        assert result is None

    def test_none_args_returns_none(self) -> None:
        result = load_class_names(class_names=None, classes_file=None)
        assert result is None

    def test_empty_list_falls_through_to_file(self, tmp_path: Path) -> None:
        """Empty list is falsy, so it should fall through to file."""
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(["from_file"]))
        result = load_class_names(class_names=[], classes_file=classes_file)
        assert result == ["from_file"]

    def test_empty_list_no_file_returns_none(self) -> None:
        result = load_class_names(class_names=[])
        assert result is None

    def test_dict_without_known_keys_returns_none(self, tmp_path: Path) -> None:
        data = {"unknown_key": ["bird"]}
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(data))
        result = load_class_names(classes_file=classes_file)
        assert result is None

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(["bird"]))
        result = load_class_names(classes_file=str(classes_file))
        assert result == ["bird"]

    def test_coco_format_single_category(self, tmp_path: Path) -> None:
        data = {"categories": [{"id": 0, "name": "event"}]}
        classes_file = tmp_path / "classes.json"
        classes_file.write_text(json.dumps(data))
        result = load_class_names(classes_file=classes_file)
        assert result == ["event"]
