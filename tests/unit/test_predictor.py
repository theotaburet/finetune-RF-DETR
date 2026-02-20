"""Tests for RFDETRPredictor._load_model() checkpoint introspection and Detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from rf_detr_finetuning.predictor.inference import Detection, PredictionResult, RFDETRPredictor


class TestDetection:
    """Tests for Detection dataclass."""

    def test_properties(self) -> None:
        det = Detection(bbox=[10.0, 20.0, 50.0, 80.0], score=0.9, class_id=1)
        assert det.x1 == 10.0
        assert det.y1 == 20.0
        assert det.x2 == 50.0
        assert det.y2 == 80.0
        assert det.width == 40.0
        assert det.height == 60.0
        assert det.center == (30.0, 50.0)
        assert det.area == 2400.0

    def test_to_coco(self) -> None:
        det = Detection(bbox=[10.0, 20.0, 50.0, 80.0], score=0.9, class_id=1)
        coco = det.to_coco()
        assert coco["bbox"] == [10.0, 20.0, 40.0, 60.0]  # x, y, w, h
        assert coco["score"] == 0.9
        assert coco["category_id"] == 1

    def test_to_dict(self) -> None:
        det = Detection(bbox=[10.0, 20.0, 50.0, 80.0], score=0.9, class_id=1, class_name="frog")
        d = det.to_dict()
        assert d["bbox"] == [10.0, 20.0, 50.0, 80.0]
        assert d["class_name"] == "frog"

    def test_class_name_default_none(self) -> None:
        det = Detection(bbox=[0, 0, 1, 1], score=0.5, class_id=0)
        assert det.class_name is None


class TestPredictionResult:
    """Tests for PredictionResult filtering."""

    def _make_result(self) -> PredictionResult:
        return PredictionResult(
            detections=[
                Detection(bbox=[0, 0, 10, 10], score=0.9, class_id=0),
                Detection(bbox=[20, 20, 40, 40], score=0.5, class_id=1),
                Detection(bbox=[50, 50, 70, 70], score=0.3, class_id=0),
            ]
        )

    def test_filter_by_score(self) -> None:
        result = self._make_result().filter_by_score(0.5)
        assert len(result) == 2
        assert all(d.score >= 0.5 for d in result.detections)

    def test_filter_by_class(self) -> None:
        result = self._make_result().filter_by_class([0])
        assert len(result) == 2
        assert all(d.class_id == 0 for d in result.detections)

    def test_len(self) -> None:
        assert len(self._make_result()) == 3


class TestRFDETRPredictorLoadModel:
    """Tests for RFDETRPredictor._load_model() checkpoint introspection."""

    @patch("rf_detr_finetuning.predictor.inference.torch.load")
    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_introspects_num_classes_from_checkpoint(
        self,
        mock_get_model_classes: MagicMock,
        mock_torch_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify num_classes is extracted from class_embed.bias shape."""
        # Create a fake checkpoint with 3-class bias (num_classes = 3-1 = 2)
        checkpoint = {
            "model": {"class_embed.bias": torch.zeros(3)},
        }
        mock_torch_load.return_value = checkpoint

        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        weights_path = tmp_path / "checkpoint.pth"
        weights_path.touch()

        RFDETRPredictor(model_size="base", weights_path=weights_path)

        # Verify model was constructed with num_classes=2
        call_kwargs = fake_model_class.call_args[1]
        assert call_kwargs["num_classes"] == 2

    @patch("rf_detr_finetuning.predictor.inference.torch.load")
    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_extracts_class_names_from_checkpoint_args(
        self,
        mock_get_model_classes: MagicMock,
        mock_torch_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify class_names are extracted from checkpoint args metadata."""
        args_mock = MagicMock()
        args_mock.class_names = ["bird", "frog"]

        checkpoint = {
            "model": {"class_embed.bias": torch.zeros(3)},
            "args": args_mock,
        }
        mock_torch_load.return_value = checkpoint

        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        weights_path = tmp_path / "checkpoint.pth"
        weights_path.touch()

        predictor = RFDETRPredictor(model_size="base", weights_path=weights_path)
        assert predictor.class_names == ["bird", "frog"]

    @patch("rf_detr_finetuning.predictor.inference.torch.load")
    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_caller_class_names_override_checkpoint(
        self,
        mock_get_model_classes: MagicMock,
        mock_torch_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Caller-provided class_names should take priority over checkpoint."""
        args_mock = MagicMock()
        args_mock.class_names = ["bird", "frog"]

        checkpoint = {
            "model": {"class_embed.bias": torch.zeros(3)},
            "args": args_mock,
        }
        mock_torch_load.return_value = checkpoint

        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        weights_path = tmp_path / "checkpoint.pth"
        weights_path.touch()

        predictor = RFDETRPredictor(
            model_size="base",
            weights_path=weights_path,
            class_names=["whale", "dolphin"],
        )
        assert predictor.class_names == ["whale", "dolphin"]

    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_unknown_model_size_raises(self, mock_get_model_classes: MagicMock) -> None:
        mock_get_model_classes.return_value = {"base": MagicMock()}
        with pytest.raises(ValueError, match="Unknown model size"):
            RFDETRPredictor(model_size="gigantic")

    @patch("rf_detr_finetuning.predictor.inference.torch.load")
    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_non_dict_checkpoint_handled_gracefully(
        self,
        mock_get_model_classes: MagicMock,
        mock_torch_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If checkpoint is not a dict, introspection should be skipped."""
        mock_torch_load.return_value = "not_a_dict"

        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        weights_path = tmp_path / "checkpoint.pth"
        weights_path.touch()

        RFDETRPredictor(model_size="base", weights_path=weights_path)
        # Should not crash; num_classes should not be passed
        call_kwargs = fake_model_class.call_args[1]
        assert "num_classes" not in call_kwargs

    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_no_weights_path_skips_introspection(self, mock_get_model_classes: MagicMock) -> None:
        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        RFDETRPredictor(model_size="base", weights_path=None)
        call_kwargs = fake_model_class.call_args[1]
        assert "num_classes" not in call_kwargs
        assert "pretrain_weights" not in call_kwargs

    @patch("rf_detr_finetuning.predictor.inference.torch.load")
    @patch("rf_detr_finetuning.trainer.rfdetr_wrapper._get_model_classes")
    def test_checkpoint_without_class_embed_skips_num_classes(
        self,
        mock_get_model_classes: MagicMock,
        mock_torch_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checkpoint without class_embed.bias should not set num_classes."""
        checkpoint = {"model": {"some_other_key": torch.zeros(10)}}
        mock_torch_load.return_value = checkpoint

        fake_model = MagicMock()
        fake_model_class = MagicMock(return_value=fake_model)
        mock_get_model_classes.return_value = {"base": fake_model_class}

        weights_path = tmp_path / "checkpoint.pth"
        weights_path.touch()

        RFDETRPredictor(model_size="base", weights_path=weights_path)
        call_kwargs = fake_model_class.call_args[1]
        assert "num_classes" not in call_kwargs
