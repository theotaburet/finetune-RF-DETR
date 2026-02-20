"""Tests for predict.py deprecation shim."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPredictionDeprecationWarning:
    """Verify that predict.prediction() emits a DeprecationWarning."""

    @patch("rf_detr_finetuning.predictor.RFDETRPredictor")
    def test_emits_deprecation_warning(self, mock_predictor_class: MagicMock) -> None:
        """Calling prediction() should trigger a DeprecationWarning."""
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor

        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.to_supervision.return_value = MagicMock()
        mock_predictor.predict.return_value = mock_result

        from rf_detr_finetuning.predict import prediction

        with patch("matplotlib.pyplot.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            with patch("supervision.BoxAnnotator") as mock_box:
                with patch("supervision.LabelAnnotator") as mock_label:
                    mock_box.return_value.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                    mock_label.return_value.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

                    # PT031: pytest.warns block should contain a single simple statement
                    def call_prediction():
                        prediction(image_path="test.jpg", model_size="base")

                    with pytest.warns(DeprecationWarning, match="prediction\\(\\) is deprecated"):
                        call_prediction()

    @patch("rf_detr_finetuning.predictor.RFDETRPredictor")
    def test_delegates_to_rfdetr_predictor(self, mock_predictor_class: MagicMock) -> None:
        """Prediction() should delegate to RFDETRPredictor."""
        import warnings

        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor

        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.to_supervision.return_value = MagicMock()
        mock_predictor.predict.return_value = mock_result

        from rf_detr_finetuning.predict import prediction

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("matplotlib.pyplot.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
                with patch("supervision.BoxAnnotator") as mock_box:
                    with patch("supervision.LabelAnnotator") as mock_label:
                        mock_box.return_value.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                        mock_label.return_value.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                        prediction(image_path="test.jpg", model_size="small", confidence=0.7)

        mock_predictor_class.assert_called_once_with(
            model_size="small",
            weights_path=None,
            class_names=None,
        )
        mock_predictor.predict.assert_called_once_with("test.jpg", confidence_threshold=0.7)
