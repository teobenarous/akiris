import unittest
from unittest.mock import MagicMock, mock_open, patch
import numpy as np
from app.predictor import AKIPredictor


class TestPredictor(unittest.TestCase):
    # Patch the class method directly
    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="0.65")
    @patch("app.predictor.ort.InferenceSession")
    def test_load_success(self, mock_ort, *_):
        """Verify model and threshold load correctly."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input_tensor")]
        mock_ort.return_value = mock_session

        predictor = AKIPredictor()

        self.assertIsNotNone(predictor.session)
        self.assertEqual(predictor.threshold, 0.65)
        mock_ort.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    def test_missing_model(self, _):
        """Verify graceful degradation if the ONNX model is missing."""
        with self.assertLogs("app.predictor", level="CRITICAL"):
            predictor = AKIPredictor()
            self.assertIsNone(predictor.session)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="0.5")
    @patch("app.predictor.ort.InferenceSession")
    def test_predict_logic(self, mock_ort, *_):
        """Verify probability extraction and threshold logic."""
        mock_session = MagicMock()
        mock_ort.return_value = mock_session

        predictor = AKIPredictor()
        predictor.threshold = 0.5
        predictor.input_name = "float_input"

        dummy_features = np.zeros((1, 13))

        mock_session.run.return_value = [np.array([0]), [{0: 0.7, 1: 0.3}]]
        self.assertFalse(predictor.predict(dummy_features))

        mock_session.run.return_value = [np.array([1]), [{0: 0.2, 1: 0.8}]]
        self.assertTrue(predictor.predict(dummy_features))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="0.5")
    @patch("app.predictor.ort.InferenceSession")
    def test_predict_exception(self, mock_ort, *_):
        """Verify exceptions during inference return False instead of crashing."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Tensor mismatch")
        mock_ort.return_value = mock_session

        predictor = AKIPredictor()
        with self.assertLogs("app.predictor", level="ERROR"):
            result = predictor.predict(np.zeros((1, 13)))
            self.assertFalse(result)