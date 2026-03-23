"""
Inference module.
Wraps the ONNX runtime for thread-safe prediction.
"""

import logging
from typing import Optional

import numpy as np
import onnxruntime as ort

from .config import SETTINGS

logger = logging.getLogger(__name__)


class AKIPredictor:
    """
    Loads model artifacts and serves predictions.
    """

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.threshold: float = 0.5  # defaults to unoptimized

        self._load_resources()

    def _load_resources(self) -> None:
        """Loads ONNX model and probability threshold from disk."""
        if not SETTINGS.MODEL_PATH.exists():
            logger.critical(f"Model missing at {SETTINGS.MODEL_PATH}")
            return

        try:
            if SETTINGS.THRESHOLD_PATH.exists():
                with open(SETTINGS.THRESHOLD_PATH, "r") as f:
                    self.threshold = float(f.read().strip())

            self.session = ort.InferenceSession(str(SETTINGS.MODEL_PATH))
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Model loaded. Threshold: {self.threshold}")

        except Exception as e:
            logger.exception(f"Failed to initialize model: {e}")
            self.session = None

    def predict(self, features: np.ndarray) -> bool:
        """
        Runs inference. Returns True if probability >= threshold.
        """
        if self.session is None:
            return False

        try:
            # ONNX requires float32 specifically
            inputs = {self.input_name: features.astype(np.float32)}
            outputs = self.session.run(None, inputs)
            # output format: [ label, [ { class: pr }, ... ] ]
            # we access probabilities (index 1), first item (batch 0), label 1 (Positive)
            prob_pos = outputs[1][0][1]
            return bool(prob_pos >= self.threshold)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return False
