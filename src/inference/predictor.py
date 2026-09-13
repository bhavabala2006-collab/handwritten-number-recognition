"""
predictor.py
------------
Loads the trained CNN once and exposes a single high-level function,
`predict_from_roi`, that takes a raw ROI frame (BGR, straight from
OpenCV/webcam) and returns a NumberResult (see number_composer.py).

This module is used by both:
    - app/app.py            (live webcam Flask endpoint)
    - scripts/test_model.py (manual command-line testing)
"""

from typing import Optional

import numpy as np

from config import CONFIG
from src.preprocessing.image_processing import segment_digits, has_writing
from src.inference.number_composer import (
    DigitPrediction,
    NumberResult,
    compose_number,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DigitPredictor:
    """Wraps the trained Keras model and the full inference pipeline."""

    def __init__(self, model_path=None):
        self.model_path = model_path or CONFIG.MODEL_PATH
        self._model = None  # lazy-loaded so importing this module is cheap

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Run 'python scripts/train.py' first to create it."
            )
        # Imported here (not at module top) so that simply importing this
        # file for unit tests that mock the model doesn't require
        # TensorFlow to be importable / fast to start up.
        from tensorflow import keras

        logger.info("Loading trained model from %s", self.model_path)
        self._model = keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully")

    def predict_digit_batch(self, digit_images: np.ndarray) -> list:
        """
        Run the CNN on a batch of digit images.

        Parameters
        ----------
        digit_images : np.ndarray of shape (N, SIZE, SIZE, 1)

        Returns
        -------
        list[DigitPrediction]
        """
        self._ensure_model_loaded()
        if len(digit_images) == 0:
            return []

        probabilities = self._model.predict(digit_images, verbose=0)
        results = []
        for probs in probabilities:
            digit = int(np.argmax(probs))
            confidence = float(np.max(probs))
            results.append(DigitPrediction(digit=digit, confidence=confidence))
        return results

    def predict_from_roi(self, roi_bgr: np.ndarray) -> NumberResult:
        """
        Full pipeline: raw ROI (BGR) -> NumberResult.

        This function performs exactly the same preprocessing steps that
        were used to prepare training data (see
        src/preprocessing/image_processing.py), which is essential for
        the model to behave consistently between training and live use.
        """
        if not has_writing(roi_bgr):
            return NumberResult(number=None, confidence=0.0, status="empty", raw_digits=[])

        digit_crops = segment_digits(roi_bgr)
        if len(digit_crops) == 0:
            return NumberResult(number=None, confidence=0.0, status="empty", raw_digits=[])

        batch = np.stack([d.image for d in digit_crops], axis=0)
        predictions = self.predict_digit_batch(batch)
        return compose_number(predictions)


# A module-level singleton so the Flask app (and scripts) can share one
# loaded model instead of reloading it on every request.
_default_predictor: Optional[DigitPredictor] = None


def get_predictor() -> DigitPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = DigitPredictor()
    return _default_predictor
