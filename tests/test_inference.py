"""
test_inference.py
------------------
Tests for:
  - src/inference/predictor.py  (using a fake/mocked model, so no trained
    .keras file is required to run the test suite)
  - src/inference/smoothing.py  (temporal smoothing / anti-flicker logic)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.inference.predictor import DigitPredictor
from src.inference.number_composer import NumberResult
from src.inference.smoothing import PredictionSmoother


class FakeKerasModel:
    """
    A stand-in for a real tf.keras model. Always predicts the digit
    passed to its constructor with high confidence, regardless of input,
    so tests are fast, deterministic, and require no trained weights.
    """

    def __init__(self, fixed_digit: int = 7, confidence: float = 0.95):
        self.fixed_digit = fixed_digit
        self.confidence = confidence

    def predict(self, batch, verbose=0):
        n = len(batch)
        probs = np.zeros((n, CONFIG.NUM_DIGIT_CLASSES), dtype=np.float32)
        remaining = (1.0 - self.confidence) / (CONFIG.NUM_DIGIT_CLASSES - 1)
        probs[:] = remaining
        probs[:, self.fixed_digit] = self.confidence
        return probs


def make_predictor_with_fake_model(fixed_digit=7, confidence=0.95) -> DigitPredictor:
    predictor = DigitPredictor()
    predictor._model = FakeKerasModel(fixed_digit=fixed_digit, confidence=confidence)
    return predictor


def make_roi_with_one_stroke():
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.circle(image, (150, 100), 30, (0, 0, 0), 8)
    return image


def make_blank_roi():
    return np.full((200, 300, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------

def test_predict_digit_batch_returns_expected_digit_and_confidence():
    predictor = make_predictor_with_fake_model(fixed_digit=3, confidence=0.9)
    dummy_images = np.zeros((2, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1), dtype=np.float32)
    predictions = predictor.predict_digit_batch(dummy_images)
    assert len(predictions) == 2
    for p in predictions:
        assert p.digit == 3
        assert p.confidence == pytest.approx(0.9, abs=1e-3)


def test_predict_digit_batch_empty_input_returns_empty_list():
    predictor = make_predictor_with_fake_model()
    result = predictor.predict_digit_batch(np.zeros((0, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1)))
    assert result == []


def test_predict_from_roi_blank_image_is_empty_status():
    predictor = make_predictor_with_fake_model()
    result = predictor.predict_from_roi(make_blank_roi())
    assert isinstance(result, NumberResult)
    assert result.status == "empty"
    assert result.number is None


def test_predict_from_roi_single_stroke_uses_model_prediction():
    predictor = make_predictor_with_fake_model(fixed_digit=7, confidence=0.9)
    result = predictor.predict_from_roi(make_roi_with_one_stroke())
    assert result.number == 7
    assert result.status == "ok"


# ---------------------------------------------------------------------
# Smoothing tests
# ---------------------------------------------------------------------

def test_smoother_requires_min_agreement_before_becoming_stable():
    smoother = PredictionSmoother(window_size=5, min_agreement=3)

    # Two "7"s only - not yet enough agreement to become stable.
    r1 = smoother.update(NumberResult(number=7, confidence=0.9, status="ok", raw_digits=[7]))
    r2 = smoother.update(NumberResult(number=7, confidence=0.9, status="ok", raw_digits=[7]))
    assert r1["is_stable"] is False
    assert r2["is_stable"] is False
    assert r2["number"] is None

    # Third "7" reaches min_agreement=3 -> becomes stable.
    r3 = smoother.update(NumberResult(number=7, confidence=0.9, status="ok", raw_digits=[7]))
    assert r3["is_stable"] is True
    assert r3["number"] == 7


def test_smoother_ignores_brief_flicker():
    smoother = PredictionSmoother(window_size=5, min_agreement=3)
    sequence = [7, 7, 3, 7, 7]  # a brief flicker to "3" in the middle
    last = None
    for n in sequence:
        last = smoother.update(NumberResult(number=n, confidence=0.9, status="ok", raw_digits=[n]))
    assert last["number"] == 7
    assert last["is_stable"] is True


def test_smoother_reset_clears_history():
    smoother = PredictionSmoother(window_size=5, min_agreement=2)
    smoother.update(NumberResult(number=5, confidence=0.9, status="ok", raw_digits=[5]))
    smoother.update(NumberResult(number=5, confidence=0.9, status="ok", raw_digits=[5]))
    assert smoother._stable_number == 5

    smoother.reset()
    assert smoother._stable_number is None
    assert len(smoother._history) == 0
