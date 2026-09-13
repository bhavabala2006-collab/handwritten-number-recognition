"""
test_model.py
--------------
Tests for:
  - src/model/cnn_model.py  (architecture sanity: shapes, class count)
  - src/inference/number_composer.py (the "how is 10 handled" logic)

Building the CNN is fast (no training happens here), so these tests stay
quick even though they import TensorFlow.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.model.cnn_model import build_digit_cnn
from src.inference.number_composer import DigitPrediction, compose_number


# ---------------------------------------------------------------------
# CNN architecture tests
# ---------------------------------------------------------------------

def test_model_output_shape_matches_num_classes():
    model = build_digit_cnn()
    dummy_batch = np.zeros(
        (2, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.IMAGE_CHANNELS),
        dtype=np.float32,
    )
    predictions = model.predict(dummy_batch, verbose=0)
    assert predictions.shape == (2, CONFIG.NUM_DIGIT_CLASSES)


def test_model_output_is_a_valid_probability_distribution():
    model = build_digit_cnn()
    dummy_batch = np.random.rand(
        3, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.IMAGE_CHANNELS
    ).astype(np.float32)
    predictions = model.predict(dummy_batch, verbose=0)
    row_sums = predictions.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-4)


def test_model_input_shape():
    model = build_digit_cnn()
    expected_shape = (
        None,
        CONFIG.DIGIT_IMAGE_SIZE,
        CONFIG.DIGIT_IMAGE_SIZE,
        CONFIG.IMAGE_CHANNELS,
    )
    assert tuple(model.input_shape) == expected_shape


# ---------------------------------------------------------------------
# Number composition ("10" handling) tests
# ---------------------------------------------------------------------

def test_single_digit_maps_directly_to_number():
    result = compose_number([DigitPrediction(digit=7, confidence=0.95)])
    assert result.number == 7
    assert result.status == "ok"


def test_single_zero_digit_is_unsupported():
    result = compose_number([DigitPrediction(digit=0, confidence=0.95)])
    assert result.number is None
    assert result.status == "unsupported"


def test_one_then_zero_composes_to_ten():
    result = compose_number(
        [DigitPrediction(digit=1, confidence=0.9), DigitPrediction(digit=0, confidence=0.88)]
    )
    assert result.number == 10
    assert result.status == "ok"


def test_two_digits_not_one_zero_is_unsupported():
    result = compose_number(
        [DigitPrediction(digit=3, confidence=0.9), DigitPrediction(digit=5, confidence=0.9)]
    )
    assert result.number is None
    assert result.status == "unsupported"


def test_no_digits_is_empty():
    result = compose_number([])
    assert result.number is None
    assert result.status == "empty"


def test_low_confidence_single_digit_is_uncertain():
    low_conf = CONFIG.CONFIDENCE_THRESHOLD - 0.1
    result = compose_number([DigitPrediction(digit=4, confidence=max(low_conf, 0.01))])
    assert result.number == 4
    assert result.status == "uncertain"


def test_more_than_two_digits_is_unsupported():
    predictions = [DigitPrediction(digit=d, confidence=0.9) for d in [1, 2, 3]]
    result = compose_number(predictions)
    assert result.number is None
    assert result.status == "unsupported"
