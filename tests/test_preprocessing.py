"""
test_preprocessing.py
----------------------
Unit tests for src/preprocessing/image_processing.py.

These tests use small, synthetically drawn images (via OpenCV drawing
functions) rather than real webcam captures, so they run instantly and
deterministically in CI / on any machine, without needing a camera.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.preprocessing.image_processing import (
    to_grayscale,
    reduce_noise,
    binarize,
    find_digit_bounding_boxes,
    resize_and_normalize,
    segment_digits,
    has_writing,
)


def make_blank_roi(width=300, height=200):
    """A plain white 'page' with no writing on it."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_digit_like_blob(image, center, size=40, thickness=6):
    """Draw a simple thick circle to simulate a single handwritten stroke."""
    cv2.circle(image, center, size // 2, (0, 0, 0), thickness)
    return image


def test_to_grayscale_shape():
    color = make_blank_roi()
    gray = to_grayscale(color)
    assert gray.ndim == 2
    assert gray.shape == color.shape[:2]


def test_to_grayscale_passthrough_for_already_gray():
    gray_in = np.zeros((50, 50), dtype=np.uint8)
    gray_out = to_grayscale(gray_in)
    assert gray_out.shape == gray_in.shape


def test_reduce_noise_preserves_shape():
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    blurred = reduce_noise(gray)
    assert blurred.shape == gray.shape


def test_binarize_output_is_binary():
    gray = np.full((100, 100), 200, dtype=np.uint8)
    cv2.circle(gray, (50, 50), 20, 0, -1)
    blurred = reduce_noise(gray)
    binary = binarize(blurred)
    unique_values = set(np.unique(binary).tolist())
    assert unique_values.issubset({0, 255})


def test_empty_roi_has_no_writing():
    blank = make_blank_roi()
    assert has_writing(blank) is False


def test_roi_with_stroke_has_writing():
    image = make_blank_roi()
    draw_digit_like_blob(image, center=(150, 100), size=60, thickness=8)
    assert has_writing(image) is True


def test_find_digit_bounding_boxes_finds_one_blob():
    image = make_blank_roi()
    draw_digit_like_blob(image, center=(150, 100), size=60, thickness=8)
    gray = to_grayscale(image)
    blurred = reduce_noise(gray)
    binary = binarize(blurred)
    boxes = find_digit_bounding_boxes(binary)
    assert len(boxes) == 1


def test_find_digit_bounding_boxes_finds_two_blobs_left_to_right():
    image = make_blank_roi(width=400)
    draw_digit_like_blob(image, center=(100, 100), size=50, thickness=8)
    draw_digit_like_blob(image, center=(300, 100), size=50, thickness=8)
    gray = to_grayscale(image)
    blurred = reduce_noise(gray)
    binary = binarize(blurred)
    boxes = find_digit_bounding_boxes(binary)
    assert len(boxes) == 2
    # left-to-right ordering: first box's x should be smaller than second's
    assert boxes[0][0] < boxes[1][0]


def test_resize_and_normalize_output_shape_and_range():
    crop = np.full((40, 20), 255, dtype=np.uint8)
    result = resize_and_normalize(crop, size=CONFIG.DIGIT_IMAGE_SIZE)
    assert result.shape == (CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_resize_and_normalize_handles_empty_crop():
    empty_crop = np.zeros((0, 0), dtype=np.uint8)
    result = resize_and_normalize(empty_crop, size=CONFIG.DIGIT_IMAGE_SIZE)
    assert result.shape == (CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1)


def test_segment_digits_on_blank_roi_returns_nothing():
    blank = make_blank_roi()
    digits = segment_digits(blank)
    assert digits == []


def test_segment_digits_on_single_stroke_returns_one_crop():
    image = make_blank_roi()
    draw_digit_like_blob(image, center=(150, 100), size=60, thickness=8)
    digits = segment_digits(image)
    assert len(digits) == 1
    assert digits[0].image.shape == (CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1)
