"""
image_processing.py
--------------------
The single source of truth for turning a raw camera frame (or a raw
collected sample image) into normalized digit arrays the CNN can consume.

CRITICAL DESIGN RULE (see project README, "Image Preprocessing" section):
    The exact same functions in this file are used by:
      1. scripts/collect_data.py   (when saving custom webcam samples)
      2. scripts/preprocess_data.py (when preparing the custom dataset)
      3. src/inference/predictor.py (at live prediction time)

    This guarantees that a model never sees training images that were
    prepared differently from the images it receives during real-time
    inference - a very common source of "the model works in testing but
    fails on the webcam" bugs.

Pipeline stages implemented here:
    frame (BGR)
      -> grayscale
      -> Gaussian blur (noise reduction)
      -> adaptive threshold (binarization, background/foreground handling)
      -> contour detection (find individual digit blobs)
      -> bounding box + padding + crop
      -> resize onto a square canvas (aspect-ratio preserving) + normalize
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config import CONFIG


@dataclass
class DigitCrop:
    """A single detected digit, ready for the CNN, plus its geometry."""

    image: np.ndarray  # shape (SIZE, SIZE, 1), float32, values in [0, 1]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in the ORIGINAL roi


def to_grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR (OpenCV default) frame to single-channel grayscale."""
    if frame_bgr.ndim == 2:
        return frame_bgr  # already grayscale
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a Gaussian blur to reduce sensor / lighting noise."""
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def binarize(gray_blurred: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image into a clean black-background /
    white-foreground binary image using adaptive thresholding.

    Adaptive thresholding is used (rather than a single global threshold)
    because webcam lighting is rarely uniform across the whole ROI.

    The output convention is: digit strokes = 255 (white), background = 0.
    This matches the MNIST convention, which is important for consistency
    with the model trained on MNIST.
    """
    # THRESH_BINARY_INV: dark pen strokes on a bright page become white
    # foreground on a black background after inversion.
    binary = cv2.adaptiveThreshold(
        gray_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10,
    )

    # Remove tiny speckle noise with a light morphological opening, and
    # close small gaps within a stroke with a dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def find_digit_bounding_boxes(
    binary: np.ndarray,
    min_contour_area: int = None,
    max_digits: int = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Find candidate digit blobs in a binarized image and return their
    bounding boxes sorted left-to-right (reading order), which is what
    lets us reconstruct multi-digit numbers like "10" as "1" then "0".
    """
    min_contour_area = min_contour_area or CONFIG.MIN_CONTOUR_AREA
    max_digits = max_digits or CONFIG.MAX_DIGITS

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    # Sort left -> right (reading order for multi-digit numbers)
    boxes.sort(key=lambda b: b[0])

    # Merge boxes that likely belong to the same digit (e.g. the dot of an
    # "i"-like stroke, or a broken "0"): if two boxes heavily overlap in
    # the x-axis, merge them into one bounding box.
    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        x, y, w, h = box
        prev_right = px + pw
        # If the new box starts well before the previous box ends,
        # treat them as fragments of the same digit and merge.
        if x < prev_right - min(pw, w) * 0.3:
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    # Keep only the largest `max_digits` boxes by area if we found more
    # blobs than expected (extra noise / stray marks).
    if len(merged) > max_digits:
        merged.sort(key=lambda b: b[2] * b[3], reverse=True)
        merged = merged[:max_digits]
        merged.sort(key=lambda b: b[0])

    return merged


def crop_with_padding(
    binary: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = None,
) -> np.ndarray:
    """Crop a bounding box out of the binary image with extra padding."""
    padding = padding if padding is not None else CONFIG.DIGIT_CROP_PADDING
    x, y, w, h = bbox
    h_img, w_img = binary.shape[:2]

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_img, x + w + padding)
    y1 = min(h_img, y + h + padding)

    return binary[y0:y1, x0:x1]


def resize_and_normalize(
    crop: np.ndarray, size: int = None
) -> np.ndarray:
    """
    Resize a cropped digit onto a square `size x size` canvas while
    preserving aspect ratio (matching the classic MNIST preparation
    convention: digit scaled to fit ~20x20 then centered in a 28x28
    canvas), then normalize pixel values to [0, 1].

    Returns an array of shape (size, size, 1), dtype float32.
    """
    size = size or CONFIG.DIGIT_IMAGE_SIZE
    if crop.size == 0:
        return np.zeros((size, size, 1), dtype=np.float32)

    h, w = crop.shape[:2]
    # Target the digit to occupy ~80% of the canvas, like MNIST's 20/28.
    inner = int(size * 20 / 28)
    scale = inner / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    normalized = canvas.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]  # add channel dimension


def segment_digits(roi_bgr: np.ndarray) -> List[DigitCrop]:
    """
    Full pipeline: raw ROI frame -> list of ready-for-CNN digit crops,
    ordered left-to-right.

    This is the ONE function both training-data preparation and live
    inference should call, so preprocessing never drifts between the two.
    """
    gray = to_grayscale(roi_bgr)
    blurred = reduce_noise(gray)
    binary = binarize(blurred)
    boxes = find_digit_bounding_boxes(binary)

    digits: List[DigitCrop] = []
    for box in boxes:
        crop = crop_with_padding(binary, box)
        normalized = resize_and_normalize(crop)
        digits.append(DigitCrop(image=normalized, bbox=box))

    return digits


def has_writing(roi_bgr: np.ndarray, min_ink_pixels: int = 150) -> bool:
    """
    Cheap check used to decide whether the ROI currently contains any
    handwriting at all, so the app can show an "empty" status instead of
    guessing wildly on a blank ROI.
    """
    gray = to_grayscale(roi_bgr)
    blurred = reduce_noise(gray)
    binary = binarize(blurred)
    return int(np.count_nonzero(binary)) >= min_ink_pixels
