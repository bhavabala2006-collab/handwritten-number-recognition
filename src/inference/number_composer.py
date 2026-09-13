"""
number_composer.py
-------------------
Implements the "Number 10 Handling" logic described in the README.

A normal MNIST-style classifier only ever outputs a single digit (0-9).
Our product needs to recognize the numbers 1 through 10, and "10" is
written as TWO digit strokes: "1" followed by "0".

The overall system therefore uses a two-stage approach:

    Stage 1 (segmentation): src.preprocessing.image_processing.segment_digits
        looks at the ROI, finds connected ink blobs, and returns them
        left-to-right as separate digit crops. This answers "does the
        ROI contain one digit or two?"

    Stage 2 (classification): the CNN in src.model.cnn_model classifies
        EACH individual digit blob independently as 0-9.

    This module (`number_composer`) then combines Stage 2's outputs into
    the final answer:
        - exactly one blob classified as digit d (1-9)  -> number = d
        - exactly one blob classified as digit 0        -> invalid
          (0 alone is not one of our target numbers 1-10)
        - exactly two blobs classified as "1" then "0"   -> number = 10
        - two blobs with any other digit combination     -> invalid
          (not a number in our supported range)
        - zero blobs, or more than two blobs             -> invalid
"""

from dataclasses import dataclass
from typing import List, Optional

from config import CONFIG


@dataclass
class DigitPrediction:
    digit: int          # 0-9, the CNN's predicted class for this blob
    confidence: float   # 0-1, softmax probability of that class


@dataclass
class NumberResult:
    number: Optional[int]     # 1-10, or None if no valid number was formed
    confidence: float         # 0-1, combined confidence
    status: str                # "ok" | "empty" | "uncertain" | "unsupported"
    raw_digits: List[int]      # the individual digit predictions, for debugging


def compose_number(predictions: List[DigitPrediction]) -> NumberResult:
    """Combine per-digit CNN predictions into a final 1-10 result."""
    if len(predictions) == 0:
        return NumberResult(number=None, confidence=0.0, status="empty", raw_digits=[])

    if len(predictions) == 1:
        digit_pred = predictions[0]
        if digit_pred.digit == 0:
            # "0" alone is outside the supported 1-10 range.
            return NumberResult(
                number=None,
                confidence=digit_pred.confidence,
                status="unsupported",
                raw_digits=[digit_pred.digit],
            )
        return NumberResult(
            number=digit_pred.digit,
            confidence=digit_pred.confidence,
            status="ok" if digit_pred.confidence >= CONFIG.CONFIDENCE_THRESHOLD else "uncertain",
            raw_digits=[digit_pred.digit],
        )

    if len(predictions) == 2:
        first, second = predictions
        combined_confidence = min(first.confidence, second.confidence)
        if first.digit == 1 and second.digit == 0:
            return NumberResult(
                number=10,
                confidence=combined_confidence,
                status="ok" if combined_confidence >= CONFIG.CONFIDENCE_THRESHOLD else "uncertain",
                raw_digits=[first.digit, second.digit],
            )
        return NumberResult(
            number=None,
            confidence=combined_confidence,
            status="unsupported",
            raw_digits=[first.digit, second.digit],
        )

    # More than 2 blobs detected - not something our product supports.
    return NumberResult(
        number=None,
        confidence=0.0,
        status="unsupported",
        raw_digits=[p.digit for p in predictions],
    )
