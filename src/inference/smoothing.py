"""
smoothing.py
------------
Implements temporal prediction smoothing so the UI does not flicker
rapidly between numbers while the user is mid-stroke (e.g. showing
7, 3, 7, 8, 7 within a second while the user is actually writing "7").

Approach: keep a sliding window of the last N raw predictions. The
"stable" result shown to the user only changes once the SAME number has
appeared at least `min_agreement` times within that window. This is a
simple, easy-to-explain majority-vote smoother - see README section
"Prediction Smoothing" for the rationale.
"""

from collections import deque
from typing import Optional

from config import CONFIG
from src.inference.number_composer import NumberResult


class PredictionSmoother:
    def __init__(self, window_size: int = None, min_agreement: int = None):
        self.window_size = window_size or CONFIG.SMOOTHING_WINDOW
        self.min_agreement = min_agreement or CONFIG.SMOOTHING_MIN_AGREEMENT
        self._history = deque(maxlen=self.window_size)
        self._stable_number: Optional[int] = None
        self._stable_confidence: float = 0.0

    def reset(self):
        """Clear all history - called when the user presses "Clear"."""
        self._history.clear()
        self._stable_number = None
        self._stable_confidence = 0.0

    def update(self, result: NumberResult):
        """
        Feed in the newest raw NumberResult and recompute the stable,
        smoothed result to display.

        Returns
        -------
        dict with keys: number, confidence, status, is_stable
        """
        self._history.append(result)

        # Count occurrences of each candidate number within the window,
        # only among predictions that were confident enough to count.
        counts = {}
        confidence_sums = {}
        for r in self._history:
            if r.number is None:
                continue
            counts[r.number] = counts.get(r.number, 0) + 1
            confidence_sums[r.number] = confidence_sums.get(r.number, 0.0) + r.confidence

        best_number = None
        best_count = 0
        for number, count in counts.items():
            if count > best_count:
                best_number = number
                best_count = count

        is_stable = best_number is not None and best_count >= self.min_agreement

        if is_stable:
            self._stable_number = best_number
            self._stable_confidence = confidence_sums[best_number] / best_count
        # If not yet stable, keep showing the previous stable result
        # (rather than snapping to "no result") so the UI feels smooth.

        latest = result
        return {
            "number": self._stable_number,
            "confidence": round(self._stable_confidence, 4),
            "status": latest.status,
            "is_stable": is_stable,
            "raw_number": latest.number,
            "raw_confidence": round(latest.confidence, 4),
        }
