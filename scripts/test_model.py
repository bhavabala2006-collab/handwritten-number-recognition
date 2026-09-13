"""
test_model.py
--------------
A small, manual command-line tool to sanity-check the full inference
pipeline (preprocessing + CNN + number composition) without starting the
whole Flask app. Useful right after training, or when debugging why a
particular image/frame is misclassified.

Usage
-----
    # Test a single saved image file (e.g. one of the collected ROI PNGs)
    python scripts/test_model.py --image data/raw/10/1699999999999.png

    # Grab a single live shot from the webcam and classify it
    python scripts/test_model.py --webcam
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.inference.predictor import get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_image(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: could not read image at {image_path}")
        return
    predictor = get_predictor()
    result = predictor.predict_from_roi(frame)
    print_result(result)


def test_webcam(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    print("Position your handwriting in front of the camera. Press SPACE to capture, q to quit.")
    predictor = get_predictor()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        fx0, fy0, fx1, fy1 = CONFIG.ROI_FRACTION
        x0, y0, x1, y1 = int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)
        display = frame.copy()
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.imshow("Press SPACE to capture, q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            roi = frame[y0:y1, x0:x1]
            result = predictor.predict_from_roi(roi)
            print_result(result)
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def print_result(result):
    print("\n--- Prediction ---")
    print(f"Number:      {result.number}")
    print(f"Confidence:  {result.confidence:.4f}")
    print(f"Status:      {result.status}")
    print(f"Raw digits:  {result.raw_digits}")
    print("------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Manually test the inference pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file to classify.")
    group.add_argument("--webcam", action="store_true", help="Capture a live shot from the webcam.")
    parser.add_argument("--camera-index", type=int, default=CONFIG.CAMERA_INDEX)
    args = parser.parse_args()

    if args.image:
        test_image(args.image)
    else:
        test_webcam(args.camera_index)


if __name__ == "__main__":
    main()
