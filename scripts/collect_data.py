"""
collect_data.py
----------------
Interactive webcam tool for collecting CUSTOM handwritten-number samples,
used to make the model more robust to real webcam lighting, pens, and
handwriting styles (MNIST alone was collected very differently: scanned,
centered, uniform digits).

The captured images are RAW ROI snapshots (not yet split into individual
digit crops) and are saved under:

    data/raw/<number>/<timestamp>.png

where <number> is one of 1..10. Numbers 1-9 should be written as a single
digit; number 10 should be written as the two digits "1" and "0" next to
each other, exactly as the app will expect to see it live.

Run `python scripts/preprocess_data.py` afterwards to turn these raw
captures into labeled digit crops used by scripts/train.py.

Usage
-----
    python scripts/collect_data.py --class 7
    python scripts/collect_data.py --class 10 --samples 80
    python scripts/collect_data.py --interactive

Controls while the webcam window is focused:
    c       capture a single sample right now
    a       toggle auto-capture (captures automatically whenever
            handwriting is detected inside the ROI, ~2 samples/second)
    n       (interactive mode only) move to the next class
    q / ESC quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.preprocessing.image_processing import has_writing
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_CLASSES = [str(n) for n in range(1, 11)]


def get_roi_box(frame_shape):
    """Convert the fractional ROI in config.py into pixel coordinates."""
    h, w = frame_shape[:2]
    fx0, fy0, fx1, fy1 = CONFIG.ROI_FRACTION
    return int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)


def save_sample(frame_roi, class_label: str) -> Path:
    out_dir = CONFIG.RAW_DATA_DIR / class_label
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000)}.png"
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), frame_roi)
    return out_path


def run_collection(class_label: str, target_samples: int, camera_index: int, interactive: bool):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open webcam at index %s", camera_index)
        print(
            "ERROR: could not open the webcam. Check that no other "
            "application is using it and that CAMERA_INDEX in config.py "
            "is correct."
        )
        return

    auto_capture = False
    saved_count = 0
    last_auto_capture_time = 0.0
    class_list = [class_label] if not interactive else list(VALID_CLASSES)
    class_idx = 0
    current_class = class_list[class_idx]

    window_name = "Data Collection - press q to quit"
    logger.info("Starting data collection for class '%s'", current_class)
    print(f"\nCollecting samples for number: {current_class}")
    print("Controls: [c] capture  [a] toggle auto-capture  [n] next class  [q] quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.error("Failed to read frame from webcam")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural writing experience
        x0, y0, x1, y1 = get_roi_box(frame.shape)
        roi = frame[y0:y1, x0:x1]

        writing_detected = has_writing(roi)

        display = frame.copy()
        box_color = (0, 200, 0) if writing_detected else (0, 0, 200)
        cv2.rectangle(display, (x0, y0), (x1, y1), box_color, 2)
        cv2.putText(
            display,
            f"Class: {current_class}  Saved: {saved_count}/{target_samples}  Auto: {auto_capture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, display)

        if auto_capture and writing_detected and (time.time() - last_auto_capture_time) > 0.5:
            save_sample(roi, current_class)
            saved_count += 1
            last_auto_capture_time = time.time()
            logger.info("Auto-captured sample %d for class %s", saved_count, current_class)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            save_sample(roi, current_class)
            saved_count += 1
            logger.info("Captured sample %d for class %s", saved_count, current_class)
        elif key == ord("a"):
            auto_capture = not auto_capture
        elif key == ord("n") and interactive:
            class_idx = (class_idx + 1) % len(class_list)
            current_class = class_list[class_idx]
            saved_count = 0
            print(f"\nSwitched to number: {current_class}")
        elif key in (ord("q"), 27):  # 27 = ESC
            break

        if not interactive and saved_count >= target_samples:
            print(f"Collected {saved_count} samples for class {current_class}. Done.")
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect custom handwritten number samples.")
    parser.add_argument(
        "--class",
        dest="class_label",
        type=str,
        choices=VALID_CLASSES,
        help="Which number (1-10) to collect samples for.",
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="Target number of samples to collect."
    )
    parser.add_argument(
        "--camera-index", type=int, default=CONFIG.CAMERA_INDEX, help="Webcam device index."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Cycle through all classes 1-10 in one session with the 'n' key.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.interactive and not args.class_label:
        print("ERROR: pass --class <1-10> or use --interactive")
        sys.exit(1)
    run_collection(
        class_label=args.class_label or "1",
        target_samples=args.samples,
        camera_index=args.camera_index,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
