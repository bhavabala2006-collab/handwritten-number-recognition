"""
preprocess_data.py
-------------------
Turns the RAW custom samples collected by collect_data.py (stored as
data/raw/<number 1-10>/*.png, full ROI snapshots) into individual,
labeled DIGIT crops (0-9) ready to train the CNN, using the exact same
segmentation pipeline that will be used at inference time
(src.preprocessing.image_processing.segment_digits).

Why digit-level, not number-level?
    The CNN classifies a single digit (0-9). The product-facing numbers
    1-10 are produced by combining one or two digit predictions (see
    src/inference/number_composer.py). So a raw sample of the number
    "10" is expected to contain exactly two digit blobs, which are
    automatically labeled "1" and "0" in left-to-right order. A raw
    sample of number "7" is expected to contain exactly one blob,
    labeled "7". Samples that don't segment into the expected number of
    blobs are skipped with a warning (usually means the handwriting
    touched/overlapped and should be re-captured).

Output
------
    data/processed/custom_train.npz  (X, y)
    data/processed/custom_val.npz    (X, y)
    data/processed/custom_test.npz   (X, y)

    Plus human-inspectable PNG copies of every extracted digit crop
    under data/train/<digit>/, data/validation/<digit>/, data/test/<digit>/
    so you can visually sanity-check what the model will actually train on.

Usage
-----
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --test-size 0.15 --val-size 0.15
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.preprocessing.image_processing import segment_digits
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected digit sequence for each product-facing number label.
EXPECTED_DIGITS = {
    **{str(n): [n] for n in range(1, 10)},
    "10": [1, 0],
}


def load_raw_samples():
    """Walk data/raw/<number>/*.png and segment each into digit crops."""
    images, labels = [], []
    skipped = 0

    if not CONFIG.RAW_DATA_DIR.exists():
        logger.warning("No raw data directory found at %s", CONFIG.RAW_DATA_DIR)
        return images, labels, skipped

    for number_dir in sorted(CONFIG.RAW_DATA_DIR.iterdir()):
        if not number_dir.is_dir() or number_dir.name not in EXPECTED_DIGITS:
            continue

        expected = EXPECTED_DIGITS[number_dir.name]
        sample_files = sorted(number_dir.glob("*.png"))
        logger.info("Processing %d raw samples for number '%s'", len(sample_files), number_dir.name)

        for sample_path in sample_files:
            frame = cv2.imread(str(sample_path))
            if frame is None:
                logger.warning("Could not read %s, skipping", sample_path)
                skipped += 1
                continue

            digit_crops = segment_digits(frame)

            if len(digit_crops) != len(expected):
                logger.warning(
                    "Sample %s: expected %d digit blob(s) for number '%s' but found %d - skipping.",
                    sample_path.name,
                    len(expected),
                    number_dir.name,
                    len(digit_crops),
                )
                skipped += 1
                continue

            for crop, expected_digit in zip(digit_crops, expected):
                images.append(crop.image)
                labels.append(expected_digit)

    return images, labels, skipped


def save_split(X, y, npz_path: Path, png_root: Path):
    """Save an (X, y) split both as a compact .npz and as inspectable PNGs."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if len(X) > 0:
        np.savez_compressed(npz_path, X=np.stack(X), y=np.array(y, dtype=np.int64))
    else:
        np.savez_compressed(npz_path, X=np.zeros((0, CONFIG.DIGIT_IMAGE_SIZE, CONFIG.DIGIT_IMAGE_SIZE, 1)), y=np.zeros((0,)))
    logger.info("Saved %d samples to %s", len(X), npz_path)

    for i, (img, label) in enumerate(zip(X, y)):
        digit_dir = png_root / str(int(label))
        digit_dir.mkdir(parents=True, exist_ok=True)
        # img is float32 in [0, 1] with shape (SIZE, SIZE, 1)
        uint8_img = (img.squeeze(-1) * 255).astype(np.uint8)
        cv2.imwrite(str(digit_dir / f"{i}.png"), uint8_img)


def main():
    parser = argparse.ArgumentParser(description="Preprocess custom raw samples into digit crops.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    images, labels, skipped = load_raw_samples()
    total = len(images)
    print(f"\nExtracted {total} digit crops from raw samples ({skipped} raw samples skipped).")

    if total == 0:
        print(
            "No custom samples found. This is OK - train.py will fall back to "
            "MNIST only. Run scripts/collect_data.py first if you want a "
            "webcam-specific custom dataset."
        )
        return

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(args.test_size + args.val_size), random_state=CONFIG.RANDOM_SEED, stratify=y
    )
    relative_test = args.test_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=CONFIG.RANDOM_SEED, stratify=y_temp
    )

    save_split(X_train, y_train, CONFIG.PROCESSED_DATA_DIR / "custom_train.npz", CONFIG.TRAIN_DIR)
    save_split(X_val, y_val, CONFIG.PROCESSED_DATA_DIR / "custom_val.npz", CONFIG.VALIDATION_DIR)
    save_split(X_test, y_test, CONFIG.PROCESSED_DATA_DIR / "custom_test.npz", CONFIG.TEST_DIR)

    print(
        f"Split -> train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}\n"
        "Run 'python scripts/train.py' next."
    )


if __name__ == "__main__":
    main()
