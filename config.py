"""
config.py
---------
Central configuration for the Real-Time Handwritten Number Recognition project.

Every value that a beginner might want to tune lives here instead of being
scattered (and hardcoded) throughout the codebase. Import this module
wherever a setting is needed:

    from config import CONFIG

All paths are built with `pathlib.Path` so the project works the same way
on Windows, macOS, and Linux.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (the folder that contains this config.py file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


class Config:
    """Simple namespace holding every configurable value for the project."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    TRAIN_DIR = DATA_DIR / "train"
    VALIDATION_DIR = DATA_DIR / "validation"
    TEST_DIR = DATA_DIR / "test"

    MODELS_DIR = PROJECT_ROOT / "models"
    TRAINED_MODEL_DIR = MODELS_DIR / "trained_model"
    MODEL_FILENAME = "digit_cnn.keras"
    MODEL_PATH = TRAINED_MODEL_DIR / MODEL_FILENAME
    CLASS_INDICES_PATH = TRAINED_MODEL_DIR / "class_indices.json"
    TRAINING_HISTORY_PATH = TRAINED_MODEL_DIR / "training_history.json"

    LOG_DIR = PROJECT_ROOT / "logs"

    # ------------------------------------------------------------------
    # Image / model parameters
    # ------------------------------------------------------------------
    # Every individual digit is resized to this square size before being
    # fed to the CNN. 28x28 matches the MNIST convention.
    DIGIT_IMAGE_SIZE = 28

    # Number of channels (1 = grayscale)
    IMAGE_CHANNELS = 1

    # Digit classifier always predicts a single digit 0-9.
    NUM_DIGIT_CLASSES = 10

    # The final, user-facing classes the product recognizes: 1 .. 10
    NUMBER_CLASSES = [str(n) for n in range(1, 11)]  # ["1", ..., "10"]

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1  # fraction of MNIST training data held out
    RANDOM_SEED = 42

    # Data augmentation ranges (kept mild - handwritten digits are
    # sensitive to over-distortion)
    AUGMENT_ROTATION_RANGE = 10       # degrees
    AUGMENT_WIDTH_SHIFT_RANGE = 0.08  # fraction of width
    AUGMENT_HEIGHT_SHIFT_RANGE = 0.08  # fraction of height
    AUGMENT_ZOOM_RANGE = 0.08

    # ------------------------------------------------------------------
    # Webcam / inference parameters
    # ------------------------------------------------------------------
    CAMERA_INDEX = 0

    # Region Of Interest (ROI) as a fraction of the video frame,
    # (x_min, y_min, x_max, y_max) — used by collect_data.py.
    # The browser app instead lets the user drag/draw the ROI box; this
    # value is the *default* box shown to the user.
    ROI_FRACTION = (0.30, 0.25, 0.70, 0.75)

    # Minimum confidence (0-1) before a prediction is shown as "confident"
    CONFIDENCE_THRESHOLD = 0.60

    # How many recent predictions to keep for smoothing (majority vote)
    SMOOTHING_WINDOW = 7

    # A prediction must appear at least this many times in the smoothing
    # window to be accepted as the "stable" displayed result.
    SMOOTHING_MIN_AGREEMENT = 4

    # Minimum contour area (in pixels, on a thresholded ROI) to be
    # considered a real digit stroke rather than noise.
    MIN_CONTOUR_AREA = 60

    # Padding (in pixels) added around a detected digit bounding box
    # before it is cropped, to avoid clipping strokes.
    DIGIT_CROP_PADDING = 12

    # Maximum number of digit blobs we ever try to read from the ROI.
    # The product only needs 1 or 2 (for "10").
    MAX_DIGITS = 2

    # ------------------------------------------------------------------
    # Flask app
    # ------------------------------------------------------------------
    FLASK_HOST = "127.0.0.1"
    FLASK_PORT = 5000
    FLASK_DEBUG = True


CONFIG = Config()
