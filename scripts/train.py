"""
train.py
--------
Trains the single-digit (0-9) CNN classifier and saves it to
models/trained_model/digit_cnn.keras.

Data sources
------------
1. MNIST (always used) - downloaded automatically by
   tensorflow.keras.datasets.mnist on first run and cached locally by
   Keras (no manual download needed).
2. Custom webcam data (used automatically IF PRESENT) - produced by
   running scripts/collect_data.py then scripts/preprocess_data.py, which
   creates data/processed/custom_train.npz / custom_val.npz / custom_test.npz.
   Mixing in custom data makes the model noticeably more robust to real
   webcam lighting, pen thickness, and handwriting style, since MNIST
   digits are centered, uniform, and pre-cleaned in a way a live webcam
   feed never quite is.

Usage
-----
    python scripts/train.py
    python scripts/train.py --epochs 25 --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.model.cnn_model import build_digit_cnn, build_data_augmentation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_mnist():
    """Load and normalize MNIST, matching our [0, 1], (28,28,1) convention."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    return x_train, y_train, x_test, y_test


def load_custom_split(npz_path: Path):
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    X, y = data["X"], data["y"]
    if len(X) == 0:
        return None, None
    return X.astype("float32"), y.astype("int64")


def build_tf_dataset(X, y, batch_size, augment: bool, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10_000), seed=CONFIG.RANDOM_SEED)
    ds = ds.batch(batch_size)
    if augment:
        augmenter = build_data_augmentation()
        ds = ds.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the handwritten digit CNN.")
    parser.add_argument("--epochs", type=int, default=CONFIG.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.LEARNING_RATE)
    parser.add_argument(
        "--no-custom-data",
        action="store_true",
        help="Ignore custom webcam data even if present, and train on MNIST only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(CONFIG.RANDOM_SEED)
    np.random.seed(CONFIG.RANDOM_SEED)

    logger.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()

    # Hold out a validation split from MNIST's training data.
    val_count = int(len(x_train) * CONFIG.VALIDATION_SPLIT)
    x_val, y_val = x_train[:val_count], y_train[:val_count]
    x_train, y_train = x_train[val_count:], y_train[val_count:]

    if not args.no_custom_data:
        cx_train, cy_train = load_custom_split(CONFIG.PROCESSED_DATA_DIR / "custom_train.npz")
        cx_val, cy_val = load_custom_split(CONFIG.PROCESSED_DATA_DIR / "custom_val.npz")

        if cx_train is not None:
            logger.info("Mixing in %d custom training samples", len(cx_train))
            x_train = np.concatenate([x_train, cx_train], axis=0)
            y_train = np.concatenate([y_train, cy_train], axis=0)
        if cx_val is not None:
            logger.info("Mixing in %d custom validation samples", len(cx_val))
            x_val = np.concatenate([x_val, cx_val], axis=0)
            y_val = np.concatenate([y_val, cy_val], axis=0)
    else:
        logger.info("Skipping custom data (--no-custom-data set)")

    logger.info(
        "Final dataset sizes -> train: %d, validation: %d, test: %d",
        len(x_train), len(x_val), len(x_test),
    )

    train_ds = build_tf_dataset(x_train, y_train, args.batch_size, augment=True, shuffle=True)
    val_ds = build_tf_dataset(x_val, y_val, args.batch_size, augment=False, shuffle=False)

    model = build_digit_cnn(learning_rate=args.learning_rate)
    model.summary(print_fn=lambda line: logger.info(line))

    CONFIG.TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CONFIG.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best epoch, but save again to be
    # explicit and to guarantee the file exists even if training was very
    # short.
    model.save(CONFIG.MODEL_PATH)
    logger.info("Model saved to %s", CONFIG.MODEL_PATH)

    class_indices = {str(i): str(i) for i in range(CONFIG.NUM_DIGIT_CLASSES)}
    with open(CONFIG.CLASS_INDICES_PATH, "w") as f:
        json.dump(class_indices, f, indent=2)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(CONFIG.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Final MNIST test accuracy: %.4f (loss %.4f)", test_accuracy, test_loss)
    print(f"\nTraining complete. MNIST test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {CONFIG.MODEL_PATH}")
    print("Next: run 'python scripts/evaluate.py' to evaluate in more detail,")
    print("then 'python app/app.py' to start the live webcam application.")


if __name__ == "__main__":
    main()
