"""
evaluate.py
-----------
Loads the trained model and reports detailed evaluation metrics on:
  1. The MNIST test set (10,000 images)
  2. The custom webcam test set, if scripts/preprocess_data.py produced one

Prints overall accuracy/loss plus a full per-class classification report
(precision/recall/F1) and saves a confusion matrix image so you can see
exactly which digits get confused with which (e.g. "1" vs "7").

Usage
-----
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model():
    from tensorflow import keras

    if not CONFIG.MODEL_PATH.exists():
        print(f"ERROR: no trained model found at {CONFIG.MODEL_PATH}")
        print("Run 'python scripts/train.py' first.")
        sys.exit(1)
    return keras.models.load_model(CONFIG.MODEL_PATH)


def load_mnist_test():
    from tensorflow import keras

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    y_test = y_test.astype("int64")
    return x_test, y_test


def load_custom_test():
    path = CONFIG.PROCESSED_DATA_DIR / "custom_test.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    X, y = data["X"], data["y"]
    if len(X) == 0:
        return None, None
    return X.astype("float32"), y.astype("int64")


def evaluate_split(model, X, y, name: str):
    loss, accuracy = model.evaluate(X, y, verbose=0)
    predictions = np.argmax(model.predict(X, verbose=0), axis=1)

    print(f"\n=== {name} ===")
    print(f"Samples: {len(X)}")
    print(f"Loss: {loss:.4f}   Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y, predictions, digits=3, zero_division=0))

    return confusion_matrix(y, predictions, labels=list(range(10)))


def save_confusion_matrix_plot(cm, out_path: Path, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def main():
    model = load_model()

    x_mnist, y_mnist = load_mnist_test()
    cm_mnist = evaluate_split(model, x_mnist, y_mnist, "MNIST test set")
    save_confusion_matrix_plot(
        cm_mnist, CONFIG.TRAINED_MODEL_DIR / "confusion_matrix_mnist.png", "MNIST Test Confusion Matrix"
    )

    x_custom, y_custom = load_custom_test()
    if x_custom is not None:
        cm_custom = evaluate_split(model, x_custom, y_custom, "Custom webcam test set")
        save_confusion_matrix_plot(
            cm_custom,
            CONFIG.TRAINED_MODEL_DIR / "confusion_matrix_custom.png",
            "Custom Webcam Test Confusion Matrix",
        )
    else:
        print(
            "\n(No custom test set found - run scripts/collect_data.py and "
            "scripts/preprocess_data.py to create one and get a more "
            "realistic accuracy estimate for webcam use.)"
        )


if __name__ == "__main__":
    main()
