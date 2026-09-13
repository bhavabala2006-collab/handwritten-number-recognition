"""
cnn_model.py
------------
Defines the CNN architecture used to classify a single handwritten digit
(0-9). The number "10" is NOT a class of this network - it is produced by
running this single-digit classifier twice (once per detected digit blob)
and combining the results. See src/inference/predictor.py and the README
section "Number 10 Handling" for details.

Architecture summary
---------------------
Input:  28x28x1 grayscale image, normalized to [0, 1]

    Conv2D(32, 3x3) -> BatchNorm -> ReLU
    Conv2D(32, 3x3) -> BatchNorm -> ReLU
    MaxPooling2D(2x2)
    Dropout(0.25)

    Conv2D(64, 3x3) -> BatchNorm -> ReLU
    Conv2D(64, 3x3) -> BatchNorm -> ReLU
    MaxPooling2D(2x2)
    Dropout(0.25)

    Flatten
    Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5)
    Dense(10, softmax)   # digits 0-9

This is a standard, well-understood "two conv blocks + dense head"
architecture that trains to >99% validation accuracy on MNIST within a
handful of epochs on CPU, while staying easy for a student to read.
"""

from tensorflow import keras
from tensorflow.keras import layers

from config import CONFIG


def build_digit_cnn(
    input_shape=None,
    num_classes: int = None,
    learning_rate: float = None,
) -> keras.Model:
    """Build and compile the single-digit CNN classifier."""
    input_shape = input_shape or (
        CONFIG.DIGIT_IMAGE_SIZE,
        CONFIG.DIGIT_IMAGE_SIZE,
        CONFIG.IMAGE_CHANNELS,
    )
    num_classes = num_classes or CONFIG.NUM_DIGIT_CLASSES
    learning_rate = learning_rate or CONFIG.LEARNING_RATE

    inputs = keras.Input(shape=input_shape, name="digit_image")

    # ---- Convolutional block 1 -------------------------------------
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ---- Convolutional block 2 -------------------------------------
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ---- Fully connected head ---------------------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="digit_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="digit_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_data_augmentation() -> keras.Sequential:
    """
    Returns a small Keras preprocessing pipeline for on-the-fly data
    augmentation during training (rotation, shift, zoom). Because it is
    made of Keras layers, it runs only during `model.fit(training=True)`
    and is automatically skipped during evaluation/inference.
    """
    return keras.Sequential(
        [
            layers.RandomRotation(CONFIG.AUGMENT_ROTATION_RANGE / 360.0),
            layers.RandomTranslation(
                height_factor=CONFIG.AUGMENT_HEIGHT_SHIFT_RANGE,
                width_factor=CONFIG.AUGMENT_WIDTH_SHIFT_RANGE,
            ),
            layers.RandomZoom(CONFIG.AUGMENT_ZOOM_RANGE),
        ],
        name="augmentation",
    )


if __name__ == "__main__":
    # Quick manual sanity check: `python src/model/cnn_model.py`
    m = build_digit_cnn()
    m.summary()
