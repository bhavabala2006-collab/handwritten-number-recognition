# models/trained_model/

This folder holds the trained model file and related artifacts, generated
locally by running:

```
python scripts/train.py
```

It is intentionally **not** committed to source control (see `.gitignore`)
because trained model files are large binary artifacts that are easy to
regenerate and awkward to diff/review in Git.

After training, you should see:

| File | Description |
|---|---|
| `digit_cnn.keras` | The trained CNN, loaded by the webcam app and by `scripts/evaluate.py` |
| `class_indices.json` | Maps the model's output index (0-9) to its digit label |
| `training_history.json` | Per-epoch loss/accuracy, useful for plotting learning curves |
| `confusion_matrix_mnist.png` | Confusion matrix on the MNIST test set (created by `scripts/evaluate.py`) |
| `confusion_matrix_custom.png` | Confusion matrix on your custom webcam test set, if you collected one |

If you delete this folder's contents, simply re-run `scripts/train.py` to
regenerate everything.
