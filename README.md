# Real-Time Handwritten Number Recognition Using Webcam

A complete, runnable AI/ML application that uses your computer's webcam to
recognize handwritten numbers **1 through 10** in real time, in the
browser, with a live confidence score.

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [How It Works](#how-it-works)
- [AI/ML Architecture](#aiml-architecture)
- [Dataset](#dataset)
- [Number 10 Handling](#number-10-handling)
- [Image Preprocessing](#image-preprocessing)
- [Webcam Detection](#webcam-detection)
- [Prediction Smoothing](#prediction-smoothing)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Running the Webcam Application](#running-the-webcam-application)
- [Using the Application](#using-the-application)
- [Troubleshooting](#troubleshooting)
- [Improving Accuracy](#improving-accuracy)
- [Performance](#performance)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Description

This project lets you hold up a piece of paper (or write on a whiteboard)
in front of your webcam, and see the application recognize which number
from **1 to 10** you wrote, live, along with a confidence percentage.

Under the hood it combines classic computer vision (OpenCV) for locating
and cleaning up the handwriting with a Convolutional Neural Network (CNN,
built with TensorFlow/Keras) for actually reading the digit(s). A small
Flask web server ties everything together into a simple browser-based
interface — no separate desktop GUI toolkit required.

## Features

- **Live webcam feed** with a clearly marked detection area (ROI).
- **Real-time inference** — predictions update several times per second.
- **Recognizes 1–10**, including the two-digit number "10", via a
  purpose-built two-stage detection + classification design.
- **Confidence score** shown as both a percentage and a progress bar.
- **Prediction smoothing** so the displayed result doesn't flicker while
  you're still writing.
- **Custom dataset collection tool** so you can fine-tune the model on
  your own handwriting, pen, paper, and lighting conditions.
- **Full training/evaluation pipeline** — nothing is a black box; you can
  retrain the model from scratch at any time.
- **Runs entirely on CPU** — no GPU required.
- **Automated tests** covering preprocessing, the model, and inference.

## How It Works

```
Webcam
  → Frame capture            (browser captures a frame from the <video> feed)
  → Region of Interest (ROI) (a fixed box the user writes inside)
  → Grayscale conversion     (drop color information - only shape matters)
  → Noise reduction          (Gaussian blur smooths out sensor/lighting noise)
  → Thresholding             (adaptive threshold -> clean black/white image)
  → Contour detection        (find each separate ink blob = one digit stroke)
  → Crop + pad + resize      (each digit blob -> a 28x28 image, MNIST-style)
  → CNN                      (classifies each 28x28 image as a digit 0-9)
  → Number composition       (combine 1 or 2 digit predictions into 1-10)
  → Prediction smoothing     (majority vote across recent frames)
  → UI                       (number + confidence + status shown live)
```

Every one of these stages is implemented as a small, dedicated, tested
function rather than one big script — see
[Project Structure](#project-structure) for exactly where each stage
lives in the code.

## AI/ML Architecture

**What is a CNN, and why use one here?**
A Convolutional Neural Network is a type of neural network that learns to
recognize visual patterns (edges, curves, loops) by sliding small filters
across an image. This makes it naturally suited to handwriting
recognition, where the same digit can appear in slightly different
positions, sizes, or stroke styles, but the local shapes (a loop for "0",
a vertical stroke for "1") stay recognizable.

**Design decision — why a single-digit classifier, not an 11-class
classifier?**
"1" through "9" are single strokes, but "10" is fundamentally *two*
digits ("1" and "0") written side by side. Rather than inventing an
artificial 11th class for "10" (which a digit-shaped CNN has no natural
way to represent, since "10" isn't a single visual glyph), this project
uses a **two-stage architecture**, as suggested by the project brief:

- **Stage 1 — Segmentation (classical CV, not learned):** OpenCV contour
  detection looks at the cleaned-up ROI and finds 1 or 2 separate ink
  blobs, ordered left-to-right.
- **Stage 2 — Classification (the CNN):** each individual blob is resized
  to 28x28 and classified as a digit 0-9 by the same, single CNN.
- **Composition (plain Python logic):** the digit prediction(s) are
  combined: one blob → that digit is the number (1-9); two blobs → if
  they read "1" then "0", the number is 10.

This keeps the model itself simple (10-class MNIST-style classifier,
extremely well understood, fast to train, easy for a beginner to reason
about) while still correctly handling the two-digit "10" case. See
[Number 10 Handling](#number-10-handling) below for the full detail, and
`src/inference/number_composer.py` for the implementation.

**Model specification:**

| Property | Value |
|---|---|
| Input image size | 28 x 28 x 1 (grayscale) |
| Output classes | 10 (digits 0-9) |
| Conv layers | 4 (two 32-filter, two 64-filter, all 3x3) |
| Activation | ReLU (hidden layers), Softmax (output layer) |
| Pooling | 2x2 max pooling after each conv block |
| Regularization | Batch Normalization + Dropout (0.25 / 0.25 / 0.5) |
| Dense head | Flatten -> Dense(256) -> Dense(10, softmax) |
| Loss function | Sparse categorical cross-entropy |
| Optimizer | Adam |
| Data augmentation | Random rotation, translation, zoom (applied only during training) |

See `src/model/cnn_model.py` for the exact, fully commented Keras
implementation.

**Training process:** the model trains on MNIST (60,000 training images,
10,000 held out for testing), with 10% of the training set further held
out for validation. If you've collected custom webcam samples (see
[Dataset](#dataset)), those are automatically mixed into the training and
validation sets. Training uses early stopping and learning-rate reduction
on plateau so it won't obviously overfit or waste time once accuracy
stops improving.

**Validation process:** validation accuracy is tracked every epoch;
`ModelCheckpoint` keeps only the best-performing version of the model on
disk.

**Testing process:** after training, `scripts/evaluate.py` runs the saved
model against the full MNIST test set (and your custom test set, if any)
and reports accuracy, a per-digit precision/recall/F1 report, and a
confusion matrix image.

## Dataset

**Primary dataset — MNIST:** the classic dataset of 70,000 handwritten
digit images (0-9), downloaded automatically the first time you run
`scripts/train.py` (via `tensorflow.keras.datasets.mnist`, no manual
download needed). It gives the model a strong, general foundation for
recognizing digit shapes.

**Why MNIST alone isn't quite enough:** MNIST digits are scanned,
centered, evenly lit, and drawn on a uniform background. A live webcam
feed of you holding up a piece of paper has none of those guarantees —
uneven lighting, pen thickness, paper texture, and camera noise. So this
project also supports a **custom dataset** collected directly from your
own webcam, which is mixed into training automatically if present.

**Dataset structure on disk:**

```
data/
├── raw/                 # full ROI snapshots you capture, one folder per NUMBER (1-10)
│   ├── 1/  2/  ...  10/
├── processed/            # .npz files of labeled DIGIT crops (0-9) used for training
│   ├── custom_train.npz
│   ├── custom_val.npz
│   └── custom_test.npz
├── train/                # human-inspectable PNG copies of the digit crops, per DIGIT (0-9)
├── validation/
└── test/
```

**Important note on folder labels:** `data/raw/` is organized by the
**product-facing number** you wrote (1-10), because that's what you, the
person collecting data, are writing and thinking about. But the CNN
itself is a *digit* classifier (0-9) — so `scripts/preprocess_data.py`
automatically segments each raw sample into its individual digit(s) and
re-labels them by digit before splitting into `data/train/`,
`data/validation/`, and `data/test/` (each organized into `0/` .. `9/`
subfolders). This is explained in more detail in
[Number 10 Handling](#number-10-handling).

**Collecting your own data:** see
[Dataset Preparation](#dataset-preparation) below — `scripts/collect_data.py`
gives you an interactive webcam tool so you never have to manually create
files.

## Number 10 Handling

This is the trickiest requirement in the whole project, so here is the
complete explanation in one place.

**The problem:** the CNN can only ever classify a single 28x28 image as
one digit, 0 through 9. But "10" is written as two digits next to each
other. A digit classifier has no natural "10th class" that looks like
"10" — it's not a single glyph the way "7" or "3" is.

**The solution — segment first, classify second, combine last:**

1. **Segment** (`src/preprocessing/image_processing.segment_digits`):
   OpenCV's contour detection finds every separate ink blob in the
   thresholded ROI and returns their bounding boxes sorted left-to-right.
   Writing "7" produces 1 blob. Writing "10" produces 2 blobs (the "1"
   and the "0").
2. **Classify each blob independently**
   (`src/inference/predictor.DigitPredictor.predict_digit_batch`): each
   blob is cropped, padded, resized to 28x28, and run through the CNN,
   producing a predicted digit (0-9) and confidence for *that blob only*.
3. **Compose the final number**
   (`src/inference/number_composer.compose_number`):
   - **1 blob**, predicted digit `d` (1-9) → the number is `d`.
   - **1 blob**, predicted digit `0` → invalid (0 alone isn't in our 1-10
     range) — reported as `unsupported`.
   - **2 blobs**, predicted as `1` then `0` (in that left-to-right order)
     → the number is **10**.
   - **2 blobs**, any other combination → invalid/`unsupported` (not a
     number this product recognizes).
   - **0 blobs** → `empty` (nothing written yet).
   - **3+ blobs** → `unsupported` (likely noise, stray marks, or writing
     that touched/overlapped — ask the user to rewrite more clearly).

**Why this is more robust than an 11-class model:** it reuses a single,
well-tested digit classifier for both single- and multi-digit cases,
keeps the model itself simple and fast to train, and makes the "10"
logic fully transparent and easy to debug (you can print exactly which
digits were detected and why a result was accepted or rejected).

**Practical tip:** when writing "10", write the "1" and "0" reasonably
close together (like a normal two-digit number) but make sure they don't
visually touch/merge into one blob — a small gap between them helps the
contour detector separate them correctly.

## Image Preprocessing

All of the following steps live in
`src/preprocessing/image_processing.py`, and — critically — **the exact
same functions are used for custom data preprocessing and for live
inference.** This project deliberately avoids the common bug where a
model is trained on one kind of image preparation and then given
differently-prepared images at inference time.

1. **Grayscale conversion** (`to_grayscale`) — drop color, keep shape.
2. **Gaussian blur / noise reduction** (`reduce_noise`) — smooths out
   sensor noise and small lighting artifacts before thresholding.
3. **Adaptive thresholding / binarization** (`binarize`) — converts the
   grayscale image into clean black background / white foreground
   strokes, using a *local* (adaptive) threshold so uneven lighting
   across the page doesn't break detection. A light morphological
   open+dilate cleans up speckle noise and small gaps.
4. **Contour detection** (`find_digit_bounding_boxes`) — finds each
   separate stroke/blob, filters out anything too small to be a real
   digit (`MIN_CONTOUR_AREA` in `config.py`), merges fragments that
   likely belong to the same digit, and sorts left-to-right.
5. **Cropping + padding** (`crop_with_padding`) — crops each detected
   blob out of the binary image with a margin so strokes aren't clipped.
6. **Resize + normalize** (`resize_and_normalize`) — resizes the digit
   onto a 28x28 canvas the same way MNIST digits are prepared (digit
   scaled to fit roughly 20x20 pixels, centered in the 28x28 canvas), then
   scales pixel values from `[0, 255]` to `[0.0, 1.0]`.

## Webcam Detection

The webcam itself is accessed **in the browser**, via the standard
`navigator.mediaDevices.getUserMedia` API (see `app/static/js/script.js`)
— this avoids OpenCV camera-access quirks across different operating
systems and lets the app run through a normal web page.

- A fixed, clearly outlined **detection area (ROI)** is drawn over the
  video feed (green rectangle).
- Roughly 3 times per second, the JavaScript captures **only the ROI
  region** from the current video frame, encodes it as a JPEG, and POSTs
  it to the Flask backend's `/predict` endpoint.
- The backend decodes the image, runs the full preprocessing + CNN
  pipeline, and returns the prediction as JSON.
- If the ROI is currently blank (no ink detected), the backend
  short-circuits and reports `status: "empty"` without wasting a model
  call.

*(Optional local CLI alternative: `scripts/test_model.py --webcam` lets
you test the pipeline directly with OpenCV's own camera window, without
the browser, useful for quick debugging.)*

## Prediction Smoothing

Raw, frame-by-frame predictions can flicker — e.g. showing `7, 3, 7, 8,
7` in under a second while you're mid-stroke on a "7". This is solved in
`src/inference/smoothing.PredictionSmoother`:

- It keeps a sliding window of the last `SMOOTHING_WINDOW` (default 7)
  raw predictions.
- The number displayed to the user only **changes** once the same number
  has appeared at least `SMOOTHING_MIN_AGREEMENT` (default 4) times
  within that window (simple majority-vote smoothing).
- Until a new number reaches that agreement threshold, the **previous**
  stable result stays on screen instead of blanking out — so the UI
  feels calm and confident rather than jumpy.
- Pressing **Clear** resets the smoothing history via the `/reset`
  endpoint, so leftover votes from your last number don't bias the next
  one.

Both `SMOOTHING_WINDOW` and `SMOOTHING_MIN_AGREEMENT` are configurable in
`config.py`.

## Project Structure

```
handwritten-number-recognition/
│
├── README.md                    <- you are here
├── requirements.txt
├── .gitignore
├── LICENSE
├── config.py                    <- all tunable settings in one place
├── pytest.ini
│
├── data/
│   ├── raw/<1-10>/               <- your raw collected webcam samples
│   ├── processed/                <- .npz files of labeled digit crops
│   ├── train/<0-9>/               <- inspectable PNGs of the training digit crops
│   ├── validation/<0-9>/
│   └── test/<0-9>/
│
├── models/
│   ├── trained_model/             <- digit_cnn.keras + metrics (generated by train.py)
│   └── README.md
│
├── scripts/
│   ├── collect_data.py           <- interactive webcam data collection tool
│   ├── preprocess_data.py        <- raw samples -> labeled digit crops -> train/val/test split
│   ├── train.py                  <- trains and saves the CNN
│   ├── evaluate.py               <- detailed evaluation + confusion matrices
│   └── test_model.py             <- quick manual CLI test (image file or single webcam shot)
│
├── src/
│   ├── preprocessing/
│   │   └── image_processing.py   <- the shared preprocessing pipeline (training + inference)
│   ├── model/
│   │   └── cnn_model.py          <- CNN architecture + data augmentation
│   ├── inference/
│   │   ├── predictor.py          <- loads the model, runs the full ROI -> NumberResult pipeline
│   │   ├── number_composer.py    <- the "10" handling logic
│   │   └── smoothing.py          <- temporal prediction smoothing
│   └── utils/
│       └── logger.py             <- shared, consistent logging setup
│
├── app/
│   ├── app.py                    <- Flask backend (/, /predict, /reset, /health)
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js          <- webcam capture, ROI overlay, prediction polling
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_inference.py
│
└── notebooks/
    └── experimentation.ipynb     <- optional scratchpad for visualizing data/results
```

## Requirements

- **Python:** 3.9 – 3.11 recommended (TensorFlow 2.16 supports these).
- **pip:** any recent version.
- **Virtual environment:** strongly recommended (instructions below).
- **Webcam:** any standard USB or built-in webcam recognized by your OS
  and browser.
- **GPU/CUDA:** **not required.** Everything in this project (training
  and inference) is designed to run comfortably on a normal CPU. If you
  do have a CUDA-compatible GPU and a matching `tensorflow` GPU build
  installed, training will simply be faster automatically — no code
  changes needed.
- **OS:** Windows, macOS, or Linux. All file paths use `pathlib` and are
  cross-platform.
- **Browser:** any modern browser that supports `getUserMedia` (Chrome,
  Edge, Firefox). The page must be served from `http://127.0.0.1` (or
  `https://`), since browsers block camera access on plain `http://` for
  non-localhost addresses.

All Python dependencies are listed in `requirements.txt`.

## Installation

Clone or download this project, then open a terminal **in the project's
root folder** (the one containing this `README.md`).

**1. Check your Python version:**

```
python --version
```

**2. Create a virtual environment:**

```
python -m venv venv
```

**3. Activate it:**

Windows CMD:
```
venv\Scripts\activate.bat
```

Windows PowerShell:
```
venv\Scripts\Activate.ps1
```
*(If PowerShell blocks the script, see [Troubleshooting](#troubleshooting).)*

Linux / macOS:
```
source venv/bin/activate
```

**4. Install dependencies:**

```
pip install -r requirements.txt
```

## Dataset Preparation

You have two options, and they can be combined:

**Option A — MNIST only (fastest way to get started):**
Do nothing extra. `scripts/train.py` downloads and uses MNIST
automatically the first time you run it.

**Option B — Add your own webcam data (recommended for real-world
accuracy):**

1. Collect samples for each number 1-10:
   ```
   python scripts/collect_data.py --class 7 --samples 60
   ```
   or cycle through every number in one session:
   ```
   python scripts/collect_data.py --interactive
   ```
   Controls: `c` = capture one sample, `a` = toggle auto-capture
   (captures automatically whenever writing is detected), `n` = next
   class (interactive mode), `q` = quit. Aim for at least 40-60 samples
   per number, written in a few different positions/angles for variety.
   Remember: for "10", write the "1" and "0" as two clearly separate
   strokes.

2. Turn your raw captures into labeled training data:
   ```
   python scripts/preprocess_data.py
   ```
   This segments each raw sample into individual digit crops, splits
   them into train/validation/test sets, and saves everything under
   `data/processed/` (plus inspectable PNGs under `data/train/`,
   `data/validation/`, `data/test/`).

## Training the Model

```
python scripts/train.py
```

What this does:
- Downloads MNIST (first run only; cached afterwards by Keras).
- Automatically mixes in your custom data from `data/processed/`, if
  present.
- Builds the CNN (`src/model/cnn_model.py`) and trains it with data
  augmentation, early stopping, and learning-rate reduction.
- Saves the best model to `models/trained_model/digit_cnn.keras`, plus
  `class_indices.json` and `training_history.json`.

Optional flags:
```
python scripts/train.py --epochs 25 --batch-size 64 --learning-rate 0.0005
python scripts/train.py --no-custom-data   # train on MNIST only, even if custom data exists
```

Training on CPU typically takes a few minutes for the default 15 epochs
on MNIST alone (exact time depends on your machine).

## Evaluating the Model

```
python scripts/evaluate.py
```

Expected output: overall loss/accuracy on the MNIST test set (typically
>99% for this architecture), a full per-digit precision/recall/F1
classification report, and — if you have a custom test set — the same
metrics for your own webcam data (usually a more realistic estimate of
real-world performance than MNIST alone). Confusion matrix images are
saved to `models/trained_model/confusion_matrix_mnist.png` and
`confusion_matrix_custom.png`.

## Running the Webcam Application

```
python app/app.py
```

Then open your browser to:

```
http://127.0.0.1:5000
```

## Using the Application

1. Start the application (`python app/app.py`) and open the URL above.
2. Click **Start Camera** and allow the browser's webcam permission
   prompt.
3. Hold your handwritten number so it sits fully inside the green
   detection box.
4. Write clearly, with the "1" and "0" of "10" as two distinct strokes if
   applicable.
5. Wait about a second — the prediction and confidence will stabilize
   (this is the smoothing mechanism at work; see
   [Prediction Smoothing](#prediction-smoothing)).
6. Read the confidence percentage and status badge (Confident / Low
   confidence / No handwriting detected / Not a recognized number).
7. Click **Clear** before writing a new number, to reset the smoothing
   history and avoid a mix of old and new predictions.
8. Click **Stop Camera** when you're done.

## Troubleshooting

**Webcam not detected / black video area:**
Confirm no other application (Zoom, Teams, another browser tab) is
currently using the camera, and that your OS's privacy settings allow
browser camera access.

**Webcam permission denied:**
Check your browser's site settings (usually a camera icon in the address
bar) and allow camera access for `127.0.0.1`, then reload the page.

**OpenCV camera error when running `scripts/collect_data.py` /
`scripts/test_model.py --webcam`:**
Try a different `--camera-index` (0, 1, 2...) — some systems enumerate
webcams differently, especially with a built-in laptop camera plus an
external one.

**TensorFlow installation failure:**
Ensure you're using a supported Python version (3.9-3.11) inside the
virtual environment. On Apple Silicon Macs, `tensorflow` should still
install via pip for CPU use; if you hit issues, check the latest
TensorFlow installation notes for macOS.

**"Model file missing" error in the app:**
You need to train the model first: `python scripts/train.py`. The app
checks for `models/trained_model/digit_cnn.keras` on every prediction
request and will return a clear error message if it isn't found.

**Low prediction accuracy / wrong predictions:**
See [Improving Accuracy](#improving-accuracy) below.

**Number 10 not detected correctly:**
Make sure the "1" and "0" don't touch each other (the contour detector
needs a visible gap to treat them as two separate blobs) and that both
digits are fully inside the detection box. If they're too close, the
merging logic in `find_digit_bounding_boxes` may combine them into one
blob, which the composer will treat as `unsupported`.

**Python version problems:**
Run `python --version` and confirm it's 3.9-3.11. If your system's
default `python` points to Python 2 or an unsupported Python 3 version,
try `python3` or `py -3.11` instead when creating the virtual
environment.

**Port already in use (Flask):**
Change `FLASK_PORT` in `config.py` to a free port (e.g. `5050`), or stop
whatever else is using port 5000.

**Missing dependencies:**
Re-run `pip install -r requirements.txt` inside the activated virtual
environment; make sure the `(venv)` prefix is showing in your terminal
prompt.

**Windows PowerShell activation issues** (`cannot be loaded because
running scripts is disabled on this system`):
Run PowerShell as Administrator once and execute:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then retry `venv\Scripts\Activate.ps1`. Alternatively, use Windows CMD's
`venv\Scripts\activate.bat` instead.

## Improving Accuracy

- **Collect more custom samples** — 40-60 per number is a reasonable
  minimum; more (and more varied) is better.
- **Increase handwriting diversity** — different pens, slightly
  different stroke styles, different people if possible.
- **Use consistent, good lighting** — avoid strong backlight or harsh
  shadows across the writing surface.
- **Use a clear, plain, high-contrast background** — a plain white or
  light-colored page works best with the current thresholding approach.
- **Keep the writing consistently inside the ROI** — avoid partially
  cropped digits.
- **Train for more epochs** — increase `--epochs` if validation accuracy
  is still improving when training stops.
- **Tune hyperparameters** — try different `--learning-rate` or
  `--batch-size` values; watch `models/trained_model/training_history.json`
  or the notebook's training-curve plot for over/underfitting.
- **Adjust `CONFIDENCE_THRESHOLD`, `MIN_CONTOUR_AREA`, or
  `DIGIT_CROP_PADDING` in `config.py`** if digits are being segmented
  incorrectly for your specific setup.

## Performance

Exact numbers depend heavily on your CPU, so treat the following as
**approximate expectations**, not guaranteed benchmarks:

- **Training:** on a typical modern laptop CPU, training on MNIST alone
  for the default 15 epochs generally finishes in a few minutes.
- **Inference (single prediction):** a single digit classification on
  CPU typically takes well under 100ms once the model is loaded, so the
  ROI capture-and-predict loop (every ~350ms, configurable in
  `script.js`) comfortably keeps up in real time on ordinary laptop
  hardware.
- **Accuracy:** this CNN architecture typically reaches **>99%** accuracy
  on the MNIST test set. Real-world webcam accuracy will usually be
  somewhat lower unless you've added custom webcam training data (see
  [Improving Accuracy](#improving-accuracy)), since webcam conditions
  differ meaningfully from MNIST's clean, scanned digits.

## Limitations

- The system recognizes numbers **1 through 10 only** — no decimals,
  negative numbers, or numbers above 10.
- It expects roughly one or two clearly separated digit strokes per
  prediction; overlapping/touching digits, multiple numbers in the ROI
  at once, or heavily stylized cursive-like writing are not supported.
- It works best with a clear, dark pen on a plain, light background;
  very low contrast or extremely cluttered backgrounds will reduce
  segmentation quality.
- The current Flask app keeps a single, global smoothing state, which is
  fine for one person using the app at a time on their own machine, but
  would need a per-session smoother (e.g. keyed by a browser/session ID)
  to correctly support multiple simultaneous users.
- Recognition quality depends significantly on lighting and camera
  quality, as with any webcam-based computer vision system.

## Future Improvements

- Better multi-digit / multi-number recognition (numbers beyond 10,
  decimals, simple arithmetic expressions).
- Mobile support (responsive UI, and/or a native mobile camera pipeline).
- Optional GPU acceleration for faster training on larger custom datasets.
- Replacing classical contour-based segmentation with a learned object
  detector for more robust digit localization in cluttered scenes.
- Hand/pen tip tracking to draw directly in the air instead of on paper.
- Fully browser-only inference (e.g. TensorFlow.js) to remove the need
  for a Python backend at prediction time.
- Exporting the model to TensorFlow Lite or ONNX for lightweight/embedded
  deployment.
- Cloud deployment (containerizing the Flask app for a hosted demo).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE)
for the full text.
