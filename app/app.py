"""
app.py
------
Flask backend for the real-time handwritten number recognition web app.

Responsibilities
-----------------
1. Serve the single-page frontend (templates/index.html + static assets).
2. Expose a `/predict` endpoint: the browser captures the ROI from the
   live <video> feed onto a <canvas>, encodes it as a base64 JPEG, and
   POSTs it here. This endpoint decodes the image, runs it through the
   full preprocessing + CNN + number-composition pipeline, applies
   temporal smoothing, and returns JSON.
3. Expose a `/reset` endpoint used by the "Clear" button to reset the
   smoothing history.

Run with:
    python app/app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.inference.predictor import get_predictor
from src.inference.smoothing import PredictionSmoother
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

# A single global smoother is fine for this project: it's a local,
# single-user demo application (one person, one webcam, one browser tab).
# See README "Limitations" section for how this could be extended to
# multiple concurrent users (e.g. a per-session smoother keyed by a
# session/browser id).
_smoother = PredictionSmoother()


def decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a `data:image/...;base64,...` string into a BGR numpy array."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    binary_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(binary_data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    return render_template(
        "index.html",
        confidence_threshold=CONFIG.CONFIDENCE_THRESHOLD,
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    frame = decode_base64_image(image_data)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        predictor = get_predictor()
        raw_result = predictor.predict_from_roi(frame)
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        return (
            jsonify(
                {
                    "error": (
                        "No trained model found. Run 'python scripts/train.py' "
                        "first, then restart the app."
                    )
                }
            ),
            500,
        )
    except Exception:  # noqa: BLE001 - surface a clean error to the UI
        logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Unexpected error during prediction. See server logs."}), 500

    smoothed = _smoother.update(raw_result)
    return jsonify(smoothed)


@app.route("/reset", methods=["POST"])
def reset():
    _smoother.reset()
    return jsonify({"status": "reset"})


@app.route("/health")
def health():
    """Simple endpoint to confirm the server + model are ready."""
    model_ready = CONFIG.MODEL_PATH.exists()
    return jsonify({"status": "ok", "model_ready": model_ready})


if __name__ == "__main__":
    if not CONFIG.MODEL_PATH.exists():
        logger.warning(
            "No trained model found at %s. The app will start, but /predict "
            "will fail until you run 'python scripts/train.py'.",
            CONFIG.MODEL_PATH,
        )
    app.run(host=CONFIG.FLASK_HOST, port=CONFIG.FLASK_PORT, debug=CONFIG.FLASK_DEBUG)
