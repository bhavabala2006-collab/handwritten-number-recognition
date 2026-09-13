/**
 * script.js
 * ---------
 * Handles:
 *   - Requesting webcam access and starting/stopping the video stream
 *   - Drawing the fixed detection-area (ROI) rectangle over the video
 *   - Periodically capturing just the ROI region and sending it to the
 *     Flask backend's /predict endpoint
 *   - Updating the prediction / confidence / status UI
 *   - The "Clear" button, which resets both the UI and the backend's
 *     temporal smoothing history (see src/inference/smoothing.py)
 *
 * NOTE: ROI_FRACTION below must match CONFIG.ROI_FRACTION in config.py
 * so the collection script, training data, and live app all agree on
 * where the "detection area" is.
 */

const ROI_FRACTION = { x0: 0.30, y0: 0.25, x1: 0.70, y1: 0.75 };
const CAPTURE_INTERVAL_MS = 350; // ~3 predictions/second is plenty for stability

const video = document.getElementById("video");
const overlay = document.getElementById("roi-overlay");
const captureCanvas = document.getElementById("capture-canvas");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const clearBtn = document.getElementById("clear-btn");
const statusLine = document.getElementById("status-line");

const numberDisplay = document.getElementById("number-display");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceValue = document.getElementById("confidence-value");
const statusBadge = document.getElementById("status-badge");
const rawNumberEl = document.getElementById("raw-number");
const rawConfidenceEl = document.getElementById("raw-confidence");

let mediaStream = null;
let captureTimer = null;

function drawRoiOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;

  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const x = ROI_FRACTION.x0 * overlay.width;
  const y = ROI_FRACTION.y0 * overlay.height;
  const w = (ROI_FRACTION.x1 - ROI_FRACTION.x0) * overlay.width;
  const h = (ROI_FRACTION.y1 - ROI_FRACTION.y0) * overlay.height;

  ctx.strokeStyle = "#4caf50";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);

  ctx.fillStyle = "#4caf50";
  ctx.font = "14px Segoe UI, Arial, sans-serif";
  ctx.fillText("Write your number here", x, y - 8 > 10 ? y - 8 : y + h + 18);
}

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (err) {
    statusLine.textContent =
      "Could not access the webcam. Check browser permissions and that no other app is using the camera.";
    console.error("getUserMedia error:", err);
    return;
  }

  video.srcObject = mediaStream;
  await video.play();

  drawRoiOverlay();
  window.addEventListener("resize", drawRoiOverlay);

  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusLine.textContent = "Camera running - hold your handwriting inside the box.";

  captureTimer = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);
}

function stopCamera() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusLine.textContent = "Camera stopped.";
}

/**
 * Capture ONLY the ROI region from the current video frame, accounting
 * for the fact that the <video> element is mirrored via CSS
 * (transform: scaleX(-1)) so writing feels natural, like looking in a
 * mirror. We mirror the canvas draw the same way so the region we crop
 * matches exactly what the user sees inside the green box.
 */
function captureRoiFrame() {
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  if (!videoWidth || !videoHeight) return null;

  // Draw the full (mirrored) frame onto an offscreen canvas first.
  const fullCanvas = document.createElement("canvas");
  fullCanvas.width = videoWidth;
  fullCanvas.height = videoHeight;
  const fullCtx = fullCanvas.getContext("2d");
  fullCtx.translate(videoWidth, 0);
  fullCtx.scale(-1, 1);
  fullCtx.drawImage(video, 0, 0, videoWidth, videoHeight);

  // Now crop out just the ROI, in native video resolution.
  const rx = ROI_FRACTION.x0 * videoWidth;
  const ry = ROI_FRACTION.y0 * videoHeight;
  const rw = (ROI_FRACTION.x1 - ROI_FRACTION.x0) * videoWidth;
  const rh = (ROI_FRACTION.y1 - ROI_FRACTION.y0) * videoHeight;

  captureCanvas.width = rw;
  captureCanvas.height = rh;
  const captureCtx = captureCanvas.getContext("2d");
  captureCtx.drawImage(fullCanvas, rx, ry, rw, rh, 0, 0, rw, rh);

  return captureCanvas.toDataURL("image/jpeg", 0.85);
}

async function captureAndPredict() {
  const dataUrl = captureRoiFrame();
  if (!dataUrl) return;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      statusLine.textContent = errBody.error || "Prediction request failed.";
      return;
    }

    const result = await response.json();
    updateResultUI(result);
  } catch (err) {
    console.error("Prediction request error:", err);
    statusLine.textContent = "Lost connection to the server.";
  }
}

function updateResultUI(result) {
  const { number, confidence, status, raw_number, raw_confidence } = result;

  numberDisplay.textContent = number !== null && number !== undefined ? number : "—";

  const pct = Math.round((confidence || 0) * 100);
  confidenceBar.style.width = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;

  rawNumberEl.textContent = raw_number !== null && raw_number !== undefined ? raw_number : "—";
  rawConfidenceEl.textContent = `${Math.round((raw_confidence || 0) * 100)}%`;

  statusBadge.classList.remove(
    "status-idle",
    "status-ok",
    "status-uncertain",
    "status-empty",
    "status-unsupported"
  );

  let badgeText = "Idle";
  let badgeClass = "status-idle";

  switch (status) {
    case "ok":
      badgeText = "Confident";
      badgeClass = "status-ok";
      break;
    case "uncertain":
      badgeText = "Low confidence";
      badgeClass = "status-uncertain";
      break;
    case "empty":
      badgeText = "No handwriting detected";
      badgeClass = "status-empty";
      break;
    case "unsupported":
      badgeText = "Not a recognized number (1-10)";
      badgeClass = "status-unsupported";
      break;
    default:
      break;
  }

  statusBadge.textContent = badgeText;
  statusBadge.classList.add(badgeClass);

  confidenceBar.style.background =
    status === "ok" ? "#4caf50" : status === "uncertain" ? "#e0a552" : "#555";
}

async function clearResult() {
  try {
    await fetch("/reset", { method: "POST" });
  } catch (err) {
    console.error("Reset request error:", err);
  }
  numberDisplay.textContent = "—";
  confidenceBar.style.width = "0%";
  confidenceValue.textContent = "0%";
  rawNumberEl.textContent = "—";
  rawConfidenceEl.textContent = "0%";
  statusBadge.textContent = "Idle";
  statusBadge.className = "status-badge status-idle";
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
clearBtn.addEventListener("click", clearResult);

window.addEventListener("beforeunload", stopCamera);
