"""Loading and inference for the steel surface defect detector.

The detector is a YOLO26n model fine-tuned on NEU-DET (see scripts/train.py).
This module is the only place that knows about YOLO: the API layer just calls
load() and detect(), so another detector can be swapped in here without
touching main.py.
"""

import io
import json
import os
import threading
from pathlib import Path

# A server must never pip-install packages because of a request. Ultralytics
# auto-installs missing extras by default; switch that off before importing it.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Ultralytics replaces PIL.Image.open with a version that tries to install a
# HEIF plugin when it meets an unknown file. Uploads are untrusted, so decode
# them with Pillow's original function. Covered by test_rejects_non_image.
from ultralytics.utils.patches import _image_open as _pil_open  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH") or MODELS_DIR / "defect_yolo26n.pt")
METRICS_PATH = MODELS_DIR / "metrics.json"

IMG_SIZE = 640
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_SIDE = 4096

# Ultralytics models are not thread-safe; requests run in a thread pool.
_predict_lock = threading.Lock()


class InvalidImageError(ValueError):
    """The uploaded bytes are not an image we can run inference on."""


def load() -> YOLO:
    """Load the trained weights. Fails fast if they are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model weights not found at {MODEL_PATH}; run scripts/train.py"
        )
    model = YOLO(str(MODEL_PATH), task="detect")
    # Warm-up run so the first real request is not slow.
    model.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), verbose=False)
    return model


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def decode_image(data: bytes) -> Image.Image:
    """Turn uploaded bytes into an RGB image, rejecting anything unsafe."""
    if not data:
        raise InvalidImageError("empty file")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError(
            f"file larger than {MAX_IMAGE_BYTES // 1024 // 1024} MB"
        )
    try:
        image = _pil_open(io.BytesIO(data))
        image.verify()  # cheap integrity check without decoding pixels
        image = _pil_open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("file is not a valid image") from exc
    if max(image.size) > MAX_IMAGE_SIDE:
        raise InvalidImageError(f"image side larger than {MAX_IMAGE_SIDE} px")
    return image.convert("RGB")


def _predict(model: YOLO, image: Image.Image, conf: float):
    with _predict_lock:
        return model.predict(image, imgsz=IMG_SIZE, conf=conf, verbose=False)[0]


def detect(model: YOLO, image: Image.Image, conf: float = 0.25) -> list[dict]:
    """Run the detector on one image and return its detections."""
    result = _predict(model, image, conf)
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        cls = int(box.cls[0])
        detections.append(
            {
                "label": result.names[cls],
                "confidence": round(float(box.conf[0]), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def annotate(model: YOLO, image: Image.Image, conf: float = 0.25) -> bytes:
    """Return the image as JPEG with the detected boxes drawn on it."""
    result = _predict(model, image, conf)
    plotted = result.plot()[:, :, ::-1]  # BGR -> RGB
    buffer = io.BytesIO()
    Image.fromarray(plotted).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
