"""FastAPI service exposing the steel surface defect detector over HTTP."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app import model as model_module

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("defect-api")

API_VERSION = "2.0.0"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than on every request."""
    state["model"] = model_module.load()
    state["metrics"] = model_module.load_metrics()
    logger.info("model loaded from %s", model_module.MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Steel Defect Detection API",
    description=(
        "Detects six types of surface defects on hot-rolled steel "
        "(crazing, inclusion, patches, pitted surface, rolled-in scale, "
        "scratches) using a fine-tuned YOLO26n model."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: Box


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    defect_count: int
    detections: list[Detection]


ImageFile = Annotated[UploadFile, File(description="Image of a steel surface")]
Confidence = Annotated[float, Query(ge=0.01, le=1.0, description="Minimum confidence")]


def read_image(file: UploadFile):
    try:
        return model_module.decode_image(
            file.file.read(model_module.MAX_IMAGE_BYTES + 1)
        )
    except model_module.InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    """Liveness check, used by Docker and deployment platforms."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model")
def model_info() -> dict:
    """Which model is being served, its classes, and its test-set metrics."""
    return {
        "api_version": API_VERSION,
        "weights": model_module.MODEL_PATH.name,
        "classes": list(state["model"].names.values()),
        "metrics": state["metrics"],
    }


# Inference endpoints are plain `def`: FastAPI runs them in a worker thread,
# so a slow prediction does not block the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: ImageFile,
    conf: Confidence = 0.25,
) -> dict:
    """Detect defects in one image and return them as JSON."""
    image = read_image(file)
    detections = model_module.detect(state["model"], image, conf=conf)
    return {
        "image_width": image.width,
        "image_height": image.height,
        "defect_count": len(detections),
        "detections": detections,
    }


@app.post(
    "/predict/image",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def predict_image(
    file: ImageFile,
    conf: Confidence = 0.25,
) -> Response:
    """Detect defects and return the image with boxes drawn on it."""
    image = read_image(file)
    jpeg = model_module.annotate(state["model"], image, conf=conf)
    return Response(content=jpeg, media_type="image/jpeg")
