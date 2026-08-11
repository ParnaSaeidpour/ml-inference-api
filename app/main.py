"""FastAPI service exposing the trained model over HTTP."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import model as model_module

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than on every request."""
    state["model"] = model_module.load()
    yield
    state.clear()


app = FastAPI(
    title="ML Inference API",
    description="Serves a trained image classifier over a REST endpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    pixels: list[float] = Field(
        ...,
        description="Flattened 8x8 grayscale image (64 values, 0-16).",
        min_length=model_module.N_FEATURES,
        max_length=model_module.N_FEATURES,
    )


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


@app.get("/health")
def health() -> dict:
    """Liveness check, used by Docker and deployment platforms."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    """Classify a single image."""
    try:
        return model_module.predict(state["model"], request.pixels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
