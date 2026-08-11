# ML Inference API

A small machine learning service that shows the full path from a trained model to a
running, tested, containerised API. Built to demonstrate deployment practice, not
model complexity.

![CI](https://github.com/ParnaSaeidpour/ml-inference-api/actions/workflows/ci.yml/badge.svg)

## What it does

An image classifier is trained on the scikit-learn digits dataset (8x8 grayscale
images, 10 classes) and served over HTTP with FastAPI. The service is packaged in a
Docker container, and every push to `main` automatically runs the test suite and
builds the image through GitHub Actions.

Test accuracy: **0.96** (Random Forest, 100 trees, 20% held-out test set).

## Why this project

Training a model is only part of the work. This repository covers the part that is
usually missing from student projects:

- serving a model behind a documented REST API
- loading the model once at startup rather than per request
- input validation with clear error responses
- automated tests, including a guard against silent accuracy regressions
- a reproducible container that runs the same way anywhere
- continuous integration that blocks a broken build

## Architecture

```
app/model.py   training, loading and inference logic
app/main.py    FastAPI application and endpoints
tests/         pytest suite, run automatically on every push
Dockerfile     container definition
.github/       GitHub Actions CI pipeline
```

The API layer knows nothing about the model beyond `predict()` and
`predict_proba()`, so a different model can be swapped in without touching
`main.py`.

## Endpoints

| Method | Path       | Purpose                          |
|--------|------------|----------------------------------|
| GET    | `/health`  | Liveness check for orchestrators |
| POST   | `/predict` | Classify one image               |
| GET    | `/docs`    | Interactive OpenAPI docs         |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,5,13,9,1,0,0, 0,0,13,15,10,15,5,0, 0,3,15,2,0,11,8,0,
                  0,4,12,0,0,8,8,0, 0,5,8,0,0,9,8,0, 0,4,11,0,1,12,7,0,
                  0,2,14,5,10,12,0,0, 0,0,6,13,10,0,0,0]}'
```

```json
{ "prediction": 0, "confidence": 0.99 }
```

## Running it

**Locally**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**With Docker**

```bash
docker build -t ml-inference-api .
docker run -p 8000:8000 ml-inference-api
```

Then open http://localhost:8000/docs

**Tests**

```bash
pytest -v
```

## CI pipeline

Defined in `.github/workflows/ci.yml`. On every push and pull request to `main`:

1. **test** — installs dependencies and runs the pytest suite
2. **docker** — builds the image and smoke-tests the running container, and only
   runs if the tests passed

A failing test stops the build, so a broken model or endpoint never reaches an
image.

## Notes on the container

- `python:3.11-slim` base image keeps the image small
- dependencies are installed before the code is copied, so Docker caches that layer
- the model is trained at build time, so the container starts immediately
- the process runs as a non-root user
- a `HEALTHCHECK` lets Docker and deployment platforms detect an unhealthy container

## Adapting this to another model

Replace the training logic in `app/model.py`, update `N_FEATURES`, and adjust the
request schema in `app/main.py`. Everything else stays the same.

## Author

Parna Saeidpour — AI Engineer, Jönköping, Sweden
[LinkedIn](https://www.linkedin.com/in/parna-saeidpour-185a4652/) ·
[GitHub](https://github.com/ParnaSaeidpour)
