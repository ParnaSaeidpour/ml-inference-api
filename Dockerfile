# Small base image keeps the final container light
FROM python:3.11-slim

WORKDIR /code

# Install dependencies first so Docker caches this layer
# and does not reinstall them every time the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Train the model at build time so the container starts fast
RUN python -m app.model

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /code
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
