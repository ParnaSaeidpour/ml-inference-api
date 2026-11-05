# Small base image keeps the final container light
FROM python:3.11-slim

# OpenCV (used by Ultralytics) needs these system libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# CPU-only PyTorch: the default build ships ~2 GB of CUDA libraries we don't use
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.1 \
      --index-url https://download.pytorch.org/whl/cpu

# Install dependencies before copying the code so Docker caches this layer
# and does not reinstall them every time the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/defect_yolo26n.pt models/metrics.json ./models/

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /code
USER appuser

# Keep Ultralytics offline at runtime: no update checks, analytics or
# automatic pip installs triggered by requests
ENV YOLO_OFFLINE=1 \
    YOLO_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
