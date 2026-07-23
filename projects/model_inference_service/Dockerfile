FROM python:3.11-slim

# Unbuffered logs, no .pyc, and the model location the API reads.
# Override MODEL_PATH at run time (-e MODEL_PATH=...) to point at your mounted model.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/distilbart-mnli \
    MODEL_TASK=zero-shot-classification \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (model-agnostic; you don't rebuild to swap models).
COPY model_wrapper.py app.py client.py test_smoke.py ./

# Mount point for the model you provide as a file location. Kept empty in the
# image — bind-mount your model dir here (see docker-compose.yml), OR bake a
# model in by copying it before build and pointing MODEL_PATH at it.
RUN mkdir -p /models

EXPOSE 8000

# One worker = one copy of the model in RAM. Scale with more containers behind a
# load balancer once you know the memory footprint.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
