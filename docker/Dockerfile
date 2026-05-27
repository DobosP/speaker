# syntax=docker/dockerfile:1
#
# Remote-host image for the Speaker voice assistant: runs the LiveKit worker
# (sherpa-onnx STT/TTS + the always_on_agent brain) and the FastAPI token/web
# server. The LLM runs in the separate `ollama` GPU service; this image is
# CPU-only and intentionally lean (no torch).

# ---- build stage: resolve wheels (throwaway; keeps the runtime image small) ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY docker/requirements-docker.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements-docker.txt

# ---- runtime stage ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app
WORKDIR /app

COPY --from=build /wheels /wheels
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /tmp/requirements-docker.txt \
    && rm -rf /wheels /tmp/requirements-docker.txt

# App code only (see .dockerignore for what is excluded).
COPY always_on_agent/ always_on_agent/
COPY core/ core/
COPY remote/ remote/
COPY tools/ tools/
COPY web/ web/
COPY config.json ./
COPY docker/entrypoint.py docker/download_models.py /usr/local/bin/

# Fail the build early if the lean dependency set can't import the entrypoints.
RUN python -c "import core.app, remote.worker, remote.token_server"

ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
CMD ["python", "-m", "remote.worker"]
