# syntax=docker/dockerfile:1
#
# Light remote-host image: the LiveKit worker (sherpa-onnx STT/TTS + the
# always_on_agent brain) and the FastAPI token/web server. The LLM runs in your
# host-local Ollama -- it is NOT in this image. Single stage + lean deps (no
# torch) so it builds fast and the deps layer caches across code edits.

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app
WORKDIR /app

# Deps first so editing app code never re-triggers a reinstall (fast rebuilds).
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm -f /tmp/requirements-docker.txt

# App code only (see .dockerignore for what is excluded).
COPY always_on_agent/ always_on_agent/
COPY core/ core/
# utils/ holds the Postgres-backed smart-memory engine. always_on_agent.memory
# lazily imports utils.memory only when DATABASE_URL is set (in-RAM default needs
# neither this code nor psycopg), so copying it keeps that path resolvable; with
# DATABASE_URL unset it is never imported. See docker/requirements-docker.txt for
# the optional psycopg line that the real Postgres backend needs.
COPY utils/ utils/
COPY remote/ remote/
COPY tools/ tools/
COPY web/ web/
COPY config.json ./
COPY docker/entrypoint.py docker/download_models.py /usr/local/bin/

# Fail the build early if the lean dependency set can't import the entrypoints.
RUN python -c "import core.app, remote.worker, remote.token_server"

# Run unprivileged. Code stays root-owned (read-only to the app), and the
# entrypoint writes its merged config to a tmpfs, so this image runs cleanly
# under `read_only: true` + a non-root user.
RUN useradd --system --uid 10001 --user-group app
USER app

ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
CMD ["python", "-m", "remote.worker"]
