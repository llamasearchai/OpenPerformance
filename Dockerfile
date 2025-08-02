# syntax=docker/dockerfile:1.7

# =========================
# Builder image
# =========================
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install build tooling and pin docs tools to avoid surprises
RUN python -m pip install --upgrade pip setuptools wheel

# Copy metadata and sources first for better layer caching
COPY pyproject.toml ./
COPY python/ ./python/

# Install runtime deps only (editable disabled in builder to produce wheel metadata)
RUN python -m pip install -e ".[dist]" --config-settings editable_mode=compat || \
    python -m pip install -e .

# =========================
# Runtime image
# =========================
FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_WORKERS=2

WORKDIR /app

# Install only minimal OS deps
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy source for runtime (if needed for templates/static); prefer installing package
COPY pyproject.toml ./
COPY python/ ./python/

# Install app (runtime deps only)
RUN python -m pip install --upgrade pip && \
    pip install -e .

# Create non-root user
RUN useradd -m -u 1000 openperf && chown -R openperf:openperf /app
USER openperf

EXPOSE 8000

# Default command: use console script entry point fixed to mlperf.api.main
# Prefer the package script we defined: openperf-server
CMD ["openperf-server", "--host", "0.0.0.0", "--port", "8000"]