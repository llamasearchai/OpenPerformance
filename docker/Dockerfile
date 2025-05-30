# Base image with CUDA support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Rust for memory profiling
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi

# Build Rust components (if Rust source exists)
RUN if [ -d "rust" ]; then cd rust && cargo build --release; fi

# Application code
WORKDIR /app

# Runtime configuration
ENV PYTHONPATH=/app/python
ENV MLPERF_CACHE_DIR=/cache
ENV LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p /cache /app/logs /app/results

EXPOSE 8000
CMD ["uvicorn", "python.mlperf.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 