# OpenPerformance Production Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY python/ ./python/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 openperf && \
    chown -R openperf:openperf /app

USER openperf

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "python.mlperf.api.main:app", "--host", "0.0.0.0", "--port", "8000"]