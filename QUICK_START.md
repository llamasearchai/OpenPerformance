# ML Performance Platform - Quick Start

Welcome to the ML Performance Engineering Platform! This guide will get you up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Optional: NVIDIA GPU for advanced profiling

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/llamasearchai/OpenPerformance.git
cd OpenPerformance
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r dev-requirements.txt
```

### 3. Configure Environment

```bash
# Copy configuration template
cp config.env.example config.env

# Edit configuration (optional)
nano config.env
```

## Quick Start

### Start the Platform

```bash
# Start all services
python start_platform.py

# Or use the convenience script
./scripts/setup.sh
```

### Access the Web Interface

Open your browser to `http://localhost:8000`

### Run Your First Benchmark

```bash
# Run a simple training benchmark
python -m mlperf.cli benchmark --model resnet50 --framework pytorch

# View results
python -m mlperf.cli results --latest
```

## Key Features Ready to Use

**Hardware Detection** - GPU, CPU, memory monitoring
**Performance Optimization** - Distributed training recommendations
**Memory Tracking** - Real-time usage analysis
**AI Recommendations** - OpenAI-powered suggestions
**REST API** - Production-ready endpoints
**CLI Tools** - Developer-friendly commands
**Docker Deployment** - Scalable container architecture
**Real-time Monitoring** - Grafana dashboards

## Example Usage

### Python API

```python
from mlperf.optimization import DistributedOptimizer
from mlperf.hardware import get_gpu_info

# Initialize optimizer
optimizer = DistributedOptimizer()

# Get hardware information
gpus = get_gpu_info()
print(f"Detected {len(gpus)} GPUs")

# Optimize for your model
config = optimizer.optimize_model_parallel(
    model_size_gb=7.5,
    num_gpus=len(gpus),
    device_memory_gb=gpus[0].memory_gb if gpus else 0
)
```

### REST API

```bash
# Get system metrics
curl http://localhost:8000/system/metrics

# Start a benchmark
curl -X POST http://localhost:8000/benchmarks \
  -H "Content-Type: application/json" \
  -d '{"model": "bert-base", "framework": "pytorch"}'
```

## Next Steps

1. **Explore the Web Interface** - Navigate to http://localhost:8000
2. **Read the Documentation** - Check out [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Run Benchmarks** - Try different models and frameworks
4. **Configure for Production** - See [ENTERPRISE_DEPLOYMENT_GUIDE.md](ENTERPRISE_DEPLOYMENT_GUIDE.md)

**Platform is production-ready with comprehensive testing and all core features operational!** 