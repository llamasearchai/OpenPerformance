# OpenPerformance Documentation

Welcome to the OpenPerformance ML Performance Engineering Platform documentation.

## Overview

OpenPerformance is a comprehensive platform for optimizing and monitoring machine learning workloads with:

- **Hardware Monitoring**: Real-time CPU, memory, and GPU monitoring
- **Performance Analysis**: AI-powered optimization recommendations  
- **Distributed Training**: Advanced distributed optimization algorithms
- **CLI Interface**: Comprehensive command-line tools
- **REST API**: Full-featured API with authentication
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Quick Start

### Installation

```bash
pip install openperformance
```

### Basic Usage

```bash
# Check system information
mlperf info

# Run performance analysis
mlperf optimize --framework pytorch --batch-size 32

# Start API server
python -m uvicorn python.mlperf.api.main:app --host 0.0.0.0 --port 8000
```

## API Reference

### Hardware Monitoring

Get comprehensive hardware information:

```python
from mlperf.hardware import get_cpu_info, get_memory_info, get_gpu_info

# Get CPU information
cpu_info = get_cpu_info()
print(f"CPU cores: {cpu_info.physical_cores}")

# Get memory information  
memory_info = get_memory_info()
print(f"Total memory: {memory_info.total_gb:.1f} GB")

# Get GPU information
gpu_info = get_gpu_info()
print(f"GPUs detected: {len(gpu_info)}")
```

### Performance Optimization

Get AI-powered optimization recommendations:

```python
from mlperf.optimization.distributed import DistributedOptimizer

# Initialize optimizer
optimizer = DistributedOptimizer(framework="pytorch")

# Get optimization recommendations
recommendations = optimizer.optimize_model_parallel(
    model_size_gb=10.0,
    gpu_count=4
)

for rec in recommendations:
    print(f"{rec.area}: {rec.suggestion}")
```

## CLI Commands

- `mlperf info` - Display system hardware information
- `mlperf benchmark` - Run performance benchmarks
- `mlperf optimize` - Optimize ML workloads
- `mlperf version` - Show version information

## Contributing

See the main [README](https://github.com/llamasearchai/OpenPerformance) for contribution guidelines.

## License

MIT License - see [LICENSE](https://github.com/llamasearchai/OpenPerformance/blob/main/LICENSE) for details.