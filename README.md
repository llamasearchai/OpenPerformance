# OpenPerformance

A comprehensive ML Performance Engineering Platform for optimizing and monitoring machine learning workloads.

[![CI](https://github.com/openperformance/openperformance/actions/workflows/ci.yml/badge.svg)](https://github.com/openperformance/openperformance/actions/workflows/ci.yml)
[![Release](https://github.com/openperformance/openperformance/actions/workflows/release.yml/badge.svg)](https://github.com/openperformance/openperformance/releases)
[![Docs](https://github.com/openperformance/openperformance/actions/workflows/pages.yml/badge.svg)](https://github.com/openperformance/openperformance/actions/workflows/pages.yml)
[![PyPI](https://img.shields.io/pypi/v/openperformance.svg)](https://pypi.org/project/openperformance/)
[![Python](https://img.shields.io/pypi/pyversions/openperformance.svg)](https://pypi.org/project/openperformance/)

## Features

- **Hardware Monitoring**: Real-time CPU, memory, and GPU monitoring
- **Performance Analysis**: AI-powered optimization recommendations
- **Distributed Training**: Advanced distributed optimization algorithms
- **CLI Interface**: Comprehensive command-line tools
- **REST API**: Full-featured API with authentication
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Production Ready**: Continuous CI with tests, type checks, security scans

## Quick Start

### Installation

```bash
# Install from PyPI
pip install openperformance

# Or install from source
git clone https://github.com/openperformance/openperformance.git
cd openperformance
pip install -e .
```

### Basic Usage

```bash
# Check system information
mlperf info

# Run performance analysis
mlperf optimize --framework pytorch --batch-size 32

# Start API server
openperf-server
# or explicitly via uvicorn with installed package path:
# python -m uvicorn mlperf.api.main:app --host 0.0.0.0 --port 8000
```

## CLI Commands

```bash
mlperf --help                    # Show all available commands
mlperf info                      # Display system hardware information
mlperf version                   # Show platform version
mlperf benchmark                 # Run performance benchmarks
mlperf profile                   # Profile Python scripts
mlperf optimize                  # Optimize ML workloads
mlperf gpt                       # AI-powered shell assistance
mlperf chat                      # Chat with ML performance AI agents
```

## API Endpoints

- `GET /health` - System health check
- `GET /system/metrics` - Real-time system metrics
- `GET /system/hardware` - Detailed hardware information
- `POST /analyze/performance` - Performance analysis and optimization
- `GET /admin/system/status` - Admin system status

## Hardware Monitoring

The platform provides comprehensive hardware monitoring:

- **CPU**: Core count, frequency, usage percentage
- **Memory**: Total, used, available memory with usage statistics
- **GPU**: NVIDIA GPU detection, memory usage, utilization metrics
- **System**: Architecture, platform information

## Performance Analysis

Get AI-powered optimization recommendations for your ML workloads:

```python
from mlperf.optimization.distributed import DistributedOptimizer

# Initialize optimizer
optimizer = DistributedOptimizer(framework="pytorch")

# Get optimization recommendations
recommendations = optimizer.optimize_model_parallel(
    model_size_gb=10.0,
    gpu_count=4
)
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenPerformance.git
cd OpenPerformance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r dev-requirements.txt
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests python/tests --cov=mlperf --cov-report=html --cov-report=term-missing

# Run specific test categories
python -m pytest tests/test_hardware.py -v
python -m pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Linting
black --check python/
ruff check python/ tests/ python/tests/

# Type checking
mypy python/mlperf/

# Security checks
bandit -r python/mlperf
pip-audit -e .
```

## Docker

### Build and Run

```bash
# Build Docker image
docker build -t openperformance .

# Run container
docker run -p 8000:8000 openperformance

# Run with Docker Compose
docker-compose up -d
```

### Docker Compose Services

- **API Server**: FastAPI application on port 8000
- **Database**: PostgreSQL for data persistence
- **Redis**: Caching and rate limiting
- **Monitoring**: Prometheus and Grafana

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/openperformance

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# API Keys
OPENAI_API_KEY=your-openai-api-key

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/openperformance.log
```

### Configuration Files

- `config.env` - Environment configuration
- `alembic.ini` - Database migration configuration
- `pyproject.toml` - Project metadata and dependencies

## Architecture

```
OpenPerformance/
├── python/mlperf/           # Main Python package
│   ├── api/                # FastAPI application
│   ├── auth/               # Authentication and security
│   ├── cli/                # Command-line interface
│   ├── hardware/           # Hardware monitoring
│   ├── optimization/       # Performance optimization
│   ├── utils/              # Utilities and helpers
│   └── workers/            # Background workers
├── tests/                  # Test suite
├── docker/                 # Docker configuration
├── k8s/                    # Kubernetes manifests
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Ensure all tests pass
- Add type hints where appropriate

## Testing

The platform includes comprehensive testing:

- **Unit Tests**: 43 tests covering all core functionality
- **Integration Tests**: Full workflow testing
- **Performance Tests**: Benchmarking and profiling
- **Security Tests**: Vulnerability scanning

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=python/mlperf --cov-report=html

# Run performance benchmarks
python -m pytest tests/performance/ -v
```

## Security

- JWT-based authentication
- Role-based access control
- Rate limiting
- Input validation
- Secure password hashing
- CORS protection

## Performance

- Optimized for production workloads
- Efficient memory usage
- Fast API response times
- Scalable architecture
- Real-time monitoring

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: Site (GitHub Pages) coming from MkDocs
- **Issues**: [GitHub Issues](https://github.com/openperformance/openperformance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/openperformance/openperformance/discussions)

## Acknowledgments

- Built with FastAPI, PyTorch, and modern Python tooling
- Inspired by MLPerf and other performance engineering tools
- Community contributions welcome

## Roadmap

- [ ] Web dashboard
- [ ] Advanced analytics
- [ ] Cloud integration
- [ ] Real-time monitoring
- [ ] Additional ML frameworks
- [ ] Performance benchmarking suite 