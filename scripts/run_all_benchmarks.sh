#!/bin/bash
set -e

mkdir -p results/training-benchmarks
mkdir -p results/inference-benchmarks
mkdir -p results/distributed-benchmarks

# Run training benchmarks
pytest tests/performance/test_training_benchmarks.py --benchmark-only --benchmark-json=results/training-benchmarks/training-benchmarks.json

# Run inference benchmarks
pytest tests/performance/test_inference_benchmarks.py --benchmark-only --benchmark-json=results/inference-benchmarks/inference-benchmarks.json

# Run distributed benchmarks
pytest tests/performance/test_distributed_benchmarks.py --benchmark-only --benchmark-json=results/distributed-benchmarks/distributed-benchmarks.json

echo "All benchmarks completed. Results saved in results/ directory." 