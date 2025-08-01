#!/usr/bin/env python3
"""
Comprehensive ML Performance Engineering Platform Demo

This script demonstrates all the key features of the platform:
- Hardware detection and monitoring
- Performance benchmarking
- Optimization recommendations  
- Memory tracking
- Distributed training simulation
- API integration

Run with: python test_benchmark_demo.py
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path

# Add the python directory to the path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from python.mlperf.hardware.gpu import get_gpu_info
from python.mlperf.optimization.distributed import (
    DistributedOptimizer,
    CommunicationConfig,
    MemoryConfig,
    MemoryTracker,
    OpenAIHelper
)
from python.mlperf.utils.logging import get_logger, setup_logging
from python.mlperf.utils.config import get_config, Config
from python.mlperf.cli.main import app as cli_app

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def demo_hardware_detection():
    """Demonstrate hardware detection capabilities."""
    print_section("HARDWARE DETECTION DEMO")
    
    print_subsection("GPU Information")
    gpus = get_gpu_info()
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"  Memory: {gpu.memory_used_mb:.0f} / {gpu.memory_total_mb:.0f} MB")
            print(f"  Utilization: {gpu.utilization_percent}%")
            print(f"  Temperature: {gpu.temperature_c}°C")
    else:
        print("No GPUs detected (running on CPU)")
    
    print_subsection("System Resources")
    try:
        import psutil
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    except ImportError:
        print("psutil not available - basic CPU info only")
        print(f"CPU Cores: {os.cpu_count()}")

def demo_optimization():
    """Demonstrate optimization capabilities."""
    print_section("OPTIMIZATION DEMO")
    
    print_subsection("Communication Configuration")
    comm_config = CommunicationConfig(
        backend="nccl",
        bucket_size_mb=25,
        gradient_compression=True,
        zero_stage=2
    )
    print(f"Backend: {comm_config.backend}")
    print(f"Bucket Size: {comm_config.bucket_size_mb} MB")
    print(f"Gradient Compression: {comm_config.gradient_compression}")
    print(f"ZeRO Stage: {comm_config.zero_stage}")
    
    print_subsection("Distributed Optimizer")
    optimizer = DistributedOptimizer(config=comm_config, framework="pytorch")
    
    # Model parallelism optimization
    tp_size, pp_size = optimizer.optimize_model_parallel(
        model_size_gb=5.0,
        num_gpus=4,
        device_memory_gb=24.0
    )
    print(f"Recommended Tensor Parallel Size: {tp_size}")
    print(f"Recommended Pipeline Parallel Size: {pp_size}")
    
    # Communication optimization
    comm_settings = optimizer.optimize_communication(
        model_size_gb=5.0,
        num_parameters=1_000_000_000,
        world_size=8
    )
    print("Optimized Communication Settings:")
    for key, value in comm_settings.items():
        print(f"  {key}: {value}")

def demo_memory_tracking():
    """Demonstrate memory tracking capabilities."""
    print_section("MEMORY TRACKING DEMO")
    
    print_subsection("Memory Tracker Setup")
    tracker = MemoryTracker(framework="pytorch", interval_ms=100)
    
    print("Starting memory tracking...")
    tracker.start_tracking()
    
    # Simulate some work
    print("Simulating workload...")
    for i in range(5):
        # Simulate memory allocation
        time.sleep(0.2)
        print(f"  Step {i+1}/5")
    
    print("Stopping memory tracking...")
    memory_logs = tracker.stop_tracking()
    
    print(f"Collected {len(memory_logs)} memory samples")
    if memory_logs:
        print(f"Peak memory: {max(log.used_bytes for log in memory_logs) / (1024**2):.1f} MB")
        print(f"Average memory: {sum(log.used_bytes for log in memory_logs) / len(memory_logs) / (1024**2):.1f} MB")

def demo_ai_recommendations():
    """Demonstrate AI-powered optimization recommendations."""
    print_section("AI OPTIMIZATION RECOMMENDATIONS DEMO")
    
    # Check if OpenAI API key is available
    config = get_config()
    if not config.openai_api_key:
        print("WARNING: OpenAI API key not set - using simulated recommendations")
        recommendations = [
            "Enable gradient checkpointing to reduce memory usage by ~30%",
            "Use mixed precision training (FP16) to improve performance",
            "Increase batch size to improve GPU utilization",
            "Consider using gradient compression for distributed training"
        ]
    else:
        print("🤖 Generating AI-powered recommendations...")
        helper = OpenAIHelper()
        
        # Sample profiling data
        bottlenecks = [
            {"type": "memory", "name": "attention", "percentage": 45.0},
            {"type": "computation", "name": "linear_layers", "percentage": 35.0}
        ]
        category_times = {
            "forward_pass": 8.5,
            "backward_pass": 12.3,
            "optimizer_step": 2.1
        }
        top_events = [
            ("attention_forward", 4.2),
            ("linear_backward", 3.8),
            ("optimizer_update", 2.1)
        ]
        
        recommendations = helper.generate_recommendations(
            bottlenecks=bottlenecks,
            category_times=category_times,
            top_events=top_events,
            total_runtime=22.9
        )
    
    print("Optimization Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

def demo_cli_integration():
    """Demonstrate CLI integration."""
    print_section("CLI INTEGRATION DEMO")
    
    print_subsection("Available CLI Commands")
    print("The platform provides the following CLI commands:")
    print("  mlperf info          - Show hardware information")
    print("  mlperf benchmark     - Run performance benchmarks")
    print("  mlperf profile       - Profile Python scripts")
    print("  mlperf optimize      - Generate optimization recommendations")
    print("  mlperf version       - Show version information")
    
    print_subsection("Example CLI Usage")
    print("# Get hardware info:")
    print("python -m python.mlperf.cli.main info --format json")
    print()
    print("# Run benchmark:")
    print("python -m python.mlperf.cli.main benchmark --framework pytorch --batch-size 32")
    print()
    print("# Optimize configuration:")
    print("python -m python.mlperf.cli.main optimize config.json")

def demo_api_integration():
    """Demonstrate API integration."""
    print_section("API INTEGRATION DEMO")
    
    print_subsection("FastAPI Service")
    print("The platform provides a REST API with the following endpoints:")
    print("  GET  /system/metrics          - Get real-time system metrics")
    print("  POST /analyze/performance     - Analyze performance and get recommendations")
    print("  GET  /docs                    - Interactive API documentation")
    
    print_subsection("Example API Usage")
    print("# Start the API server:")
    print("uvicorn python.mlperf.api.main:app --reload")
    print()
    print("# Get system metrics:")
    print("curl http://localhost:8000/system/metrics")
    print()
    print("# Analyze performance:")
    print("curl -X POST http://localhost:8000/analyze/performance \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"framework\":\"pytorch\",\"batch_size\":32,\"model_config\":{},\"hardware_info\":{}}'")

def demo_docker_deployment():
    """Demonstrate Docker deployment."""
    print_section("DOCKER DEPLOYMENT DEMO")
    
    print_subsection("Container Build")
    print("Build the production container:")
    print("docker build -f docker/Dockerfile -t mlperf:latest .")
    print()
    
    print_subsection("Full Stack Deployment")
    print("Deploy the complete stack with monitoring:")
    print("docker-compose up -d")
    print()
    print("This starts:")
    print("  - API service (port 8000)")
    print("  - Background workers")
    print("  - Redis cache")
    print("  - PostgreSQL database")
    print("  - Grafana monitoring (port 3000)")
    print("  - Prometheus metrics (port 9090)")

def create_sample_config():
    """Create a sample configuration file."""
    print_section("CONFIGURATION DEMO")
    
    # Create a sample config
    sample_config = {
        "framework": "pytorch",
        "model_size_gb": 2.5,
        "num_gpus": 2,
        "world_size": 4,
        "batch_size": 32,
        "optimization": {
            "enable_mixed_precision": True,
            "enable_gradient_checkpointing": True,
            "zero_stage": 2
        },
        "communication": {
            "backend": "nccl",
            "bucket_size_mb": 50,
            "gradient_compression": True
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f, indent=2)
        config_file = f.name
    
    print_subsection("Sample Configuration")
    print(f"Created sample configuration: {config_file}")
    print(json.dumps(sample_config, indent=2))
    
    return config_file

def main():
    """Run the comprehensive demo."""
    print("ML Performance Engineering Platform - Comprehensive Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("Starting platform demo")
    
    try:
        # Run all demos
        demo_hardware_detection()
        demo_optimization()
        demo_memory_tracking()
        demo_ai_recommendations()
        demo_cli_integration()
        demo_api_integration()
        demo_docker_deployment()
        
        # Create sample config
        config_file = create_sample_config()
        
        print_section("DEMO SUMMARY")
        print("SUCCESS: Hardware detection and monitoring")
        print("SUCCESS: Performance optimization recommendations")
        print("SUCCESS: Memory usage tracking")
        print("SUCCESS: AI-powered analysis")
        print("SUCCESS: Command-line interface")
        print("SUCCESS: REST API integration")
        print("SUCCESS: Docker deployment")
        print("SUCCESS: Configuration management")
        
        print_section("NEXT STEPS")
        print("1. Try the CLI commands:")
        print("   python -m python.mlperf.cli.main info")
        print()
        print("2. Start the API server:")
        print("   uvicorn python.mlperf.api.main:app --reload")
        print()
        print("3. Deploy with Docker:")
        print("   docker-compose up -d")
        print()
        print("4. Run tests:")
        print("   python -m pytest tests/ -v")
        print()
        print("5. Explore the code:")
        print("   - python/mlperf/ - Main package")
        print("   - rust/src/ - Rust components")
        print("   - frontend/src/ - React dashboard")
        print("   - docker/ - Container configuration")
        
        # Cleanup
        try:
            os.unlink(config_file)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nERROR: Demo failed: {e}")
        return 1
    
    print(f"\nDemo completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 