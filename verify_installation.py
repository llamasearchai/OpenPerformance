#!/usr/bin/env python3
"""
Verification script for ML Performance Engineering Platform.

This script tests that all components are working correctly.
"""

import sys
import traceback
import subprocess
from pathlib import Path

def print_success(message):
    """Print success message in green."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message in red."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message in yellow."""
    print(f"⚠️  {message}")

def print_info(message):
    """Print info message."""
    print(f"ℹ️  {message}")

def test_imports():
    """Test that all modules can be imported."""
    print_info("Testing module imports...")
    
    try:
        # Test core package import
        import python.mlperf
        print_success("Core package imported successfully")
        
        # Test hardware module
        from python.mlperf.hardware import get_gpu_info, GPUInfo
        print_success("Hardware module imported successfully")
        
        # Test utils module
        from python.mlperf.utils import get_logger, Config
        print_success("Utils module imported successfully")
        
        # Test optimization module
        from python.mlperf.optimization import DistributedOptimizer, CommunicationConfig
        print_success("Optimization module imported successfully")
        
        # Test CLI module
        from python.mlperf.cli import app
        print_success("CLI module imported successfully")
        
        # Test API module
        from python.mlperf.api import app as api_app
        print_success("API module imported successfully")
        
        # Test workers module
        from python.mlperf.workers import BenchmarkWorker
        print_success("Workers module imported successfully")
        
        return True
        
    except Exception as e:
        print_error(f"Import failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_detection():
    """Test GPU detection functionality."""
    print_info("Testing GPU detection...")
    
    try:
        from python.mlperf.hardware import get_gpu_info
        
        gpus = get_gpu_info()
        
        if gpus:
            print_success(f"Detected {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print_info(f"  GPU {i}: {gpu.name} ({gpu.memory_total_mb} MB)")
        else:
            print_warning("No GPUs detected (this is normal if running on CPU-only systems)")
        
        return True
        
    except Exception as e:
        print_error(f"GPU detection failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print_info("Testing configuration...")
    
    try:
        from python.mlperf.utils import Config
        
        config = Config()
        print_success("Configuration loaded successfully")
        
        # Test OpenAI API key loading
        from python.mlperf.utils.config import get_openai_api_key
        api_key = get_openai_api_key()
        
        if api_key:
            print_success("OpenAI API key found")
        else:
            print_warning("OpenAI API key not found (set OPENAI_API_KEY environment variable for AI features)")
        
        return True
        
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_optimization():
    """Test optimization functionality."""
    print_info("Testing optimization functionality...")
    
    try:
        from python.mlperf.optimization import DistributedOptimizer, CommunicationConfig
        
        # Test basic optimizer creation
        config = CommunicationConfig()
        optimizer = DistributedOptimizer(config=config, framework="pytorch")
        print_success("Distributed optimizer created successfully")
        
        # Test model parallelism optimization
        tp_size, pp_size = optimizer.optimize_model_parallel(
            model_size_gb=1.0,
            num_gpus=1,
            device_memory_gb=24.0
        )
        print_success(f"Model parallelism optimization: TP={tp_size}, PP={pp_size}")
        
        # Test communication optimization
        comm_settings = optimizer.optimize_communication(
            model_size_gb=1.0,
            num_parameters=1000000,
            world_size=1
        )
        print_success("Communication optimization completed")
        
        return True
        
    except Exception as e:
        print_error(f"Optimization test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_tracking():
    """Test memory tracking functionality."""
    print_info("Testing memory tracking...")
    
    try:
        from python.mlperf.optimization import MemoryTracker
        
        # Create memory tracker
        tracker = MemoryTracker(framework="pytorch")
        print_success("Memory tracker created successfully")
        
        # Test getting memory usage (without actually starting tracking)
        try:
            memory_usage = tracker._get_memory_usage()
            print_success("Memory usage detection working")
        except Exception as e:
            print_warning(f"Memory usage detection failed (this is normal without PyTorch/GPUs): {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Memory tracking test failed: {e}")
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI functionality."""
    print_info("Testing CLI...")
    
    try:
        # Test CLI help command
        result = subprocess.run(
            [sys.executable, "-m", "python.mlperf.cli.main", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("CLI help command works")
        else:
            print_error(f"CLI help command failed: {result.stderr}")
            return False
        
        # Test version command
        result = subprocess.run(
            [sys.executable, "-m", "python.mlperf.cli.main", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("CLI version command works")
            print_info(f"  Output: {result.stdout.strip()}")
        else:
            print_warning(f"CLI version command failed: {result.stderr}")
        
        return True
        
    except Exception as e:
        print_error(f"CLI test failed: {e}")
        traceback.print_exc()
        return False

def test_api_import():
    """Test that the API can be imported."""
    print_info("Testing API import...")
    
    try:
        from python.mlperf.api.main import app
        print_success("FastAPI app imported successfully")
        
        # Test that routes are registered
        routes = [route.path for route in app.routes]
        print_success(f"API has {len(routes)} routes registered")
        
        return True
        
    except Exception as e:
        print_error(f"API import test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print_info("Testing dependencies...")
    
    required_packages = [
        "numpy",
        "typer",
        "rich",
        "fastapi",
        "uvicorn",
        "redis",
        "celery",
        "openai"
    ]
    
    optional_packages = [
        "torch",
        "tensorflow",
        "jax",
        "psutil",
        "pynvml"
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Required package '{package}' is available")
        except ImportError:
            missing_required.append(package)
            print_error(f"Required package '{package}' is missing")
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"Optional package '{package}' is available")
        except ImportError:
            missing_optional.append(package)
            print_warning(f"Optional package '{package}' is missing")
    
    if missing_required:
        print_error(f"Missing required packages: {', '.join(missing_required)}")
        print_info("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print_warning(f"Missing optional packages: {', '.join(missing_optional)}")
        print_info("Some features may not be available")
    
    return True

def test_project_structure():
    """Test that project structure is correct."""
    print_info("Testing project structure...")
    
    required_files = [
        "pyproject.toml",
        "docker-compose.yml",
        "python/mlperf/__init__.py",
        "python/mlperf/api/__init__.py",
        "python/mlperf/cli/__init__.py",
        "python/mlperf/hardware/__init__.py",
        "python/mlperf/optimization/__init__.py",
        "python/mlperf/utils/__init__.py",
        "python/mlperf/workers/__init__.py",
        "tests/test_distributed.py",
        "tests/test_hardware.py",
        "docker/Dockerfile"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"Found {file_path}")
        else:
            missing_files.append(file_path)
            print_error(f"Missing {file_path}")
    
    if missing_files:
        print_error(f"Missing {len(missing_files)} required files")
        return False
    
    print_success("Project structure is correct")
    return True

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("ML Performance Engineering Platform - Installation Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("GPU Detection", test_gpu_detection),
        ("Optimization", test_optimization),
        ("Memory Tracking", test_memory_tracking),
        ("CLI", test_cli),
        ("API Import", test_api_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print_success("🎉 All tests passed! The installation is working correctly.")
        print_info("You can now:")
        print_info("  - Run CLI commands: python -m python.mlperf.cli.main --help")
        print_info("  - Start the API server: uvicorn python.mlperf.api.main:app --reload")
        print_info("  - Run the full stack: docker-compose up")
        return 0
    else:
        print_error(f"❌ {total - passed} tests failed. Please check the errors above.")
        print_info("Common issues:")
        print_info("  - Missing dependencies (install with: poetry install)")
        print_info("  - GPU libraries not available (install PyTorch, TensorFlow, etc.)")
        print_info("  - OpenAI API key not set (export OPENAI_API_KEY=your_key)")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 