#!/usr/bin/env python3
"""
ML Performance Engineering Platform - Quick Start Script

This script helps you get started with the platform quickly by:
1. Checking dependencies
2. Setting up the environment
3. Running the demo
4. Starting services

Usage:
    python start_platform.py --help
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List

def print_banner():
    """Print the platform banner."""
    banner = """
🚀 ML Performance Engineering Platform
========================================
Complete platform for ML optimization and monitoring
Features: GPU monitoring, distributed optimization, AI recommendations
"""
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "fastapi", "uvicorn", "typer", "rich", "psutil", "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_optional_dependencies():
    """Check optional dependencies and report status."""
    optional_packages = {
        "torch": "PyTorch (for neural network optimization)",
        "tensorflow": "TensorFlow (for ML framework integration)",
        "openai": "OpenAI (for AI-powered recommendations)",
        "pynvml": "NVIDIA ML (for GPU monitoring)",
        "docker": "Docker (for containerized deployment)"
    }
    
    print("\n📋 Optional Dependencies:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️  {package} - {description} (optional)")

def setup_environment():
    """Set up the environment variables and directories."""
    print("\n🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ["cache", "logs", "outputs", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Set environment variables
    env_vars = {
        "PYTHONPATH": str(Path.cwd() / "python"),
        "LOG_LEVEL": "INFO",
        "MLPERF_CACHE_DIR": "./cache",
        "MLPERF_OUTPUT_DIR": "./outputs"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")

def run_verification():
    """Run the verification script."""
    print("\n🔍 Running verification...")
    try:
        result = subprocess.run([
            sys.executable, "verify_installation.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Verification passed")
            return True
        else:
            print("❌ Verification failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Verification timed out")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def run_demo():
    """Run the comprehensive demo."""
    print("\n🎬 Running comprehensive demo...")
    try:
        result = subprocess.run([
            sys.executable, "test_benchmark_demo.py"
        ], timeout=60)
        
        if result.returncode == 0:
            print("✅ Demo completed successfully")
            return True
        else:
            print("❌ Demo failed")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def start_api_server(port: int = 8000, reload: bool = True):
    """Start the API server."""
    print(f"\n🌐 Starting API server on port {port}...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "python.mlperf.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        print(f"🚀 API server starting at http://localhost:{port}")
        print("📚 API documentation at http://localhost:{port}/docs")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 API server stopped")

def start_docker_services():
    """Start Docker services."""
    print("\n🐳 Starting Docker services...")
    
    if not subprocess.run(["docker", "--version"], capture_output=True).returncode == 0:
        print("❌ Docker not available")
        return False
    
    try:
        # Build and start services
        subprocess.run(["docker-compose", "up", "--build", "-d"], check=True)
        print("✅ Docker services started")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(10)
        
        print("\n🎯 Services available at:")
        print("  - API: http://localhost:8000")
        print("  - API Docs: http://localhost:8000/docs")
        print("  - Grafana: http://localhost:3000 (admin/admin123)")
        print("  - Prometheus: http://localhost:9090")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker services failed: {e}")
        return False

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], timeout=120)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️ Some tests failed")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    examples = """
🎯 Quick Usage Examples:

1. CLI Commands:
   python -m python.mlperf.cli.main info
   python -m python.mlperf.cli.main benchmark --framework pytorch
   python -m python.mlperf.cli.main profile script.py

2. API Usage:
   curl http://localhost:8000/system/metrics
   curl -X POST http://localhost:8000/analyze/performance -d '{...}'

3. Docker Deployment:
   docker-compose up -d

4. Run Tests:
   python -m pytest tests/ -v

5. Environment Setup:
   cp config.env.example .env
   # Edit .env with your configuration
"""
    print(examples)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Performance Engineering Platform Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["check", "demo", "api", "docker", "test", "full"],
        default="full",
        help="What to run (default: full)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for API server"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Always check dependencies first
    check_python_version()
    setup_environment()
    
    success = True
    
    if args.mode in ["check", "full"]:
        if not check_dependencies():
            print("\n❌ Please install missing dependencies first")
            sys.exit(1)
        check_optional_dependencies()
    
    if args.mode in ["demo", "full"]:
        if not run_verification():
            print("⚠️ Verification failed, but continuing...")
        
        if not run_demo():
            print("⚠️ Demo failed, but continuing...")
            success = False
    
    if args.mode in ["test", "full"] and not args.skip_tests:
        if not run_tests():
            print("⚠️ Some tests failed, but continuing...")
            success = False
    
    if args.mode == "api":
        start_api_server(port=args.port, reload=not args.no_reload)
    
    elif args.mode == "docker":
        if not start_docker_services():
            success = False
    
    elif args.mode == "full":
        print("\n🎉 Platform setup complete!")
        
        if success:
            print("✅ All components working correctly")
        else:
            print("⚠️ Some components had issues (check logs)")
        
        show_usage_examples()
        
        # Ask user what to do next
        print("\n❓ What would you like to do next?")
        print("1. Start API server")
        print("2. Start Docker services")
        print("3. Show usage examples")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                start_api_server(port=args.port, reload=not args.no_reload)
            elif choice == "2":
                start_docker_services()
            elif choice == "3":
                show_usage_examples()
            elif choice == "4":
                print("👋 Thanks for using ML Performance Engineering Platform!")
            else:
                print("Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 