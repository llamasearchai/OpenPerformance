#!/usr/bin/env python3
"""
OpenPerformance Platform Verification Script
Tests all major components to ensure the platform is fully functional.
"""

import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Testing: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"PASS: {description}")
            return True
        else:
            print(f"FAIL: {description}")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"ERROR: {description} - {e}")
        return False

def test_cli_commands():
    """Test all CLI commands."""
    print("\n" + "="*60)
    print("TESTING CLI COMMANDS")
    print("="*60)
    
    tests = [
        ("mlperf --help", "CLI help command"),
        ("mlperf version", "CLI version command"),
        ("mlperf info", "CLI hardware info command"),
    ]
    
    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
    
    return passed, len(tests)

def test_python_imports():
    """Test all major Python module imports."""
    print("\n" + "="*60)
    print("TESTING PYTHON IMPORTS")
    print("="*60)
    
    modules = [
        "python.mlperf",
        "python.mlperf.hardware.cpu",
        "python.mlperf.hardware.memory", 
        "python.mlperf.hardware.gpu",
        "python.mlperf.api.main",
        "python.mlperf.auth.jwt",
        "python.mlperf.optimization.distributed",
        "python.mlperf.utils.config",
        "python.mlperf.utils.logging",
    ]
    
    passed = 0
    for module in modules:
        try:
            __import__(module)
            print(f"PASS: Import {module}")
            passed += 1
        except ImportError as e:
            print(f"FAIL: Import {module} - {e}")
        except Exception as e:
            print(f"ERROR: Import {module} - {e}")
    
    return passed, len(modules)

def test_hardware_modules():
    """Test hardware monitoring functionality."""
    print("\n" + "="*60)
    print("TESTING HARDWARE MODULES")
    print("="*60)
    
    test_code = """
import sys
sys.path.insert(0, '.')

from python.mlperf.hardware.cpu import get_cpu_info
from python.mlperf.hardware.memory import get_memory_info
from python.mlperf.hardware.gpu import get_gpu_info

# Test CPU info
cpu_info = get_cpu_info()
print(f"CPU: {cpu_info.physical_cores} cores, {cpu_info.frequency_mhz} MHz")

# Test memory info
memory_info = get_memory_info()
print(f"Memory: {memory_info.total_gb:.1f} GB total, {memory_info.usage_percent:.1f}% used")

# Test GPU info
gpu_info = get_gpu_info()
print(f"GPUs: {len(gpu_info)} detected")

print("Hardware modules working correctly")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("PASS: Hardware modules")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"FAIL: Hardware modules - {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Hardware modules - {e}")
        return False

def test_api_server():
    """Test API server functionality."""
    print("\n" + "="*60)
    print("TESTING API SERVER")
    print("="*60)
    
    # Start server in background
    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "python.mlperf.api.main:app", 
             "--host", "127.0.0.1", "--port", "8001"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        import requests
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if response.status_code == 200:
                print("PASS: API server health endpoint")
                health_data = response.json()
                print(f"   Status: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"FAIL: API server health endpoint - Status {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: API server health endpoint - {e}")
            return False
        finally:
            server_process.terminate()
            server_process.wait()
            
    except Exception as e:
        print(f"ERROR: API server - {e}")
        return False

def test_pytest():
    """Run pytest to verify all tests pass."""
    print("\n" + "="*60)
    print("TESTING PYTEST SUITE")
    print("="*60)
    
    return run_command("python -m pytest tests/ -q", "Pytest test suite")

def main():
    """Run all verification tests."""
    print("OpenPerformance Platform Verification")
    print("="*60)
    
    results = []
    
    # Test CLI commands
    passed, total = test_cli_commands()
    results.append(("CLI Commands", passed, total))
    
    # Test Python imports
    passed, total = test_python_imports()
    results.append(("Python Imports", passed, total))
    
    # Test hardware modules
    if test_hardware_modules():
        results.append(("Hardware Modules", 1, 1))
    else:
        results.append(("Hardware Modules", 0, 1))
    
    # Test API server
    if test_api_server():
        results.append(("API Server", 1, 1))
    else:
        results.append(("API Server", 0, 1))
    
    # Test pytest suite
    if test_pytest():
        results.append(("Pytest Suite", 1, 1))
    else:
        results.append(("Pytest Suite", 0, 1))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    for category, passed, total in results:
        percentage = (passed / total * 100) if total > 0 else 0
        status = "PASS" if passed == total else "FAIL"
        print(f"{category:20} {passed}/{total} ({percentage:5.1f}%) {status}")
        total_passed += passed
        total_tests += total
    
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    overall_status = "ALL TESTS PASSED" if total_passed == total_tests else "SOME TESTS FAILED"
    
    print("-" * 60)
    print(f"{'OVERALL':20} {total_passed}/{total_tests} ({overall_percentage:5.1f}%) {overall_status}")
    
    if total_passed == total_tests:
        print("\nCONGRATULATIONS! OpenPerformance platform is fully functional!")
        return 0
    else:
        print(f"\n{total_tests - total_passed} test(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 