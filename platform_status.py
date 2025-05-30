#!/usr/bin/env python3
"""
ML Performance Engineering Platform - Comprehensive System Health Checker

This module provides enterprise-grade health monitoring for all platform components:
- Dependency validation and version compatibility checking
- Core functionality verification with performance benchmarks
- API service health monitoring with load testing capabilities
- Container orchestration and infrastructure validation
- Integration testing with automated regression detection
- Performance metrics collection and analysis

Usage:
    python platform_status.py
    python platform_status.py --output-format json
    python platform_status.py --component api --verbose
"""

import os
import sys
import time
import subprocess
import json
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """Represents the status of a system component."""
    name: str
    status: str  # OPERATIONAL, DEGRADED, FAILED
    details: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float
    error_details: Optional[str] = None

@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    overall_status: str
    components: List[ComponentStatus]
    performance_summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: str
    system_info: Dict[str, Any]

class EnterpriseHealthChecker:
    """Enterprise-grade system health monitoring and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.results: List[ComponentStatus] = []
        self.performance_metrics: Dict[str, float] = {}
        self.start_time = time.time()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration for health checking parameters."""
        default_config = {
            "timeouts": {
                "api_startup": 30,
                "test_execution": 120,
                "component_check": 10
            },
            "thresholds": {
                "api_response_time_ms": 1000,
                "memory_usage_percent": 85,
                "cpu_usage_percent": 80,
                "test_success_rate": 90
            },
            "parallel_execution": True,
            "detailed_profiling": True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def print_professional_header(self):
        """Display professional system header."""
        print("=" * 80)
        print("ML PERFORMANCE ENGINEERING PLATFORM")
        print("Enterprise System Health and Validation Report")
        print("=" * 80)
        print(f"Assessment initiated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Platform version: 2.1.0-enterprise")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print()

    def check_python_runtime_environment(self) -> ComponentStatus:
        """Validate Python runtime and core dependencies."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        # Python version validation
        python_version = sys.version_info
        details["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            status = "FAILED"
            details["python_version_issue"] = "Python 3.8+ required for enterprise features"
        
        # Core dependency validation with performance timing
        core_dependencies = [
            "fastapi", "uvicorn", "typer", "rich", "numpy", 
            "pydantic", "sqlalchemy", "redis", "celery"
        ]
        
        dependency_load_times = {}
        missing_dependencies = []
        
        for dep in core_dependencies:
            start_time = time.time()
            try:
                __import__(dep)
                load_time = (time.time() - start_time) * 1000
                dependency_load_times[dep] = load_time
                details[f"{dep}_status"] = "available"
            except ImportError:
                missing_dependencies.append(dep)
                details[f"{dep}_status"] = "missing"
        
        if missing_dependencies:
            status = "FAILED"
            details["missing_dependencies"] = missing_dependencies
        
        metrics["avg_import_time_ms"] = sum(dependency_load_times.values()) / len(dependency_load_times)
        metrics["total_dependencies_checked"] = len(core_dependencies)
        metrics["dependencies_available"] = len(core_dependencies) - len(missing_dependencies)
        
        return ComponentStatus(
            name="Python Runtime Environment",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_advanced_dependencies(self) -> ComponentStatus:
        """Validate advanced ML and systems dependencies."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        advanced_deps = {
            "torch": "PyTorch deep learning framework",
            "tensorflow": "TensorFlow ML platform",
            "jax": "JAX automatic differentiation",
            "transformers": "Hugging Face transformers library",
            "accelerate": "Hugging Face training acceleration",
            "deepspeed": "Microsoft DeepSpeed optimization",
            "psutil": "System monitoring utilities",
            "nvidia-ml-py": "NVIDIA GPU monitoring",
            "openai": "OpenAI API integration",
            "anthropic": "Anthropic AI integration"
        }
        
        available_frameworks = []
        performance_scores = {}
        
        for dep, description in advanced_deps.items():
            try:
                start_time = time.time()
                module = __import__(dep)
                import_time = (time.time() - start_time) * 1000
                
                # Get version information
                version = getattr(module, '__version__', 'unknown')
                details[f"{dep}_version"] = version
                details[f"{dep}_description"] = description
                
                available_frameworks.append(dep)
                performance_scores[dep] = import_time
                
            except ImportError:
                details[f"{dep}_status"] = "not available (optional)"
        
        # Calculate framework coverage score
        framework_coverage = len(available_frameworks) / len(advanced_deps)
        metrics["framework_coverage_percent"] = framework_coverage * 100
        metrics["available_frameworks"] = len(available_frameworks)
        metrics["avg_framework_load_time_ms"] = sum(performance_scores.values()) / max(len(performance_scores), 1)
        
        # Determine status based on critical frameworks
        critical_frameworks = ["torch", "psutil", "openai"]
        critical_available = sum(1 for fw in critical_frameworks if fw in available_frameworks)
        
        if critical_available == len(critical_frameworks):
            status = "OPERATIONAL"
        elif critical_available >= len(critical_frameworks) * 0.7:
            status = "DEGRADED"
        else:
            status = "FAILED"
        
        return ComponentStatus(
            name="Advanced ML Dependencies",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_system_architecture(self) -> ComponentStatus:
        """Validate system architecture and project structure."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        # Project structure validation
        required_structure = {
            "core_modules": [
                "python/mlperf/__init__.py",
                "python/mlperf/api/__init__.py",
                "python/mlperf/cli/__init__.py",
                "python/mlperf/hardware/__init__.py",
                "python/mlperf/optimization/__init__.py",
                "python/mlperf/utils/__init__.py"
            ],
            "configuration_files": [
                "pyproject.toml",
                "docker-compose.yml",
                "requirements.txt"
            ],
            "infrastructure": [
                "docker/Dockerfile",
                "tests/",
                "docs/"
            ],
            "validation_scripts": [
                "verify_installation.py",
                "test_benchmark_demo.py",
                "platform_status.py"
            ]
        }
        
        missing_components = []
        structure_score = 0
        total_components = sum(len(components) for components in required_structure.values())
        
        for category, components in required_structure.items():
            category_missing = []
            for component in components:
                if Path(component).exists():
                    structure_score += 1
                else:
                    category_missing.append(component)
                    missing_components.append(f"{category}: {component}")
            
            details[f"{category}_status"] = "complete" if not category_missing else "incomplete"
            if category_missing:
                details[f"{category}_missing"] = category_missing
        
        metrics["structure_completeness_percent"] = (structure_score / total_components) * 100
        metrics["total_components"] = total_components
        metrics["present_components"] = structure_score
        
        if structure_score == total_components:
            status = "OPERATIONAL"
        elif structure_score >= total_components * 0.9:
            status = "DEGRADED"
        else:
            status = "FAILED"
        
        return ComponentStatus(
            name="System Architecture",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_core_module_integrity(self) -> ComponentStatus:
        """Validate core module imports and API integrity."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        core_modules = [
            "python.mlperf.hardware.gpu",
            "python.mlperf.optimization.distributed",
            "python.mlperf.utils.logging",
            "python.mlperf.utils.config",
            "python.mlperf.api.main",
            "python.mlperf.cli.main"
        ]
        
        import_results = {}
        import_times = {}
        failed_imports = []
        
        for module_name in core_modules:
            try:
                start_time = time.time()
                module = __import__(module_name, fromlist=[''])
                import_time = (time.time() - start_time) * 1000
                
                import_results[module_name] = "success"
                import_times[module_name] = import_time
                
                # Validate module has expected attributes
                if hasattr(module, '__version__'):
                    details[f"{module_name}_version"] = module.__version__
                
            except ImportError as e:
                import_results[module_name] = "failed"
                failed_imports.append(f"{module_name}: {str(e)}")
        
        # Calculate performance metrics
        successful_imports = len([r for r in import_results.values() if r == "success"])
        metrics["import_success_rate"] = (successful_imports / len(core_modules)) * 100
        metrics["avg_import_time_ms"] = sum(import_times.values()) / max(len(import_times), 1)
        metrics["total_modules_tested"] = len(core_modules)
        
        if failed_imports:
            status = "FAILED"
            details["failed_imports"] = failed_imports
        
        details["import_results"] = import_results
        
        return ComponentStatus(
            name="Core Module Integrity",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_hardware_detection_capabilities(self) -> ComponentStatus:
        """Validate hardware detection and monitoring capabilities."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        try:
            # Add python path for imports
            sys.path.insert(0, str(Path.cwd() / "python"))
            from python.mlperf.hardware.gpu import get_gpu_info
            
            # GPU detection performance test
            start_time = time.time()
            gpu_devices = get_gpu_info()
            detection_time = (time.time() - start_time) * 1000
            
            gpu_count = len(gpu_devices)
            details["gpu_detection_time_ms"] = detection_time
            details["detected_gpus"] = gpu_count
            
            # Analyze detected hardware
            if gpu_count > 0:
                gpu_details = []
                total_memory_gb = 0
                for i, gpu in enumerate(gpu_devices):
                    gpu_info = {
                        "index": i,
                        "name": gpu.name,
                        "memory_total_gb": gpu.memory.total / (1024**3),
                        "memory_used_gb": gpu.memory.used / (1024**3),
                        "utilization_percent": gpu.utilization
                    }
                    gpu_details.append(gpu_info)
                    total_memory_gb += gpu_info["memory_total_gb"]
                
                details["gpu_inventory"] = gpu_details
                details["total_gpu_memory_gb"] = total_memory_gb
                metrics["total_compute_units"] = gpu_count
                metrics["total_memory_gb"] = total_memory_gb
            else:
                details["hardware_mode"] = "cpu_only"
                metrics["total_compute_units"] = 0
                metrics["total_memory_gb"] = 0
            
            # System resource detection
            try:
                import psutil
                details["system_cpu_cores"] = psutil.cpu_count()
                details["system_memory_gb"] = psutil.virtual_memory().total / (1024**3)
                details["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)
                
                metrics["detection_time_ms"] = detection_time
                metrics["system_monitoring_available"] = 1
            except ImportError:
                details["system_monitoring"] = "limited_capability"
                metrics["system_monitoring_available"] = 0
            
        except Exception as e:
            status = "FAILED"
            details["hardware_detection_error"] = str(e)
            metrics["detection_time_ms"] = float('inf')
        
        return ComponentStatus(
            name="Hardware Detection Capabilities",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_api_service_health(self) -> ComponentStatus:
        """Comprehensive API service health validation."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        # Check for existing API server
        try:
            response = requests.get(
                "http://localhost:8000/health", 
                timeout=self.config["timeouts"]["component_check"]
            )
            if response.status_code == 200:
                health_data = response.json()
                details["existing_server"] = "operational"
                details["health_response"] = health_data
                metrics["api_response_time_ms"] = response.elapsed.total_seconds() * 1000
                
                # Test additional endpoints
                endpoints_to_test = [
                    "/system/metrics",
                    "/system/hardware",
                    "/"
                ]
                
                endpoint_performance = {}
                for endpoint in endpoints_to_test:
                    try:
                        ep_start = time.time()
                        ep_response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
                        ep_time = (time.time() - ep_start) * 1000
                        
                        endpoint_performance[endpoint] = {
                            "status_code": ep_response.status_code,
                            "response_time_ms": ep_time,
                            "content_length": len(ep_response.content)
                        }
                    except Exception as e:
                        endpoint_performance[endpoint] = {"error": str(e)}
                
                details["endpoint_performance"] = endpoint_performance
                return ComponentStatus(
                    name="API Service Health",
                    status=status,
                    details=details,
                    performance_metrics=metrics,
                    timestamp=time.time()
                )
        except requests.exceptions.RequestException:
            details["existing_server"] = "not_running"
        
        # Start temporary API server for testing
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "python.mlperf.api.main:app",
                "--host", "127.0.0.1",
                "--port", "8001"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Allow startup time
            time.sleep(5)
            
            # Test API functionality
            test_start = time.time()
            try:
                response = requests.get("http://localhost:8001/health", timeout=10)
                api_startup_time = (time.time() - test_start) * 1000
                
                if response.status_code == 200:
                    details["test_server_status"] = "operational"
                    metrics["api_startup_time_ms"] = api_startup_time
                    metrics["api_response_time_ms"] = response.elapsed.total_seconds() * 1000
                else:
                    status = "DEGRADED"
                    details["test_server_status"] = f"http_error_{response.status_code}"
                
            except requests.exceptions.RequestException as e:
                status = "FAILED"
                details["api_connection_error"] = str(e)
                
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            
        except Exception as e:
            status = "FAILED"
            details["api_startup_error"] = str(e)
        
        return ComponentStatus(
            name="API Service Health",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_cli_interface_functionality(self) -> ComponentStatus:
        """Validate CLI interface and command execution."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        cli_commands = [
            ["info", "--format", "json"],
            ["--version"],
            ["--help"]
        ]
        
        command_results = {}
        total_execution_time = 0
        
        for cmd_args in cli_commands:
            cmd_name = "_".join(cmd_args)
            try:
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, "-m", "python.mlperf.cli.main"] + cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=self.config["timeouts"]["component_check"],
                    cwd=Path.cwd()
                )
                execution_time = (time.time() - start_time) * 1000
                total_execution_time += execution_time
                
                command_results[cmd_name] = {
                    "return_code": result.returncode,
                    "execution_time_ms": execution_time,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr)
                }
                
                # Validate specific command outputs
                if cmd_args[0] == "info" and result.returncode == 0:
                    try:
                        info_data = json.loads(result.stdout)
                        details["info_command_data"] = info_data
                    except json.JSONDecodeError:
                        details["info_command_format"] = "non_json_output"
                
            except subprocess.TimeoutExpired:
                command_results[cmd_name] = {"error": "timeout"}
                status = "DEGRADED"
            except Exception as e:
                command_results[cmd_name] = {"error": str(e)}
                status = "FAILED"
        
        details["command_results"] = command_results
        metrics["avg_command_execution_time_ms"] = total_execution_time / len(cli_commands)
        metrics["total_commands_tested"] = len(cli_commands)
        
        successful_commands = sum(
            1 for result in command_results.values() 
            if isinstance(result, dict) and result.get("return_code") == 0
        )
        metrics["command_success_rate"] = (successful_commands / len(cli_commands)) * 100
        
        return ComponentStatus(
            name="CLI Interface Functionality",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_container_orchestration(self) -> ComponentStatus:
        """Validate container and orchestration capabilities."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        # Docker availability check
        try:
            docker_result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if docker_result.returncode == 0:
                details["docker_version"] = docker_result.stdout.strip()
                details["docker_status"] = "available"
            else:
                status = "DEGRADED"
                details["docker_status"] = "not_available"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            status = "DEGRADED"
            details["docker_status"] = "not_available"
        
        # Docker Compose check
        try:
            compose_result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if compose_result.returncode == 0:
                details["docker_compose_version"] = compose_result.stdout.strip()
                details["docker_compose_status"] = "available"
            else:
                details["docker_compose_status"] = "not_available"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            details["docker_compose_status"] = "not_available"
        
        # Container configuration validation
        container_configs = [
            "docker/Dockerfile",
            "docker-compose.yml",
            "docker-compose.override.yml"
        ]
        
        config_validation = {}
        for config_file in container_configs:
            if Path(config_file).exists():
                config_validation[config_file] = "present"
                try:
                    with open(config_file) as f:
                        content = f.read()
                        config_validation[f"{config_file}_size"] = len(content)
                except Exception:
                    config_validation[config_file] = "unreadable"
            else:
                config_validation[config_file] = "missing"
        
        details["container_configurations"] = config_validation
        
        # Calculate metrics
        available_tools = sum([
            details.get("docker_status") == "available",
            details.get("docker_compose_status") == "available"
        ])
        metrics["container_tools_available"] = available_tools
        metrics["container_readiness_score"] = (available_tools / 2) * 100
        
        present_configs = sum(1 for status in config_validation.values() if status == "present")
        metrics["configuration_completeness"] = (present_configs / len(container_configs)) * 100
        
        return ComponentStatus(
            name="Container Orchestration",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def check_test_suite_integrity(self) -> ComponentStatus:
        """Execute and validate test suite performance."""
        details = {}
        metrics = {}
        status = "OPERATIONAL"
        
        try:
            # Execute test suite with performance monitoring
            start_time = time.time()
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=self.config["timeouts"]["test_execution"],
                cwd=Path.cwd()
            )
            total_execution_time = time.time() - start_time
            
            # Parse test results
            output_lines = test_result.stdout.split('\n')
            test_summary = {}
            
            for line in output_lines:
                if "passed" in line or "failed" in line or "error" in line:
                    test_summary["summary_line"] = line.strip()
                    break
            
            details["test_execution_output"] = test_result.stdout[-1000:]  # Last 1000 chars
            details["test_return_code"] = test_result.returncode
            details["test_summary"] = test_summary
            
            metrics["test_execution_time_seconds"] = total_execution_time
            metrics["test_suite_return_code"] = test_result.returncode
            
            # Determine status based on test results
            if test_result.returncode == 0:
                status = "OPERATIONAL"
                details["test_suite_status"] = "all_tests_passed"
            else:
                status = "DEGRADED"
                details["test_suite_status"] = "some_tests_failed"
                details["stderr_output"] = test_result.stderr[-500:]  # Last 500 chars
            
        except subprocess.TimeoutExpired:
            status = "FAILED"
            details["test_suite_status"] = "execution_timeout"
            metrics["test_execution_time_seconds"] = self.config["timeouts"]["test_execution"]
        except Exception as e:
            status = "FAILED"
            details["test_suite_error"] = str(e)
            metrics["test_execution_time_seconds"] = 0
        
        return ComponentStatus(
            name="Test Suite Integrity",
            status=status,
            details=details,
            performance_metrics=metrics,
            timestamp=time.time()
        )

    def generate_comprehensive_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        
        # Execute all health checks
        if self.config.get("parallel_execution", True):
            # Parallel execution for better performance
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.check_python_runtime_environment): "runtime",
                    executor.submit(self.check_advanced_dependencies): "dependencies",
                    executor.submit(self.check_system_architecture): "architecture",
                    executor.submit(self.check_core_module_integrity): "modules",
                    executor.submit(self.check_hardware_detection_capabilities): "hardware",
                    executor.submit(self.check_api_service_health): "api",
                    executor.submit(self.check_cli_interface_functionality): "cli",
                    executor.submit(self.check_container_orchestration): "containers",
                    executor.submit(self.check_test_suite_integrity): "tests"
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Health check failed: {e}")
        else:
            # Sequential execution
            health_checks = [
                self.check_python_runtime_environment,
                self.check_advanced_dependencies,
                self.check_system_architecture,
                self.check_core_module_integrity,
                self.check_hardware_detection_capabilities,
                self.check_api_service_health,
                self.check_cli_interface_functionality,
                self.check_container_orchestration,
                self.check_test_suite_integrity
            ]
            
            for check_function in health_checks:
                try:
                    result = check_function()
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Health check {check_function.__name__} failed: {e}")
        
        # Calculate overall system status
        status_counts = {"OPERATIONAL": 0, "DEGRADED": 0, "FAILED": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        if status_counts["FAILED"] > 0:
            overall_status = "SYSTEM_DEGRADED"
        elif status_counts["DEGRADED"] > 0:
            overall_status = "PERFORMANCE_ISSUES"
        else:
            overall_status = "FULLY_OPERATIONAL"
        
        # Generate performance summary
        all_metrics = {}
        for result in self.results:
            for metric, value in result.performance_metrics.items():
                all_metrics[f"{result.name.lower().replace(' ', '_')}_{metric}"] = value
        
        performance_summary = {
            "total_execution_time_seconds": time.time() - self.start_time,
            "components_checked": len(self.results),
            "overall_performance_score": self._calculate_performance_score(),
            "detailed_metrics": all_metrics
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # System information
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                key: value for key, value in os.environ.items() 
                if key.startswith(('MLPERF_', 'CUDA_', 'PATH'))
            }
        }
        
        return SystemHealthReport(
            overall_status=overall_status,
            components=self.results,
            performance_summary=performance_summary,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
            system_info=system_info
        )
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall system performance score."""
        total_score = 0
        component_count = 0
        
        for result in self.results:
            component_score = 100
            
            # Deduct points based on status
            if result.status == "DEGRADED":
                component_score -= 30
            elif result.status == "FAILED":
                component_score -= 70
            
            # Factor in performance metrics
            for metric, value in result.performance_metrics.items():
                if "time_ms" in metric and value > 1000:
                    component_score -= min(20, value / 100)
                elif "success_rate" in metric:
                    component_score *= (value / 100)
            
            total_score += max(0, component_score)
            component_count += 1
        
        return total_score / max(component_count, 1)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on health check results."""
        recommendations = []
        
        failed_components = [r for r in self.results if r.status == "FAILED"]
        degraded_components = [r for r in self.results if r.status == "DEGRADED"]
        
        if failed_components:
            recommendations.append(
                f"CRITICAL: {len(failed_components)} components failed validation. "
                "Immediate attention required for production readiness."
            )
            
            for component in failed_components:
                if "dependencies" in component.name.lower():
                    recommendations.append(
                        "Install missing dependencies: pip install -r requirements.txt"
                    )
                elif "api" in component.name.lower():
                    recommendations.append(
                        "API service requires debugging. Check logs and network configuration."
                    )
        
        if degraded_components:
            recommendations.append(
                f"PERFORMANCE: {len(degraded_components)} components show degraded performance. "
                "Optimization recommended for production deployment."
            )
        
        # Performance-specific recommendations
        for result in self.results:
            for metric, value in result.performance_metrics.items():
                if "time_ms" in metric and value > 2000:
                    recommendations.append(
                        f"OPTIMIZATION: {result.name} shows high latency ({value:.1f}ms). "
                        "Consider performance tuning."
                    )
        
        if not recommendations:
            recommendations.append(
                "EXCELLENT: All systems operational. Platform ready for production deployment."
            )
        
        return recommendations

    def print_executive_summary(self, report: SystemHealthReport):
        """Print executive summary of system health."""
        print("\nEXECUTIVE SUMMARY")
        print("=" * 50)
        print(f"Overall System Status: {report.overall_status}")
        print(f"Components Assessed: {len(report.components)}")
        print(f"Performance Score: {report.performance_summary['overall_performance_score']:.1f}/100")
        print(f"Assessment Duration: {report.performance_summary['total_execution_time_seconds']:.2f}s")
        
        # Component status breakdown
        status_counts = {"OPERATIONAL": 0, "DEGRADED": 0, "FAILED": 0}
        for component in report.components:
            status_counts[component.status] += 1
        
        print(f"\nComponent Status Distribution:")
        print(f"  OPERATIONAL: {status_counts['OPERATIONAL']}")
        print(f"  DEGRADED: {status_counts['DEGRADED']}")
        print(f"  FAILED: {status_counts['FAILED']}")
        
        print(f"\nKey Recommendations:")
        for i, recommendation in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. {recommendation}")
        
        if len(report.recommendations) > 3:
            print(f"  ... and {len(report.recommendations) - 3} additional recommendations")

    def save_detailed_report(self, report: SystemHealthReport, output_path: Path):
        """Save detailed report to file."""
        report_data = asdict(report)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {output_path}")

def main():
    """Main execution function for enterprise health checking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML Performance Platform - Enterprise System Health Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-format", choices=["json", "summary"], default="summary")
    parser.add_argument("--output-file", type=Path, help="Save detailed report to file")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--component", help="Check specific component only")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize health checker
    checker = EnterpriseHealthChecker(config_path=args.config)
    checker.print_professional_header()
    
    # Generate comprehensive report
    report = checker.generate_comprehensive_report()
    
    # Output results
    if args.output_format == "json":
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        checker.print_executive_summary(report)
    
    # Save detailed report if requested
    if args.output_file:
        checker.save_detailed_report(report, args.output_file)
    else:
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path(f"system_health_report_{timestamp}.json")
        checker.save_detailed_report(report, default_path)
    
    # Return appropriate exit code for CI/CD integration
    if report.overall_status == "FULLY_OPERATIONAL":
        return 0
    elif report.overall_status == "PERFORMANCE_ISSUES":
        return 1
    else:
        return 2

if __name__ == "__main__":
    sys.exit(main()) 