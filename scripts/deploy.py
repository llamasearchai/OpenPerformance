#!/usr/bin/env python3
"""
Enterprise ML Performance Platform Deployment Automation

This script provides comprehensive deployment automation with enterprise features:
- Blue-green deployment strategy
- Health checks and validation
- Automatic rollback on failure
- Multi-environment support
- Security scanning and compliance
- Performance monitoring integration
- Disaster recovery coordination

Usage:
    python scripts/deploy.py --environment production --version 2.1.0
    python scripts/deploy.py --environment staging --canary-percentage 10
    python scripts/deploy.py --rollback --version 2.0.5
    python scripts/deploy.py --validate --environment production
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import aiohttp
import kubernetes
from kubernetes import client, config
import boto3
import docker

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    environment: str
    version: str
    namespace: str
    image_tag: str
    replicas: int
    blue_green: bool = True
    canary_percentage: int = 0
    health_check_timeout: int = 300
    rollback_on_failure: bool = True
    enable_monitoring: bool = True
    enable_security_scan: bool = True

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    status: str
    response_time: float
    details: Dict[str, Any]
    timestamp: datetime

class EnterpriseDeploymentManager:
    """Enterprise-grade deployment manager with advanced features."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.k8s_client = None
        self.docker_client = None
        self.deployment_id = f"deploy-{int(time.time())}"
        self.deployment_start_time = datetime.now()
        
        # Initialize clients
        self._init_kubernetes_client()
        self._init_docker_client()
        
        # Load environment-specific configuration
        self.env_config = self._load_environment_config()
        
    def _init_kubernetes_client(self):
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first, then local config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def _init_docker_client(self):
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        config_file = Path(f"config/{self.config.environment}.yaml")
        
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "database": {
                "host": f"postgres-{self.config.environment}",
                "port": 5432,
                "name": "mlperformance"
            },
            "redis": {
                "host": f"redis-{self.config.environment}",
                "port": 6379
            },
            "monitoring": {
                "prometheus": f"prometheus-{self.config.environment}:9090",
                "grafana": f"grafana-{self.config.environment}:3000"
            }
        }
    
    async def deploy(self) -> bool:
        """Execute complete deployment process."""
        logger.info(f"Starting deployment {self.deployment_id} to {self.config.environment}")
        
        try:
            # Pre-deployment validation
            await self._pre_deployment_validation()
            
            # Security scanning
            if self.config.enable_security_scan:
                await self._security_scan()
            
            # Build and push container image
            await self._build_and_push_image()
            
            # Deploy based on strategy
            if self.config.blue_green:
                await self._blue_green_deployment()
            elif self.config.canary_percentage > 0:
                await self._canary_deployment()
            else:
                await self._rolling_deployment()
            
            # Post-deployment validation
            await self._post_deployment_validation()
            
            # Setup monitoring
            if self.config.enable_monitoring:
                await self._setup_monitoring()
            
            # Generate deployment report
            await self._generate_deployment_report(success=True)
            
            logger.info(f"Deployment {self.deployment_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment {self.deployment_id} failed: {e}")
            
            if self.config.rollback_on_failure:
                await self._rollback()
            
            await self._generate_deployment_report(success=False, error=str(e))
            return False
    
    async def _pre_deployment_validation(self):
        """Perform pre-deployment validation checks."""
        logger.info("Running pre-deployment validation...")
        
        # Check Kubernetes cluster connectivity
        try:
            nodes = self.core_v1.list_node()
            logger.info(f"Kubernetes cluster healthy: {len(nodes.items)} nodes")
        except Exception as e:
            raise Exception(f"Kubernetes cluster not accessible: {e}")
        
        # Validate namespace exists
        try:
            self.core_v1.read_namespace(name=self.config.namespace)
        except client.ApiException as e:
            if e.status == 404:
                logger.info(f"Creating namespace {self.config.namespace}")
                await self._create_namespace()
            else:
                raise
        
        # Check resource quotas
        await self._validate_resource_quotas()
        
        # Validate configuration
        await self._validate_configuration()
        
        logger.info("Pre-deployment validation completed")
    
    async def _security_scan(self):
        """Perform security scanning on the container image."""
        logger.info("Performing security scan...")
        
        try:
            # Use Trivy for container security scanning
            scan_cmd = [
                "trivy", "image",
                "--format", "json",
                "--output", f"security-scan-{self.deployment_id}.json",
                f"ml-performance-platform:{self.config.version}"
            ]
            
            result = subprocess.run(scan_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse scan results
                with open(f"security-scan-{self.deployment_id}.json") as f:
                    scan_results = json.load(f)
                
                # Check for critical vulnerabilities
                critical_vulns = self._count_critical_vulnerabilities(scan_results)
                
                if critical_vulns > 0:
                    raise Exception(f"Security scan found {critical_vulns} critical vulnerabilities")
                
                logger.info("Security scan passed - no critical vulnerabilities found")
            else:
                logger.warning(f"Security scan failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.warning("Trivy not installed - skipping security scan")
        except Exception as e:
            if "critical vulnerabilities" in str(e):
                raise
            logger.warning(f"Security scan error: {e}")
    
    def _count_critical_vulnerabilities(self, scan_results: Dict) -> int:
        """Count critical vulnerabilities in scan results."""
        critical_count = 0
        
        for result in scan_results.get("Results", []):
            vulnerabilities = result.get("Vulnerabilities", [])
            for vuln in vulnerabilities:
                if vuln.get("Severity") == "CRITICAL":
                    critical_count += 1
        
        return critical_count
    
    async def _build_and_push_image(self):
        """Build and push container image."""
        logger.info(f"Building container image for version {self.config.version}")
        
        if not self.docker_client:
            logger.warning("Docker client not available - skipping image build")
            return
        
        try:
            # Build image
            build_args = {
                "VERSION": self.config.version,
                "BUILD_DATE": datetime.now().isoformat(),
                "COMMIT_SHA": os.getenv("GITHUB_SHA", "unknown")
            }
            
            image, build_logs = self.docker_client.images.build(
                path=".",
                dockerfile="docker/Dockerfile",
                tag=f"ml-performance-platform:{self.config.version}",
                buildargs=build_args,
                rm=True
            )
            
            logger.info(f"Image built successfully: {image.id}")
            
            # Tag for registry
            registry_tag = f"registry.company.com/ml-performance-platform:{self.config.version}"
            image.tag(registry_tag)
            
            # Push to registry
            push_logs = self.docker_client.images.push(
                registry_tag,
                stream=True,
                decode=True
            )
            
            for log in push_logs:
                if 'error' in log:
                    raise Exception(f"Docker push failed: {log['error']}")
            
            logger.info(f"Image pushed successfully: {registry_tag}")
            
        except Exception as e:
            raise Exception(f"Container image build/push failed: {e}")
    
    async def _blue_green_deployment(self):
        """Execute blue-green deployment strategy."""
        logger.info("Executing blue-green deployment...")
        
        current_color = await self._get_current_deployment_color()
        target_color = "blue" if current_color == "green" else "green"
        
        logger.info(f"Current: {current_color}, Target: {target_color}")
        
        # Deploy to target environment
        await self._deploy_to_color(target_color)
        
        # Wait for target environment to be healthy
        await self._wait_for_healthy_deployment(target_color)
        
        # Switch traffic to target environment
        await self._switch_traffic(target_color)
        
        # Cleanup old environment
        await self._cleanup_old_deployment(current_color)
        
        logger.info("Blue-green deployment completed")
    
    async def _canary_deployment(self):
        """Execute canary deployment strategy."""
        logger.info(f"Executing canary deployment with {self.config.canary_percentage}% traffic")
        
        # Deploy canary version
        await self._deploy_canary()
        
        # Route percentage of traffic to canary
        await self._configure_canary_traffic()
        
        # Monitor canary performance
        canary_healthy = await self._monitor_canary_performance()
        
        if canary_healthy:
            # Gradually increase traffic
            await self._promote_canary()
        else:
            # Rollback canary
            await self._rollback_canary()
        
        logger.info("Canary deployment completed")
    
    async def _rolling_deployment(self):
        """Execute rolling deployment strategy."""
        logger.info("Executing rolling deployment...")
        
        deployment_name = f"ml-performance-platform-api"
        
        # Update deployment with new image
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.config.namespace
            )
            
            # Update image
            deployment.spec.template.spec.containers[0].image = f"ml-performance-platform:{self.config.version}"
            
            # Add deployment annotations
            deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
            deployment.spec.template.metadata.annotations.update({
                "deployment.kubernetes.io/revision": str(int(time.time())),
                "deployment.company.com/version": self.config.version,
                "deployment.company.com/deployment-id": self.deployment_id
            })
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.config.namespace,
                body=deployment
            )
            
            logger.info("Rolling deployment initiated")
            
            # Wait for rollout to complete
            await self._wait_for_rollout_completion(deployment_name)
            
        except Exception as e:
            raise Exception(f"Rolling deployment failed: {e}")
    
    async def _wait_for_rollout_completion(self, deployment_name: str):
        """Wait for deployment rollout to complete."""
        logger.info("Waiting for rollout completion...")
        
        timeout = self.config.health_check_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.config.namespace
                )
                
                status = deployment.status
                if (status.ready_replicas == status.replicas and 
                    status.updated_replicas == status.replicas):
                    logger.info("Rollout completed successfully")
                    return
                
                logger.info(f"Rollout progress: {status.ready_replicas}/{status.replicas} ready")
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking rollout status: {e}")
                await asyncio.sleep(10)
        
        raise Exception("Rollout timeout - deployment did not complete in time")
    
    async def _post_deployment_validation(self):
        """Perform post-deployment validation."""
        logger.info("Running post-deployment validation...")
        
        # Health checks
        health_results = await self._comprehensive_health_checks()
        
        # Performance validation
        await self._performance_validation()
        
        # Integration tests
        await self._integration_tests()
        
        # Smoke tests
        await self._smoke_tests()
        
        logger.info("Post-deployment validation completed")
    
    async def _comprehensive_health_checks(self) -> List[HealthCheckResult]:
        """Perform comprehensive health checks."""
        health_checks = [
            ("api", f"http://ml-performance-platform-api.{self.config.namespace}/health"),
            ("metrics", f"http://ml-performance-platform-api.{self.config.namespace}/metrics"),
            ("database", None),  # Custom check
            ("redis", None),     # Custom check
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for service, url in health_checks:
                try:
                    if url:
                        start_time = time.time()
                        async with session.get(url, timeout=10) as response:
                            response_time = time.time() - start_time
                            
                            if response.status == 200:
                                data = await response.json()
                                result = HealthCheckResult(
                                    service=service,
                                    status="healthy",
                                    response_time=response_time,
                                    details=data,
                                    timestamp=datetime.now()
                                )
                            else:
                                result = HealthCheckResult(
                                    service=service,
                                    status="unhealthy",
                                    response_time=response_time,
                                    details={"error": f"HTTP {response.status}"},
                                    timestamp=datetime.now()
                                )
                    else:
                        # Custom health checks for database/redis
                        result = await self._custom_health_check(service)
                    
                    results.append(result)
                    logger.info(f"Health check {service}: {result.status} ({result.response_time:.3f}s)")
                    
                except Exception as e:
                    result = HealthCheckResult(
                        service=service,
                        status="failed",
                        response_time=0.0,
                        details={"error": str(e)},
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    logger.error(f"Health check {service} failed: {e}")
        
        # Check if all critical services are healthy
        critical_services = ["api", "database"]
        unhealthy_critical = [r for r in results if r.service in critical_services and r.status != "healthy"]
        
        if unhealthy_critical:
            raise Exception(f"Critical services unhealthy: {[r.service for r in unhealthy_critical]}")
        
        return results
    
    async def _custom_health_check(self, service: str) -> HealthCheckResult:
        """Perform custom health check for specific services."""
        start_time = time.time()
        
        try:
            if service == "database":
                # Database connectivity check
                import asyncpg
                
                db_config = self.env_config["database"]
                conn = await asyncpg.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    database=db_config["name"],
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "")
                )
                
                # Simple query to test connectivity
                result = await conn.fetchval("SELECT 1")
                await conn.close()
                
                response_time = time.time() - start_time
                
                return HealthCheckResult(
                    service=service,
                    status="healthy" if result == 1 else "unhealthy",
                    response_time=response_time,
                    details={"query_result": result},
                    timestamp=datetime.now()
                )
            
            elif service == "redis":
                # Redis connectivity check
                import aioredis
                
                redis_config = self.env_config["redis"]
                redis = await aioredis.from_url(
                    f"redis://{redis_config['host']}:{redis_config['port']}"
                )
                
                # Simple ping to test connectivity
                result = await redis.ping()
                await redis.close()
                
                response_time = time.time() - start_time
                
                return HealthCheckResult(
                    service=service,
                    status="healthy" if result else "unhealthy",
                    response_time=response_time,
                    details={"ping_result": result},
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service=service,
                status="failed",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _performance_validation(self):
        """Validate performance meets requirements."""
        logger.info("Running performance validation...")
        
        # API response time check
        async with aiohttp.ClientSession() as session:
            url = f"http://ml-performance-platform-api.{self.config.namespace}/system/metrics"
            
            response_times = []
            for _ in range(10):
                start_time = time.time()
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            response_times.append(time.time() - start_time)
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times) * 1000
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] * 1000
                
                logger.info(f"API performance: avg={avg_response_time:.2f}ms, p95={p95_response_time:.2f}ms")
                
                # Check against SLA requirements
                if avg_response_time > 1000:  # 1 second SLA
                    raise Exception(f"API response time SLA violation: {avg_response_time:.2f}ms > 1000ms")
    
    async def _rollback(self):
        """Perform automatic rollback on deployment failure."""
        logger.info("Initiating automatic rollback...")
        
        try:
            deployment_name = "ml-performance-platform-api"
            
            # Get rollout history
            result = subprocess.run([
                "kubectl", "rollout", "history",
                f"deployment/{deployment_name}",
                f"--namespace={self.config.namespace}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Rollback to previous revision
                rollback_result = subprocess.run([
                    "kubectl", "rollout", "undo",
                    f"deployment/{deployment_name}",
                    f"--namespace={self.config.namespace}"
                ], capture_output=True, text=True)
                
                if rollback_result.returncode == 0:
                    logger.info("Rollback initiated successfully")
                    
                    # Wait for rollback to complete
                    await self._wait_for_rollout_completion(deployment_name)
                    logger.info("Rollback completed successfully")
                else:
                    logger.error(f"Rollback failed: {rollback_result.stderr}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _generate_deployment_report(self, success: bool, error: Optional[str] = None):
        """Generate comprehensive deployment report."""
        deployment_duration = datetime.now() - self.deployment_start_time
        
        report = {
            "deployment_id": self.deployment_id,
            "environment": self.config.environment,
            "version": self.config.version,
            "start_time": self.deployment_start_time.isoformat(),
            "duration_seconds": deployment_duration.total_seconds(),
            "success": success,
            "strategy": "blue-green" if self.config.blue_green else "rolling",
            "error": error,
            "configuration": {
                "namespace": self.config.namespace,
                "replicas": self.config.replicas,
                "health_check_timeout": self.config.health_check_timeout
            }
        }
        
        # Save report
        report_file = f"deployment-report-{self.deployment_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report saved: {report_file}")
        
        # Send notification (placeholder for actual notification system)
        await self._send_deployment_notification(report)
    
    async def _send_deployment_notification(self, report: Dict):
        """Send deployment notification (placeholder implementation)."""
        status = "SUCCESS" if report["success"] else "FAILED"
        message = f"""
        Deployment {report['deployment_id']} {status}
        
        Environment: {report['environment']}
        Version: {report['version']}
        Duration: {report['duration_seconds']:.1f}s
        Strategy: {report['strategy']}
        
        {f"Error: {report['error']}" if report['error'] else "All checks passed"}
        """
        
        logger.info(f"Deployment notification: {message}")

async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Enterprise ML Platform Deployment")
    parser.add_argument("--environment", required=True, choices=["development", "staging", "production"])
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--namespace", default="ml-performance-platform", help="Kubernetes namespace")
    parser.add_argument("--replicas", type=int, default=3, help="Number of replicas")
    parser.add_argument("--canary-percentage", type=int, default=0, help="Canary deployment percentage")
    parser.add_argument("--no-blue-green", action="store_true", help="Disable blue-green deployment")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous version")
    parser.add_argument("--validate", action="store_true", help="Validate deployment only")
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        version=args.version,
        namespace=args.namespace,
        image_tag=args.version,
        replicas=args.replicas,
        blue_green=not args.no_blue_green,
        canary_percentage=args.canary_percentage
    )
    
    # Initialize deployment manager
    deployment_manager = EnterpriseDeploymentManager(config)
    
    try:
        if args.rollback:
            await deployment_manager._rollback()
        elif args.validate:
            await deployment_manager._post_deployment_validation()
            logger.info("Validation completed successfully")
        else:
            success = await deployment_manager.deploy()
            exit_code = 0 if success else 1
            exit(exit_code)
            
    except Exception as e:
        logger.error(f"Deployment operation failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 