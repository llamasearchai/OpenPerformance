from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..optimization.distributed import DistributedOptimizer, CommunicationConfig
from ..hardware.gpu import get_gpu_info
from ..utils.logging import get_logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = get_logger(__name__)
app = FastAPI(
    title="ML Performance Engineering Platform API",
    description="REST API for ML performance optimization and monitoring",
    version="0.1.0"
)

class PerformanceRequest(BaseModel):
    framework: str
    batch_size: int
    model_config: Dict[str, Any]
    hardware_info: Dict[str, Any]

class OptimizationRecommendation(BaseModel):
    area: str
    suggestion: str
    estimated_impact: float

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Performance Engineering Platform API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "system_metrics": "/system/metrics",
            "performance_analysis": "/analyze/performance",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": __import__("time").time(),
        "components": {
            "gpu_detection": len(get_gpu_info()) > 0,
            "psutil_available": PSUTIL_AVAILABLE
        }
    }

@app.post("/analyze/performance", response_model=List[OptimizationRecommendation])
async def analyze_performance(request: PerformanceRequest):
    """Analyze performance data and provide optimization recommendations"""
    try:
        # Validate framework
        supported_frameworks = ["pytorch", "tensorflow", "jax"]
        if request.framework not in supported_frameworks:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported framework: {request.framework}. Supported: {supported_frameworks}"
            )
        
        # Initialize distributed optimizer
        config = CommunicationConfig()
        optimizer = DistributedOptimizer(config=config, framework=request.framework)
        
        # Get hardware info
        gpu_info = get_gpu_info()
        
        # Generate recommendations
        recommendations = []
        
        # Memory optimization based on GPU info
        if gpu_info:
            total_memory_gb = sum(gpu.memory.total for gpu in gpu_info) / (1024**3)
            used_memory_gb = sum(gpu.memory.used for gpu in gpu_info) / (1024**3)
            memory_utilization = (used_memory_gb / total_memory_gb) if total_memory_gb > 0 else 0
            
            if memory_utilization > 0.8:
                recommendations.append(
                    OptimizationRecommendation(
                        area="Memory",
                        suggestion="Enable gradient checkpointing and mixed precision",
                        estimated_impact=0.3
                    )
                )
            
            if memory_utilization < 0.5:
                recommendations.append(
                    OptimizationRecommendation(
                        area="Memory",
                        suggestion="Increase batch size to improve GPU utilization",
                        estimated_impact=0.2
                    )
                )
        
        # Communication optimization for multi-GPU setups
        if len(gpu_info) > 1:
            recommendations.append(
                OptimizationRecommendation(
                    area="Distributed",
                    suggestion="Enable gradient compression and increase bucket size",
                    estimated_impact=0.25
                )
            )
        
        # Model-specific optimizations
        model_size_gb = request.model_config.get("size_gb", 1.0)
        if model_size_gb > 10:
            recommendations.append(
                OptimizationRecommendation(
                    area="Model Parallelism",
                    suggestion="Consider model parallelism for large models",
                    estimated_impact=0.4
                )
            )
        
        # Framework-specific recommendations
        if request.framework == "pytorch":
            recommendations.append(
                OptimizationRecommendation(
                    area="Framework",
                    suggestion="Use torch.compile for better performance",
                    estimated_impact=0.15
                )
            )
        elif request.framework == "tensorflow":
            recommendations.append(
                OptimizationRecommendation(
                    area="Framework",
                    suggestion="Enable mixed precision training",
                    estimated_impact=0.2
                )
            )
        
        # Ensure we always return at least one recommendation
        if not recommendations:
            recommendations.append(
                OptimizationRecommendation(
                    area="General",
                    suggestion="System appears well optimized",
                    estimated_impact=0.0
                )
            )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/system/metrics")
async def get_system_metrics():
    """Get real-time system metrics"""
    try:
        metrics = {
            "timestamp": __import__("time").time(),
            "gpu_info": [gpu.to_dict() for gpu in get_gpu_info()],
            "cpu_usage": 0.0,
            "memory_usage": {},
            "disk_usage": {}
        }
        
        # Add system metrics if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                metrics["cpu_usage"] = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                metrics["memory_usage"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                }
                
                disk = psutil.disk_usage('/')
                metrics["disk_usage"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            except Exception as e:
                logger.warning(f"Failed to get detailed system metrics: {e}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@app.get("/system/hardware")
async def get_hardware_info():
    """Get detailed hardware information."""
    try:
        hardware_info = {
            "timestamp": __import__("time").time(),
            "gpus": [gpu.to_dict() for gpu in get_gpu_info()],
            "cpu_info": {},
            "system_info": {}
        }
        
        if PSUTIL_AVAILABLE:
            try:
                hardware_info["cpu_info"] = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "cpu_usage_per_core": psutil.cpu_percent(percpu=True, interval=0.1)
                }
                
                memory = psutil.virtual_memory()
                hardware_info["system_info"] = {
                    "total_memory": memory.total,
                    "boot_time": psutil.boot_time(),
                    "platform": __import__("platform").platform()
                }
            except Exception as e:
                logger.warning(f"Failed to get detailed hardware info: {e}")
        
        return hardware_info
        
    except Exception as e:
        logger.error(f"Hardware info collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hardware info failed: {str(e)}")

# Add CORS middleware for frontend integration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 