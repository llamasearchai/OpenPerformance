"""
Configuration management for ML Performance Engineering Platform.

This module handles API keys, environment variables, and system configuration.
"""

import os
import logging
import json
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Main configuration class."""
    openai_api_key: Optional[str] = None
    log_level: str = "INFO"
    log_file: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite:///mlperf.db"
    cache_dir: str = "./cache"
    results_dir: str = "./results"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            database_url=os.getenv("DATABASE_URL", "sqlite:///mlperf.db"),
            cache_dir=os.getenv("MLPERF_CACHE_DIR", "./cache"),
            results_dir=os.getenv("MLPERF_RESULTS_DIR", "./results")
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'Config':
        """Load configuration from a file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, config_path: Path) -> None:
        """Save configuration to a file."""
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or config."""
    return os.getenv("OPENAI_API_KEY")


def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment variables."""
    return (
        os.getenv("ANTHROPIC_API_KEY") or 
        os.getenv("MLPERF_ANTHROPIC_KEY")
    )


def get_wandb_api_key() -> Optional[str]:
    """Get Weights & Biases API key from environment variables."""
    return (
        os.getenv("WANDB_API_KEY") or 
        os.getenv("MLPERF_WANDB_KEY")
    )


def get_mlflow_tracking_uri() -> str:
    """Get MLflow tracking URI."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def get_redis_url() -> str:
    """Get Redis connection URL."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_database_url() -> str:
    """Get database connection URL."""
    return os.getenv("DATABASE_URL", "sqlite:///mlperf.db")


def get_log_level() -> str:
    """Get logging level from environment."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def get_cache_directory() -> Path:
    """Get cache directory path."""
    cache_dir = Path(os.getenv("MLPERF_CACHE_DIR", "~/.cache/mlperf")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_output_directory() -> Path:
    """Get output directory for results."""
    output_dir = Path(os.getenv("MLPERF_OUTPUT_DIR", "./outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_distributed_config() -> Dict[str, Any]:
    """Get distributed training configuration."""
    return {
        "backend": os.getenv("MLPERF_DISTRIBUTED_BACKEND", "nccl"),
        "master_addr": os.getenv("MASTER_ADDR", "localhost"),
        "master_port": int(os.getenv("MASTER_PORT", "29500")),
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
        "rank": int(os.getenv("RANK", "0")),
        "local_rank": int(os.getenv("LOCAL_RANK", "0")),
    }


def get_gpu_config() -> Dict[str, Any]:
    """Get GPU configuration."""
    return {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", "0.8")),
        "allow_growth": os.getenv("GPU_ALLOW_GROWTH", "true").lower() == "true",
        "mixed_precision": os.getenv("MIXED_PRECISION", "false").lower() == "true",
    }


def get_profiling_config() -> Dict[str, Any]:
    """Get profiling configuration."""
    return {
        "enable_profiling": os.getenv("ENABLE_PROFILING", "false").lower() == "true",
        "profile_memory": os.getenv("PROFILE_MEMORY", "true").lower() == "true",
        "profile_cpu": os.getenv("PROFILE_CPU", "true").lower() == "true",
        "profile_gpu": os.getenv("PROFILE_GPU", "true").lower() == "true",
        "profile_network": os.getenv("PROFILE_NETWORK", "false").lower() == "true",
        "trace_file_format": os.getenv("TRACE_FILE_FORMAT", "chrome"),
        "sample_rate": float(os.getenv("PROFILE_SAMPLE_RATE", "0.01")),
    }


def get_optimization_config() -> Dict[str, Any]:
    """Get optimization configuration."""
    return {
        "enable_activation_checkpointing": os.getenv("ACTIVATION_CHECKPOINTING", "false").lower() == "true",
        "enable_gradient_compression": os.getenv("GRADIENT_COMPRESSION", "false").lower() == "true",
        "enable_mixed_precision": os.getenv("MIXED_PRECISION", "false").lower() == "true",
        "enable_cpu_offload": os.getenv("CPU_OFFLOAD", "false").lower() == "true",
        "bucket_size_mb": int(os.getenv("BUCKET_SIZE_MB", "25")),
        "compression_ratio": float(os.getenv("COMPRESSION_RATIO", "0.01")),
    }


def get_api_config() -> Dict[str, Any]:
    """Get API server configuration."""
    return {
        "host": os.getenv("API_HOST", "0.0.0.0"),
        "port": int(os.getenv("API_PORT", "8000")),
        "workers": int(os.getenv("API_WORKERS", "1")),
        "reload": os.getenv("API_RELOAD", "false").lower() == "true",
        "debug": os.getenv("API_DEBUG", "false").lower() == "true",
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    }


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration."""
    return {
        "enable_prometheus": os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true",
        "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
        "enable_grafana": os.getenv("ENABLE_GRAFANA", "false").lower() == "true",
        "grafana_url": os.getenv("GRAFANA_URL", "http://localhost:3000"),
        "metrics_collection_interval": int(os.getenv("METRICS_INTERVAL", "10")),
    }


def get_security_config() -> Dict[str, Any]:
    """Get security configuration."""
    return {
        "secret_key": os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        "enable_auth": os.getenv("ENABLE_AUTH", "false").lower() == "true",
        "jwt_expiry_hours": int(os.getenv("JWT_EXPIRY_HOURS", "24")),
        "rate_limit": os.getenv("RATE_LIMIT", "100/minute"),
    }


def get_config() -> Config:
    """Get the current configuration."""
    # Try to load from file first, then environment
    config_file = Path("config.yaml")
    if config_file.exists():
        return Config.from_file(config_file)
    
    config_file = Path("config.json")
    if config_file.exists():
        return Config.from_file(config_file)
    
    # Fallback to environment
    return Config.from_env()


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = Config()
    return config 