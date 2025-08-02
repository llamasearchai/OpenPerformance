"""
ML Performance Engineering Platform

A comprehensive platform for optimizing and monitoring machine learning workloads
across distributed systems.
"""

__version__ = "1.0.3"
__author__ = "OpenPerformance Team"
__description__ = "Enterprise-grade ML Performance Engineering Platform for optimization, monitoring, and deployment"

# Core imports
from .utils.logging import get_logger
from .utils.config import Config

# Make common functionality easily accessible
logger = get_logger(__name__)

def get_version() -> str:
    """Get the package version."""
    return __version__