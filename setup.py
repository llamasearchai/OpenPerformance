#!/usr/bin/env python
"""Setup script for OpenPerformance ML Platform.

Note: pyproject.toml is the primary packaging source. This setup.py remains for legacy tooling.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openperformance",
    version="1.0.3",
    author="OpenPerformance Team",
    author_email="team@openperformance.ai",
    description="Enterprise-grade ML Performance Engineering Platform for optimization, monitoring, and deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openperformance/openperformance",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0,<0.110.0",
        "uvicorn[standard]>=0.24.0,<0.30.0",
        "pydantic>=2.5.0,<3.0.0",
        "pydantic-settings>=2.1.0",
        "typer[all]>=0.9.0,<0.15.0",
        "redis>=5.0.0,<6.0.0",
        "sqlalchemy>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "psutil>=5.9.0",
        "openai>=1.20.0",
        "httpx>=0.25.0,<0.30.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
            "tox>=4.0.0",
            "hatch>=1.9.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "sphinx>=7.2.0",
        ],
        "gpu": [
            "nvidia-ml-py>=12.535.0",
            "pynvml>=11.5.0",
            "GPUtil>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlperf=mlperf.cli.main:app",
            "openperf=mlperf.cli.main:app",
            "openperf-server=mlperf.api.main:start_server",
        ],
    },
)