#!/bin/bash

# ShadowForge OS v5.1 - Dependency Installation Script
# This script installs all required Python packages for the ShadowForge AI OS

set -e

echo "🚀 Installing ShadowForge OS Dependencies..."
echo "=============================================="

# Add pip to PATH
export PATH=/home/zeroday/.local/bin:$PATH

# Core dependencies for immediate functionality
echo "📦 Installing core dependencies..."
pip install --break-system-packages \
    pytest pytest-asyncio aiohttp pydantic fastapi \
    python-dotenv psutil numpy pandas requests \
    uvicorn structlog rich typer scikit-learn \
    langchain langchain-openai langchain-community \
    crewai

echo "✅ Core dependencies installed successfully!"

# Additional useful packages
echo "📦 Installing additional packages..."
pip install --break-system-packages \
    scipy matplotlib seaborn plotly \
    sqlalchemy alembic redis motor \
    cryptography bcrypt \
    pytest-cov black isort mypy

echo "✅ Additional packages installed successfully!"

echo ""
echo "🎉 All dependencies installed!"
echo "🧪 You can now run tests with: pytest -q --asyncio-mode=auto"
echo "🚀 To start ShadowForge OS: python shadowforge.py"
echo ""