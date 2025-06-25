#!/bin/bash
# ShadowForge AI OS Launcher Script

echo "🚀 Starting ShadowForge AI Operating System..."
echo "================================================"

# Change to the AI OS directory
cd /home/zeroday/ShadowForge-OS

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Run the AI OS
python3 launch_ai_os.py

echo "👋 ShadowForge AI OS has stopped."
