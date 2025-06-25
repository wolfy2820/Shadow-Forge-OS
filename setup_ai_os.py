#!/usr/bin/env python3
"""
ShadowForge AI OS Setup Script
Prepares the system for running the AI Operating System
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "aiohttp",
        "websockets", 
        "asyncio",
        "pathlib",
        "dataclasses"
    ]
    
    print("🐍 Installing Python packages...")
    
    for package in packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package}"
        )
        if not success:
            print(f"⚠️ Failed to install {package}, continuing anyway...")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs = [
        "/home/zeroday/ShadowForge-OS/data",
        "/home/zeroday/ShadowForge-OS/logs", 
        "/home/zeroday/ShadowForge-OS/web",
        "/home/zeroday/ShadowForge-OS/businesses",
        "/home/zeroday/ShadowForge-OS/crypto"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_desktop_shortcut():
    """Create desktop shortcut for easy access"""
    desktop_file = """[Desktop Entry]
Version=1.0
Type=Application
Name=ShadowForge AI OS
Comment=AI-Controlled Business Operating System
Exec=python3 /home/zeroday/ShadowForge-OS/launch_ai_os.py
Icon=/home/zeroday/ShadowForge-OS/icon.png
Terminal=true
Categories=Development;AI;Business;
"""
    
    desktop_path = Path.home() / "Desktop" / "ShadowForge-AI-OS.desktop"
    
    try:
        with open(desktop_path, "w") as f:
            f.write(desktop_file)
        
        # Make executable
        os.chmod(desktop_path, 0o755)
        print(f"✅ Desktop shortcut created: {desktop_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create desktop shortcut: {e}")

def create_launcher_script():
    """Create simple launcher script"""
    launcher_script = """#!/bin/bash
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
"""
    
    launcher_path = Path("/home/zeroday/ShadowForge-OS/start_ai_os.sh")
    
    try:
        with open(launcher_path, "w") as f:
            f.write(launcher_script)
        
        # Make executable
        os.chmod(launcher_path, 0o755)
        print(f"✅ Launcher script created: {launcher_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create launcher script: {e}")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check available disk space (simple check)
    import shutil
    total, used, free = shutil.disk_usage("/home")
    free_gb = free // (1024**3)
    
    if free_gb < 1:
        print(f"⚠️ Low disk space: {free_gb}GB free")
    else:
        print(f"✅ Disk space: {free_gb}GB free")
    
    return True

def create_environment_template():
    """Create environment variables template"""
    env_template = """# ShadowForge AI OS Environment Variables
# Copy this file to .env and fill in your API keys

# AI Model API Keys (at least one recommended)
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here" 
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Crypto Wallet (OPTIONAL - for real transactions)
# WARNING: Only use test wallets, never your main wallet!
export SOLANA_PRIVATE_KEY="your_solana_private_key_here"

# System Configuration
export AI_OS_DEBUG="true"
export AI_OS_PORT="8080"

# Load this file with: source .env
"""
    
    env_path = Path("/home/zeroday/ShadowForge-OS/.env.template")
    
    try:
        with open(env_path, "w") as f:
            f.write(env_template)
        print(f"✅ Environment template created: {env_path}")
        print(f"💡 Copy to .env and add your API keys for full functionality")
        
    except Exception as e:
        print(f"⚠️ Could not create environment template: {e}")

def main():
    """Main setup function"""
    print("🤖 ShadowForge AI Operating System Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Create directories
    create_directories()
    
    # Install Python packages
    install_python_packages()
    
    # Create helper scripts
    create_launcher_script()
    create_desktop_shortcut()
    create_environment_template()
    
    print("\n" + "=" * 50)
    print("✅ ShadowForge AI OS Setup Complete!")
    print("=" * 50)
    print()
    print("🚀 Quick Start:")
    print("  1. Add API keys to .env file (optional for demo)")
    print("  2. Run: python3 launch_ai_os.py")
    print("  3. Open: http://localhost:8080")
    print()
    print("📋 What's Available:")
    print("  • Natural language terminal interface")
    print("  • Web-based real-time dashboard") 
    print("  • Autonomous business creation")
    print("  • Crypto wallet integration")
    print("  • AI model integration")
    print("  • Browser automation")
    print("  • System control capabilities")
    print()
    print("💡 Try Commands Like:")
    print("  • 'create business: AI writing assistant'")
    print("  • 'check wallet balance'")
    print("  • 'research market trends'")
    print("  • 'show system status'")
    print()
    print("🌐 Web Interface: http://localhost:8080")
    print("🖥️ Terminal: Run launch_ai_os.py")
    print()
    print("Ready to revolutionize AI-powered business creation! 🚀")

if __name__ == "__main__":
    main()