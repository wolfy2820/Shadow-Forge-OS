#!/usr/bin/env python3
"""
ShadowForge AI OS - Interactive Launcher
Provides both interactive terminal and web interface modes
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def show_banner():
    """Display the AI OS banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  🤖 ShadowForge AI Operating System v1.0 - "Omni-Forge Pro"                ║
║  🚀 The Ultimate AI-Powered Creation & Commerce Platform                    ║
║                                                                              ║
║  💼 Autonomous Business Creation  🧠 AI Model Integration                   ║
║  💰 Crypto Wallet Management     🌐 Web Browser Automation                  ║
║  📊 Real-time Analytics          🔮 Market Prediction Engine                ║
║  🎯 Natural Language Control     ⚡ Lightning Fast Operations               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main launcher interface"""
    clear_screen()
    show_banner()
    
    print("\n🎮 Choose your AI OS experience:")
    print("1. 🖥️  Interactive Terminal Mode (full control)")
    print("2. 🌐 Web Interface Mode (visual dashboard)")
    print("3. 🎬 Demo Mode (automated showcase)")
    print("4. 📋 System Status")
    print("5. ❌ Exit")
    
    while True:
        try:
            choice = input("\n🤖 Select option (1-5): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Interactive Terminal Mode...")
                print("💡 Use 'help' for commands, 'exit' to quit")
                time.sleep(2)
                # Run in interactive mode
                os.system("python3 ai_os_standalone.py")
                break
                
            elif choice == "2":
                print("\n🌐 Starting Web Interface Mode...")
                print("🔗 Dashboard: http://localhost:8080")
                print("⚡ Running in background - press Ctrl+C to stop")
                time.sleep(2)
                # Run in background mode
                os.system("python3 ai_os_standalone.py")
                break
                
            elif choice == "3":
                print("\n🎬 Starting Demo Mode...")
                print("📺 Automated business creation showcase")
                time.sleep(2)
                # Run demo
                os.system("python3 ai_os_standalone.py")
                break
                
            elif choice == "4":
                print("\n📊 System Status:")
                print("=" * 50)
                print("🤖 AI OS Version: v1.0")
                print("🐍 Python:", sys.version.split()[0])
                print("📁 Directory:", os.getcwd())
                print("🌐 Web Port: 8080")
                
                # Check if processes are running
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", "ai_os_standalone.py"], 
                        capture_output=True, text=True
                    )
                    if result.stdout.strip():
                        print("🟢 AI OS Status: RUNNING")
                        print("🔗 Web Interface: http://localhost:8080")
                    else:
                        print("🔴 AI OS Status: STOPPED")
                except:
                    print("🔄 AI OS Status: UNKNOWN")
                
                print("=" * 50)
                input("\n📋 Press Enter to continue...")
                
            elif choice == "5":
                print("\n👋 Thanks for using ShadowForge AI OS!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()