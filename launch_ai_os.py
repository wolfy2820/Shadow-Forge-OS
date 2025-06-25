#!/usr/bin/env python3
"""
ShadowForge AI OS Launcher
Complete AI Operating System with Web Interface and Business Intelligence

Run this to start the complete AI Operating System with:
- Natural language terminal interface
- Web-based real-time dashboard
- Crypto wallet integration
- Autonomous business creation
- AI model integration (Claude, GPT, etc.)
- Browser automation
- System control capabilities
"""

import asyncio
import logging
import sys
import os
import signal
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_os_core import ShadowForgeAIOS, AIConfig
from web_interface import WebInterface

class AIOperatingSystemLauncher:
    """Complete AI Operating System Launcher"""
    
    def __init__(self):
        self.logger = logging.getLogger("AIOperatingSystemLauncher")
        self.ai_os = None
        self.web_interface = None
        self.is_running = False
        
    async def initialize_ai_os(self):
        """Initialize the AI Operating System"""
        try:
            print("🚀 ShadowForge AI Operating System v1.0")
            print("=" * 60)
            print("The World's First AI-Controlled Business OS")
            print("=" * 60)
            
            # Get API configuration from user
            config = await self._get_configuration()
            
            # Initialize AI OS
            self.ai_os = ShadowForgeAIOS(config)
            await self.ai_os.initialize()
            
            # Initialize Web Interface
            self.web_interface = WebInterface(self.ai_os, port=8080)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ AI OS initialization failed: {e}")
            return False
    
    async def _get_configuration(self) -> AIConfig:
        """Get configuration from user or environment"""
        print("\n🔧 Configuration Setup")
        print("=" * 30)
        
        # Try to get from environment variables first
        config = AIConfig(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            solana_wallet_private_key=os.getenv("SOLANA_PRIVATE_KEY", ""),
            developer_mode=True,
            business_creation_enabled=True,
            crypto_trading_enabled=False,
            max_transaction_amount=100.0
        )
        
        # Interactive configuration if no env vars
        if not any([config.openrouter_api_key, config.openai_api_key, config.anthropic_api_key]):
            print("\n⚠️ No API keys found in environment variables.")
            print("You can run with demo mode or provide API keys.")
            print("\nDemo mode will simulate AI responses.")
            
            mode = input("\nRun in demo mode? (y/n) [default: y]: ").strip().lower()
            
            if mode != 'n':
                print("✅ Running in demo mode")
                config.developer_mode = True
            else:
                # Get API keys interactively
                print("\n🔑 API Key Configuration:")
                print("You can get API keys from:")
                print("- OpenRouter: https://openrouter.ai/")
                print("- OpenAI: https://platform.openai.com/")
                print("- Anthropic: https://console.anthropic.com/")
                
                openrouter_key = input("\nOpenRouter API Key (optional): ").strip()
                openai_key = input("OpenAI API Key (optional): ").strip()
                anthropic_key = input("Anthropic API Key (optional): ").strip()
                
                if openrouter_key:
                    config.openrouter_api_key = openrouter_key
                if openai_key:
                    config.openai_api_key = openai_key
                if anthropic_key:
                    config.anthropic_api_key = anthropic_key
        
        # Solana wallet configuration
        if not config.solana_wallet_private_key:
            print("\n💰 Crypto Wallet Configuration:")
            print("For real crypto transactions, provide a Solana wallet private key.")
            print("⚠️ WARNING: Only use a test wallet, never your main wallet!")
            
            wallet_key = input("\nSolana Private Key (optional, for demo leave empty): ").strip()
            if wallet_key:
                config.solana_wallet_private_key = wallet_key
                print("✅ Crypto wallet configured")
            else:
                print("✅ Running with demo wallet")
        
        print(f"\n✅ Configuration complete!")
        print(f"API Keys: {bool(config.openrouter_api_key or config.openai_api_key or config.anthropic_api_key)}")
        print(f"Crypto Wallet: {bool(config.solana_wallet_private_key)}")
        print(f"Developer Mode: {config.developer_mode}")
        
        return config
    
    async def start_services(self):
        """Start all AI OS services"""
        try:
            print("\n🌐 Starting services...")
            
            # Start web interface in background
            web_task = asyncio.create_task(self.web_interface.start_server())
            
            # Wait a moment for web server to start
            await asyncio.sleep(2)
            
            print(f"✅ Web Interface: http://localhost:8080")
            print(f"✅ WebSocket API: ws://localhost:8081")
            print(f"✅ Terminal Interface: Active")
            
            self.is_running = True
            
            # Start background monitoring
            monitor_task = asyncio.create_task(self._background_monitoring())
            
            return web_task, monitor_task
            
        except Exception as e:
            self.logger.error(f"❌ Service startup failed: {e}")
            raise
    
    async def _background_monitoring(self):
        """Background monitoring and updates"""
        while self.is_running:
            try:
                # Send periodic updates to web interface
                if self.web_interface and self.web_interface.connected_clients:
                    status = self.ai_os.get_system_status()
                    
                    await self.web_interface.broadcast_update({
                        "type": "status",
                        "status": status
                    })
                    
                    # Business updates
                    if self.ai_os.business.active_businesses:
                        await self.web_interface.broadcast_update({
                            "type": "business_update", 
                            "businesses": self.ai_os.business.active_businesses
                        })
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Background monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def run_terminal_interface(self):
        """Run the terminal interface"""
        try:
            print("\n" + "=" * 60)
            print("🖥️ SHADOWFORGE AI OS TERMINAL")
            print("=" * 60)
            print("🌐 Web Interface: http://localhost:8080")
            print("💬 Natural Language Commands Enabled")
            print("🤖 AI Business Creation Active")
            print("💰 Crypto Wallet Integration Ready")
            print("=" * 60)
            print("\n💡 Try these commands:")
            print("  • 'create business: AI productivity tool'")
            print("  • 'check my wallet balance'")
            print("  • 'browse google trends for market research'") 
            print("  • 'show system status'")
            print("  • 'help' for more commands")
            print("  • 'web' to open web interface")
            print("  • 'exit' to shutdown")
            print("\n")
            
            while self.is_running:
                try:
                    # Get user input
                    user_input = input("🤖 AI OS > ").strip()
                    
                    if user_input.lower() in ['exit', 'quit', 'shutdown']:
                        print("👋 Shutting down AI Operating System...")
                        break
                    
                    elif user_input.lower() == 'web':
                        print("🌐 Web interface: http://localhost:8080")
                        continue
                    
                    elif user_input.lower() == 'clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        continue
                    
                    elif user_input.lower() == 'help':
                        await self._show_help()
                        continue
                    
                    elif user_input.lower() == 'status':
                        await self._show_status()
                        continue
                    
                    elif user_input.lower() == 'demo':
                        await self._run_demo()
                        continue
                    
                    elif user_input:
                        # Process command through AI OS
                        print(f"🔄 Processing: {user_input}")
                        
                        result = await self.ai_os.execute_command(user_input)
                        
                        if result["success"]:
                            print("✅ Success!")
                            await self._display_result(result)
                        else:
                            print(f"❌ Error: {result.get('error', 'Unknown error')}")
                            if "suggestion" in result:
                                print(f"💡 Suggestion: {result['suggestion']}")
                
                except KeyboardInterrupt:
                    print("\n👋 Shutting down...")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"❌ Terminal interface error: {e}")
    
    async def _show_help(self):
        """Show comprehensive help"""
        print("""
🤖 ShadowForge AI Operating System - Commands

💼 BUSINESS COMMANDS:
  create business [idea]     - Create new autonomous business
  show businesses           - List active businesses
  business status          - Show business performance
  launch [business name]   - Launch specific business

💰 CRYPTO COMMANDS:
  check wallet            - Show wallet balance and status
  wallet status          - Detailed wallet information
  send payment [amount]  - Send crypto payment
  
🌐 WEB & RESEARCH:
  browse [url]           - Navigate to website
  research [topic]       - Research market trends
  market analysis        - Analyze current opportunities
  
🖥️ SYSTEM COMMANDS:
  show files            - List directory contents
  system info          - Show system information
  status               - Show AI OS status
  processes            - Show running processes
  
🧠 AI COMMANDS:
  ask [question]        - Query AI models
  reasoning [problem]   - Use advanced reasoning
  creative [task]       - Use creative AI
  
🎮 SPECIAL COMMANDS:
  demo                 - Run business creation demo
  web                  - Show web interface URL
  help                 - Show this help
  clear                - Clear terminal
  exit                 - Shutdown AI OS

🗣️ NATURAL LANGUAGE:
You can also use natural language like:
  "Create an AI writing assistant business"
  "Check my cryptocurrency balance"
  "Research trending topics on social media"
  "Show me the performance of my businesses"
        """)
    
    async def _show_status(self):
        """Show detailed system status"""
        status = self.ai_os.get_system_status()
        
        print(f"""
🤖 AI Operating System Status
{'='*40}
Version: {status['ai_os_version']}
Running: {status['running']}
Uptime: {status['uptime']}

📊 Components Status:
  System Commands: {'✅' if status['components']['system_command'] else '❌'}
  Browser Control: {'✅' if status['components']['browser_control'] else '❌'}
  Crypto Wallet: {'✅' if status['components']['crypto_wallet'] else '❌'}
  AI Models: {'✅' if status['components']['ai_models'] else '❌'}
  Business Engine: {'✅' if status['components']['business_automation'] > 0 else '❌'} ({status['components']['business_automation']} active)

📈 Session Statistics:
  Commands Processed: {status['session_stats']['commands_processed']}
  Successful Commands: {status['session_stats']['successful_commands']}
  Active Businesses: {status['session_stats']['active_businesses']}
  Total Revenue: ${status['session_stats']['total_business_revenue']:.2f}

🔧 Developer Panel:
  API Status: {status['developer_panel']['api_status']}
  System Health: Available
        """)
    
    async def _run_demo(self):
        """Run a business creation demo"""
        print("\n🎮 Running Business Creation Demo...")
        print("=" * 40)
        
        demo_businesses = [
            "AI-powered content writing assistant",
            "Social media automation platform", 
            "Crypto portfolio tracker",
            "No-code website builder"
        ]
        
        for i, business_idea in enumerate(demo_businesses, 1):
            print(f"\n📝 Demo {i}/4: Creating '{business_idea}'")
            
            result = await self.ai_os.execute_command(f"create business: {business_idea}")
            
            if result["success"]:
                print(f"✅ '{business_idea}' created successfully!")
                if "business_plan" in result:
                    plan = result["business_plan"]
                    print(f"   💰 Estimated Revenue: ${plan.get('estimated_revenue', 0):,.2f}")
                    print(f"   📊 Market Score: {plan.get('market_score', 0):.1%}")
            else:
                print(f"❌ Failed to create '{business_idea}'")
            
            await asyncio.sleep(1)  # Brief pause between demos
        
        print(f"\n🎉 Demo Complete! Check 'show businesses' to see results.")
    
    async def _display_result(self, result: dict):
        """Display command result in formatted way"""
        try:
            # Business results
            if "business_plan" in result:
                plan = result["business_plan"]
                print(f"🏢 Business: {plan['name']}")
                print(f"   💰 Revenue Est: ${plan.get('estimated_revenue', 0):,.2f}")
                print(f"   📊 Market Score: {plan.get('market_score', 0):.1%}")
                print(f"   🆔 ID: {plan['id']}")
            
            # Wallet results
            elif "balance" in result:
                print(f"💰 Wallet Balance: ${result['balance']:.2f}")
                if "wallet_address" in result:
                    addr = result['wallet_address']
                    print(f"📍 Address: {addr[:8]}...{addr[-8:] if len(addr) > 16 else addr}")
            
            # Business list
            elif "businesses" in result:
                businesses = result["businesses"]
                if businesses:
                    print(f"🏢 Active Businesses ({len(businesses)}):")
                    for business in businesses:
                        status = business.get('status', 'unknown')
                        revenue = business.get('total_revenue', 0)
                        print(f"   • {business['name']}: {status} (${revenue:.2f})")
                else:
                    print("🏢 No businesses created yet")
            
            # System output
            elif "output" in result:
                output = result["output"].strip()
                if output:
                    print("📋 Output:")
                    for line in output.split('\n')[:10]:  # Limit output
                        print(f"   {line}")
            
            # Page content (browser)
            elif "page_data" in result:
                page = result["page_data"]
                print(f"🌐 Page: {page['url']}")
                print(f"   Status: {page['status']}")
                print(f"   Size: {page['content_length']} bytes")
            
        except Exception as e:
            print(f"📄 Result: {result}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\n🛑 Received shutdown signal...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/zeroday/ShadowForge-OS/ai_os.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and run AI OS
    launcher = AIOperatingSystemLauncher()
    
    try:
        # Setup signal handlers
        launcher.setup_signal_handlers()
        
        # Initialize AI OS
        if not await launcher.initialize_ai_os():
            print("❌ Failed to initialize AI Operating System")
            return
        
        # Start services
        web_task, monitor_task = await launcher.start_services()
        
        # Run terminal interface (blocks until user exits)
        terminal_task = asyncio.create_task(launcher.run_terminal_interface())
        
        # Wait for terminal to finish
        await terminal_task
        
        # Cancel other tasks
        web_task.cancel()
        monitor_task.cancel()
        
        print("👋 ShadowForge AI Operating System shutdown complete")
        
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
    except Exception as e:
        print(f"❌ AI OS error: {e}")
        logging.exception("AI OS startup failed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)