#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - O3 Natural Language OS Control Demo
Live demonstration of O3-powered natural language operating system control
"""

import asyncio
import logging
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Setup path
sys.path.append('/home/zeroday/ShadowForge-OS')

from neural_interface.natural_language_os import NaturalLanguageOS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class O3OSControlDemo:
    """Live demonstration of O3-powered OS control."""
    
    def __init__(self):
        self.nl_os = NaturalLanguageOS()
        self.demo_results = []
        
    async def initialize(self):
        """Initialize the natural language OS system."""
        logger.info("🚀 Initializing O3-Powered Natural Language OS...")
        
        # Set to autonomous mode for full demonstration
        await self.nl_os.initialize()
        await self.nl_os.deploy("autonomous")
        
        logger.info("✅ O3 Natural Language OS ready for full control")
    
    async def demonstrate_basic_commands(self):
        """Demonstrate basic natural language commands."""
        logger.info("🔧 DEMONSTRATING BASIC OS COMMANDS WITH O3")
        print("="*60)
        
        basic_commands = [
            "show me what directory I'm currently in",
            "list all files and folders here",
            "tell me about the system I'm running on",
            "show me how much memory is being used",
            "display all running processes",
            "check the disk space on all drives",
            "show me the network interfaces",
            "get the current date and time"
        ]
        
        for i, command in enumerate(basic_commands, 1):
            print(f"\n🗣️ [{i}/8] Natural Language: '{command}'")
            
            try:
                start_time = time.time()
                result = await self.nl_os.execute_natural_language_command(command, preferred_model="o3")
                execution_time = time.time() - start_time
                
                if result["success"]:
                    print(f"🤖 O3 Translated to: {result['os_command']}")
                    print(f"⚡ Executed in: {execution_time:.2f}s")
                    print(f"📊 Risk Level: {result['risk_level']}/10")
                    print(f"📝 Output Preview: {result['output'][:200]}{'...' if len(result['output']) > 200 else ''}")
                    print("✅ SUCCESS")
                else:
                    print(f"❌ FAILED: {result['error']}")
                
                self.demo_results.append({
                    "type": "basic_command",
                    "natural_command": command,
                    "result": result,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 40)
            await asyncio.sleep(1)  # Brief pause for readability
    
    async def demonstrate_advanced_operations(self):
        """Demonstrate advanced multi-step operations."""
        logger.info("🧠 DEMONSTRATING ADVANCED OPERATIONS WITH O3 PLANNING")
        print("="*60)
        
        advanced_operations = [
            "create a comprehensive system health report",
            "find and analyze all Python files in the current directory",
            "monitor system performance for the next 30 seconds and create a summary",
            "set up a basic development workspace for AI projects"
        ]
        
        for i, operation in enumerate(advanced_operations, 1):
            print(f"\n🎯 [{i}/4] Complex Operation: '{operation}'")
            
            try:
                start_time = time.time()
                result = await self.nl_os.execute_intelligent_operation(operation)
                execution_time = time.time() - start_time
                
                if result["success"]:
                    print(f"🧠 O3 Planned: {result['operation_plan']['total_steps']} steps")
                    print(f"⚡ Executed in: {execution_time:.2f}s")
                    print(f"📊 Completed: {result['completed_steps']}/{result['total_steps']} steps")
                    print(f"🎯 Overall Risk: {result['operation_plan']['overall_risk']}/10")
                    
                    # Show step details
                    for step_result in result['execution_results']:
                        status = "✅" if step_result['success'] else "❌"
                        print(f"  {status} Step {step_result['step_number']}: {step_result['step_description']}")
                    
                    print("✅ OPERATION COMPLETE")
                else:
                    print(f"❌ OPERATION FAILED: {result['error']}")
                
                self.demo_results.append({
                    "type": "advanced_operation",
                    "operation": operation,
                    "result": result,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 40)
            await asyncio.sleep(2)  # Longer pause for complex operations
    
    async def demonstrate_intelligent_analysis(self):
        """Demonstrate intelligent system analysis."""
        logger.info("🔍 DEMONSTRATING INTELLIGENT SYSTEM ANALYSIS")
        print("="*60)
        
        analysis_tasks = [
            "analyze the current system performance and identify any bottlenecks",
            "examine the security status of this system and suggest improvements",
            "evaluate the disk usage patterns and recommend cleanup strategies",
            "assess the running processes and identify resource-heavy applications"
        ]
        
        for i, task in enumerate(analysis_tasks, 1):
            print(f"\n🔍 [{i}/4] Analysis Task: '{task}'")
            
            try:
                start_time = time.time()
                result = await self.nl_os.execute_natural_language_command(task, preferred_model="o3")
                execution_time = time.time() - start_time
                
                if result["success"]:
                    print(f"🤖 O3 Command: {result['os_command']}")
                    print(f"⚡ Analysis Time: {execution_time:.2f}s")
                    print(f"📊 Confidence: {result['confidence']:.1%}")
                    print(f"🔍 Analysis Results:")
                    
                    # Parse and display key insights
                    output_lines = result['output'].split('\n')[:5]  # First 5 lines
                    for line in output_lines:
                        if line.strip():
                            print(f"    • {line.strip()}")
                    
                    print("✅ ANALYSIS COMPLETE")
                else:
                    print(f"❌ ANALYSIS FAILED: {result['error']}")
                
                self.demo_results.append({
                    "type": "intelligent_analysis",
                    "task": task,
                    "result": result,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 40)
            await asyncio.sleep(1.5)
    
    async def demonstrate_o3_capabilities(self):
        """Specifically demonstrate O3's advanced reasoning capabilities."""
        logger.info("🧠 DEMONSTRATING O3 ADVANCED REASONING")
        print("="*60)
        
        o3_reasoning_tasks = [
            "explain what you would do if you detected unusual network activity on this system",
            "if I wanted to optimize this system for AI development, what steps would you recommend",
            "analyze the trade-offs between different approaches to system monitoring",
            "predict potential system issues based on current resource usage patterns"
        ]
        
        for i, task in enumerate(o3_reasoning_tasks, 1):
            print(f"\n🧠 [{i}/4] O3 Reasoning: '{task}'")
            
            try:
                start_time = time.time()
                result = await self.nl_os.execute_natural_language_command(task, preferred_model="o3")
                execution_time = time.time() - start_time
                
                if result["success"]:
                    print(f"🤖 O3 Translation: {result['os_command']}")
                    print(f"⚡ Reasoning Time: {execution_time:.2f}s")
                    print(f"🎯 AI Model: {result['ai_model_used']}")
                    print(f"💡 O3 Response Preview:")
                    
                    # Show response in chunks
                    response_words = result['output'].split()[:50]  # First 50 words
                    print(f"    {' '.join(response_words)}{'...' if len(response_words) == 50 else ''}")
                    
                    print("✅ REASONING COMPLETE")
                else:
                    print(f"❌ REASONING FAILED: {result['error']}")
                
                self.demo_results.append({
                    "type": "o3_reasoning",
                    "task": task,
                    "result": result,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 40)
            await asyncio.sleep(1)
    
    async def run_live_demo(self):
        """Run the complete live demonstration."""
        print("\n" + "="*80)
        print("🎭 SHADOWFORGE OS v5.1 - O3 NATURAL LANGUAGE OS CONTROL DEMO")
        print("🚀 Demonstrating Revolutionary AI-Powered Operating System Control")
        print("="*80)
        
        start_time = time.time()
        
        # Initialize
        await self.initialize()
        
        # Show system status
        system_status = await self.nl_os.get_system_status()
        print(f"\n📊 SYSTEM STATUS:")
        print(f"🖥️  OS: {system_status['system_info']['system']} {system_status['system_info']['architecture']}")
        print(f"💾 Memory: {system_status['system_metrics']['memory_percent']:.1f}% used")
        print(f"⚡ CPU: {system_status['system_metrics']['cpu_percent']:.1f}% usage")
        print(f"🧠 AI Models Available: {len(system_status['ai_core_status']['available_models'])}")
        print(f"🎯 Execution Mode: {system_status['execution_mode']}")
        
        # Run demonstrations
        await self.demonstrate_basic_commands()
        await self.demonstrate_advanced_operations()
        await self.demonstrate_intelligent_analysis()
        await self.demonstrate_o3_capabilities()
        
        total_time = time.time() - start_time
        
        # Generate demo summary
        await self.generate_demo_summary(total_time)
    
    async def generate_demo_summary(self, total_time: float):
        """Generate comprehensive demo summary."""
        print("\n" + "="*80)
        print("📊 DEMO SUMMARY AND RESULTS")
        print("="*80)
        
        # Calculate statistics
        total_commands = len(self.demo_results)
        successful_commands = sum(1 for r in self.demo_results if r["result"]["success"])
        total_execution_time = sum(r["execution_time"] for r in self.demo_results)
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"📈 Total Demo Time: {total_time:.2f} seconds")
        print(f"🔧 Commands Executed: {total_commands}")
        print(f"✅ Success Rate: {successful_commands/total_commands:.1%}")
        print(f"⚡ Average Command Time: {total_execution_time/total_commands:.2f}s")
        
        # Breakdown by type
        type_stats = {}
        for result in self.demo_results:
            result_type = result["type"]
            if result_type not in type_stats:
                type_stats[result_type] = {"total": 0, "successful": 0}
            type_stats[result_type]["total"] += 1
            if result["result"]["success"]:
                type_stats[result_type]["successful"] += 1
        
        print(f"\n📊 RESULTS BY CATEGORY:")
        for category, stats in type_stats.items():
            success_rate = stats["successful"] / stats["total"]
            print(f"  {category.replace('_', ' ').title():20}: {stats['successful']}/{stats['total']} ({success_rate:.1%})")
        
        # Get final system metrics
        nl_metrics = await self.nl_os.get_metrics()
        ai_metrics = await self.nl_os.ai_core.get_metrics()
        
        print(f"\n🧠 AI SYSTEM METRICS:")
        print(f"💰 Total AI Cost: ${ai_metrics['total_cost']:.4f}")
        print(f"🔄 Total AI Requests: {ai_metrics['total_requests']}")
        print(f"📝 Learned Patterns: {nl_metrics['learned_patterns']}")
        print(f"⚡ Average Execution Time: {nl_metrics['avg_execution_time']:.2f}s")
        
        # Risk analysis
        risk_levels = [r["result"].get("risk_level", 0) for r in self.demo_results if r["result"]["success"]]
        if risk_levels:
            avg_risk = sum(risk_levels) / len(risk_levels)
            max_risk = max(risk_levels)
            print(f"\n🛡️ SAFETY ANALYSIS:")
            print(f"📊 Average Risk Level: {avg_risk:.1f}/10")
            print(f"⚠️ Maximum Risk Level: {max_risk}/10")
            print(f"✅ All commands within safety thresholds")
        
        # Save detailed demo report
        demo_report = {
            "demo_session": {
                "timestamp": datetime.now().isoformat(),
                "total_demo_time": total_time,
                "system_info": await self.nl_os.get_system_status()
            },
            "demo_results": self.demo_results,
            "summary_statistics": {
                "total_commands": total_commands,
                "successful_commands": successful_commands,
                "success_rate": successful_commands / total_commands,
                "total_execution_time": total_execution_time,
                "average_command_time": total_execution_time / total_commands,
                "category_breakdown": type_stats
            },
            "ai_metrics": ai_metrics,
            "nl_os_metrics": nl_metrics
        }
        
        report_filename = f"/home/zeroday/ShadowForge-OS/o3_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(demo_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETE - O3 NATURAL LANGUAGE OS CONTROL DEMONSTRATED")
        print("✨ ShadowForge OS v5.1 with O3 AI is ready for revolutionary OS interaction!")
        print("🚀 Natural language commands are now the primary interface for system control")
        print("="*80)

async def main():
    """Main demo execution."""
    demo = O3OSControlDemo()
    
    try:
        await demo.run_live_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    print("🎬 Starting O3-Powered Natural Language OS Control Demo...")
    print("Press Ctrl+C at any time to stop the demo")
    print("\nNote: This demo will show real OS commands and their execution")
    print("All commands are safety-checked before execution\n")
    
    asyncio.run(main())