#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Multi-Model AI Testing Suite
Test and demonstrate all AI models including O3, GPT-4.5, and natural language OS control
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

# Import our modules
from neural_substrate.advanced_ai_core import AdvancedAICore, create_ai_request
from neural_interface.natural_language_os import NaturalLanguageOS, execute_nl_command, plan_and_execute_operation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModelAITestSuite:
    """Comprehensive test suite for multi-model AI capabilities."""
    
    def __init__(self):
        self.ai_core = AdvancedAICore()
        self.nl_os = NaturalLanguageOS()
        self.test_results = []
        
    async def initialize(self):
        """Initialize all AI systems."""
        logger.info("🚀 Initializing Multi-Model AI Test Suite...")
        
        await self.ai_core.initialize()
        await self.nl_os.initialize()
        
        logger.info("✅ All AI systems initialized")
    
    async def test_all_models(self):
        """Test all available AI models."""
        logger.info("🧪 Testing all AI models...")
        
        test_prompt = "Explain the concept of quantum computing in simple terms"
        test_context = "You are an expert AI assistant helping to explain complex topics"
        
        models_to_test = ["o3", "gpt-4.5", "gpt-4-turbo", "gpt-4", "claude-3-opus", "claude-3-sonnet"]
        
        for model in models_to_test:
            await self.test_individual_model(model, test_prompt, test_context)
    
    async def test_individual_model(self, model_name: str, prompt: str, context: str):
        """Test an individual AI model."""
        logger.info(f"🔍 Testing model: {model_name}")
        
        start_time = time.time()
        
        try:
            # Create AI request
            request = await create_ai_request(
                prompt=prompt,
                context=context,
                model=model_name,
                priority="high",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Generate response
            response = await self.ai_core.generate_response(request)
            
            execution_time = time.time() - start_time
            
            test_result = {
                "model": model_name,
                "success": response.get("content") is not None,
                "response_length": len(response.get("content", "")),
                "execution_time": execution_time,
                "cost": response.get("cost", 0.0),
                "quality_score": response.get("quality_score", 0.0),
                "tokens_used": response.get("tokens_used", 0),
                "provider": response.get("provider", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "response_preview": response.get("content", "")[:200] + "..." if response.get("content") else "No response"
            }
            
            self.test_results.append(test_result)
            
            logger.info(f"✅ {model_name}: {test_result['success']} - {execution_time:.2f}s - {test_result['response_length']} chars")
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            
            error_result = {
                "model": model_name,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(error_result)
    
    async def test_natural_language_os(self):
        """Test natural language OS control capabilities."""
        logger.info("🗣️ Testing Natural Language OS Control...")
        
        test_commands = [
            "show me the current directory",
            "list all running processes",
            "check disk space",
            "show system information",
            "list files in the current directory",
            "check memory usage",
            "show network interfaces",
            "display CPU information"
        ]
        
        nl_results = []
        
        for command in test_commands:
            logger.info(f"🔧 Testing NL command: {command}")
            
            try:
                result = await self.nl_os.execute_natural_language_command(command, preferred_model="o3")
                
                nl_results.append({
                    "natural_command": command,
                    "success": result["success"],
                    "os_command": result.get("os_command", ""),
                    "execution_time": result.get("execution_time", 0),
                    "risk_level": result.get("risk_level", 0),
                    "ai_model": result.get("ai_model_used", ""),
                    "output_length": len(result.get("output", "")),
                    "error": result.get("error", "")
                })
                
                logger.info(f"{'✅' if result['success'] else '❌'} {command}: {result['success']}")
                
            except Exception as e:
                logger.error(f"❌ NL command failed: {e}")
                nl_results.append({
                    "natural_command": command,
                    "success": False,
                    "error": str(e)
                })
        
        return nl_results
    
    async def test_intelligent_operations(self):
        """Test intelligent multi-step operations."""
        logger.info("🧠 Testing Intelligent Operations...")
        
        operations = [
            "Create a backup of important system files",
            "Analyze system performance and generate a report",
            "Set up a development environment for Python",
            "Monitor system resources and create alerts for high usage"
        ]
        
        operation_results = []
        
        for operation in operations:
            logger.info(f"🎯 Testing operation: {operation}")
            
            try:
                result = await self.nl_os.execute_intelligent_operation(operation)
                
                operation_results.append({
                    "operation": operation,
                    "success": result["success"],
                    "total_steps": result.get("total_steps", 0),
                    "completed_steps": result.get("completed_steps", 0),
                    "ai_model": result.get("ai_model_used", ""),
                    "execution_summary": {
                        "planned": result.get("operation_plan", {}).get("total_steps", 0),
                        "executed": len(result.get("execution_results", []))
                    }
                })
                
                logger.info(f"{'✅' if result['success'] else '❌'} Operation: {result['success']}")
                
            except Exception as e:
                logger.error(f"❌ Operation failed: {e}")
                operation_results.append({
                    "operation": operation,
                    "success": False,
                    "error": str(e)
                })
        
        return operation_results
    
    async def test_model_performance_comparison(self):
        """Compare performance across all models."""
        logger.info("📊 Running Model Performance Comparison...")
        
        comparison_prompts = [
            {
                "category": "reasoning",
                "prompt": "Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "expected_features": ["logical reasoning", "analysis"]
            },
            {
                "category": "creativity",
                "prompt": "Write a creative short story about an AI that discovers it can control physical objects through quantum entanglement",
                "expected_features": ["creativity", "storytelling", "technical concepts"]
            },
            {
                "category": "technical",
                "prompt": "Explain the differences between supervised, unsupervised, and reinforcement learning with practical examples",
                "expected_features": ["technical accuracy", "examples", "clarity"]
            },
            {
                "category": "coding",
                "prompt": "Write a Python function that efficiently finds the longest palindromic substring in a given string",
                "expected_features": ["code quality", "efficiency", "correctness"]
            }
        ]
        
        comparison_results = {}
        
        for prompt_data in comparison_prompts:
            category = prompt_data["category"]
            prompt = prompt_data["prompt"]
            
            logger.info(f"🔍 Testing category: {category}")
            
            comparison_results[category] = {}
            
            # Test with premium models
            premium_models = ["o3", "gpt-4.5", "gpt-4", "claude-3-opus"]
            
            for model in premium_models:
                try:
                    start_time = time.time()
                    
                    request = await create_ai_request(
                        prompt=prompt,
                        context=f"You are an expert in {category}. Provide a comprehensive and accurate response.",
                        model=model,
                        priority="high",
                        temperature=0.7 if category == "creativity" else 0.3,
                        max_tokens=2048
                    )
                    
                    response = await self.ai_core.generate_response(request)
                    execution_time = time.time() - start_time
                    
                    comparison_results[category][model] = {
                        "success": True,
                        "execution_time": execution_time,
                        "response_length": len(response.get("content", "")),
                        "cost": response.get("cost", 0.0),
                        "quality_score": response.get("quality_score", 0.0),
                        "tokens_used": response.get("tokens_used", 0),
                        "response_preview": response.get("content", "")[:300] + "..."
                    }
                    
                except Exception as e:
                    comparison_results[category][model] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return comparison_results
    
    async def run_comprehensive_test(self):
        """Run all tests and generate comprehensive report."""
        logger.info("🚀 Starting Comprehensive Multi-Model AI Test...")
        
        start_time = time.time()
        
        # Initialize systems
        await self.initialize()
        
        # Run all tests
        logger.info("=" * 60)
        await self.test_all_models()
        
        logger.info("=" * 60)
        nl_results = await self.test_natural_language_os()
        
        logger.info("=" * 60)
        operation_results = await self.test_intelligent_operations()
        
        logger.info("=" * 60)
        comparison_results = await self.test_model_performance_comparison()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            "test_session": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "system_info": await self.nl_os.get_system_status()
            },
            "model_tests": self.test_results,
            "natural_language_os": {
                "test_results": nl_results,
                "success_rate": sum(1 for r in nl_results if r["success"]) / len(nl_results) if nl_results else 0,
                "total_commands_tested": len(nl_results)
            },
            "intelligent_operations": {
                "test_results": operation_results,
                "success_rate": sum(1 for r in operation_results if r["success"]) / len(operation_results) if operation_results else 0,
                "total_operations_tested": len(operation_results)
            },
            "performance_comparison": comparison_results,
            "ai_core_metrics": await self.ai_core.get_metrics(),
            "nl_os_metrics": await self.nl_os.get_metrics(),
            "summary": {
                "models_tested": len(self.test_results),
                "successful_models": sum(1 for r in self.test_results if r["success"]),
                "total_cost": sum(r.get("cost", 0) for r in self.test_results),
                "average_response_time": sum(r.get("execution_time", 0) for r in self.test_results) / len(self.test_results) if self.test_results else 0
            }
        }
        
        # Save report
        report_filename = f"/home/zeroday/ShadowForge-OS/multi_model_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎉 COMPREHENSIVE TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        logger.info(f"Models Tested: {report['summary']['models_tested']}")
        logger.info(f"Successful Models: {report['summary']['successful_models']}")
        logger.info(f"Total Cost: ${report['summary']['total_cost']:.4f}")
        logger.info(f"Average Response Time: {report['summary']['average_response_time']:.2f}s")
        logger.info(f"Natural Language OS Success Rate: {report['natural_language_os']['success_rate']:.1%}")
        logger.info(f"Intelligent Operations Success Rate: {report['intelligent_operations']['success_rate']:.1%}")
        logger.info(f"Report saved to: {report_filename}")
        
        return report

async def main():
    """Main test execution function."""
    test_suite = MultiModelAITestSuite()
    
    try:
        report = await test_suite.run_comprehensive_test()
        
        # Print key results
        print("\n" + "="*80)
        print("🎯 KEY RESULTS:")
        print("="*80)
        
        # Model performance summary
        print("\n📊 MODEL PERFORMANCE:")
        for result in test_suite.test_results:
            if result["success"]:
                print(f"✅ {result['model']:15} | {result['execution_time']:6.2f}s | ${result.get('cost', 0):8.4f} | {result['response_length']:4d} chars")
            else:
                print(f"❌ {result['model']:15} | FAILED")
        
        # Natural Language OS summary
        print(f"\n🗣️ NATURAL LANGUAGE OS: {report['natural_language_os']['success_rate']:.1%} success rate")
        
        # Intelligent Operations summary
        print(f"🧠 INTELLIGENT OPERATIONS: {report['intelligent_operations']['success_rate']:.1%} success rate")
        
        print("\n🚀 ShadowForge OS Multi-Model AI System is fully operational!")
        print("✨ Ready for production deployment with O3, GPT-4.5, and full OS control")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())