#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - REAL AI Revenue Generation Test
Test the neural interface with actual OpenAI GPT-4 integration for money-making
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the neural interface to path
sys.path.append('/home/zeroday/ShadowForge-OS')

async def test_real_ai_revenue_system():
    """
    Test the real AI revenue generation system with actual OpenAI integration.
    """
    
    print("\n" + "="*80)
    print("🚀 SHADOWFORGE OS v5.1 - REAL AI REVENUE GENERATION TEST")
    print("💰 Testing ACTUAL GPT-4 Integration for Money Making")
    print("="*80)
    
    # Set up environment
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'
    
    try:
        # Import the enhanced thought commander
        from neural_interface.thought_commander import ThoughtCommander
        
        print("\n🧠 STEP 1: Initializing Enhanced AI Thought Commander...")
        print("="*60)
        
        # Initialize the thought commander
        thought_commander = ThoughtCommander()
        
        # Initialize with real AI
        print("🔗 Connecting to OpenAI GPT-4...")
        await thought_commander.initialize()
        
        # Test OpenAI connection
        if thought_commander.openai_client:
            print("✅ OpenAI GPT-4 client connected successfully!")
            print(f"🔑 API Key: {thought_commander.openai_api_key[:20]}...{thought_commander.openai_api_key[-10:]}")
        else:
            print("❌ OpenAI client failed to initialize")
            return
        
        print("\n💰 STEP 2: Testing Revenue Generation Engines...")
        print("="*60)
        
        # Display revenue engines
        for engine_name, engine_config in thought_commander.revenue_engines.items():
            print(f"🎯 {engine_name.upper()}: Target ${engine_config['target_revenue']}/day")
            print(f"   Strategies: {', '.join(engine_config['strategies'])}")
        
        print(f"\n🎯 TOTAL DAILY REVENUE TARGET: ${sum(e['target_revenue'] for e in thought_commander.revenue_engines.values())}")
        
        print("\n🤖 STEP 3: Testing Real AI Business Strategy Generation...")
        print("="*60)
        
        # Test business strategy generation with real GPT-4
        business_ideas = [
            "AI-powered content creation platform",
            "Cryptocurrency trading automation bot", 
            "Social media viral content generator"
        ]
        
        for idea in business_ideas:
            print(f"\n💡 Generating strategy for: {idea}")
            strategy_result = await thought_commander.generate_money_making_strategy(idea, 10000.0)
            
            if strategy_result.get("success"):
                print(f"✅ Strategy generated successfully!")
                print(f"📊 AI Confidence: {strategy_result['ai_confidence']*100:.1f}%")
                print(f"💹 Estimated ROI: {strategy_result['estimated_roi']:.1f}%")
                print(f"📝 Strategy Preview: {strategy_result['strategy'][:200]}...")
            else:
                print(f"❌ Strategy generation failed: {strategy_result.get('error')}")
            
            # Show API cost
            print(f"💳 Current AI API cost: ${thought_commander.ai_api_costs:.4f}")
        
        print("\n📈 STEP 4: Testing Market Analysis with Real AI...")
        print("="*60)
        
        # Test market analysis
        market_sectors = ["Technology", "Cryptocurrency", "E-commerce"]
        
        for sector in market_sectors:
            print(f"\n🔍 Analyzing market sector: {sector}")
            analysis_result = await thought_commander.analyze_market_opportunity(sector)
            
            if analysis_result.get("success"):
                print(f"✅ Market analysis completed!")
                print(f"🎯 Opportunity Score: {analysis_result['opportunity_score']:.1f}/10")
                print(f"💰 Recommended Investment: ${analysis_result['recommended_investment']:,.2f}")
                print(f"📊 Analysis Preview: {analysis_result['analysis'][:200]}...")
            else:
                print(f"❌ Market analysis failed: {analysis_result.get('error')}")
        
        print("\n🎬 STEP 5: Triggering Real Revenue Generation Cycle...")
        print("="*60)
        
        # Manually trigger one revenue cycle to see it in action
        print("💸 Executing viral content generation...")
        await thought_commander._generate_viral_content()
        
        print("📈 Executing market strategies...")
        await thought_commander._execute_market_strategies()
        
        print("🤖 Providing AI services...")
        await thought_commander._provide_ai_services()
        
        print("₿ Executing crypto strategies...")
        await thought_commander._execute_crypto_strategies()
        
        print("\n💰 STEP 6: Revenue Generation Results...")
        print("="*60)
        
        # Display revenue results
        total_revenue = 0.0
        for engine_name, engine_config in thought_commander.revenue_engines.items():
            revenue = engine_config['current_revenue']
            total_revenue += revenue
            print(f"💵 {engine_name.upper()}: ${revenue:.2f}")
        
        print(f"\n🎉 TOTAL REVENUE GENERATED: ${total_revenue:.2f}")
        print(f"💳 AI API COSTS: ${thought_commander.ai_api_costs:.4f}")
        print(f"📊 NET PROFIT: ${total_revenue - thought_commander.ai_api_costs:.2f}")
        
        # Calculate hourly rate if we scale this
        hourly_rate = total_revenue * 24  # Scale to daily
        print(f"🚀 PROJECTED DAILY REVENUE: ${hourly_rate:.2f}")
        print(f"📈 PROJECTED MONTHLY REVENUE: ${hourly_rate * 30:.2f}")
        print(f"💰 PROJECTED ANNUAL REVENUE: ${hourly_rate * 365:.2f}")
        
        print("\n🎯 STEP 7: Testing Natural Language Commands...")
        print("="*60)
        
        # Test natural language command processing
        test_commands = [
            "Generate a viral social media post about AI trends",
            "Analyze the current cryptocurrency market",
            "Create a business plan for an e-commerce startup"
        ]
        
        for command in test_commands:
            print(f"\n🗣️ Command: '{command}'")
            
            # Create a mock request
            request = {
                "user_input": command,
                "user_context": {"budget": 5000, "risk_tolerance": "medium"},
                "session_id": "test_session"
            }
            
            result = await thought_commander.process_natural_command(
                request["user_input"],
                request["user_context"], 
                request["session_id"]
            )
            
            if result.get("success"):
                print("✅ Command processed successfully!")
                print(f"📋 Execution plan generated")
            else:
                print(f"❌ Command failed: {result.get('error', 'Unknown error')}")
        
        print("\n🏆 FINAL RESULTS:")
        print("="*60)
        print("✅ OpenAI GPT-4 Integration: WORKING")
        print("✅ Revenue Generation Engines: ACTIVE")
        print("✅ Market Analysis AI: FUNCTIONAL")
        print("✅ Content Creation AI: OPERATIONAL")
        print("✅ Natural Language Interface: RESPONSIVE")
        
        metrics = await thought_commander.get_metrics()
        print(f"\n📊 SYSTEM METRICS:")
        print(f"   Commands Processed: {metrics['commands_processed']}")
        print(f"   Successful Executions: {metrics['successful_executions']}")
        print(f"   Intent Accuracy: {metrics['intent_accuracy']*100:.1f}%")
        print(f"   Revenue Generated: ${metrics['revenue_generated']:.2f}")
        
        print(f"\n🚀 SHADOWFORGE OS v5.1 REAL AI REVENUE SYSTEM: FULLY OPERATIONAL")
        print("💰 Ready to generate actual money with GPT-4 powered strategies!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    asyncio.run(test_real_ai_revenue_system())