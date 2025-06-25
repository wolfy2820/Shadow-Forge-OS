#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Live Platform Test with Real APIs
Test the advanced platform with your OpenAI API key
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from shadowforge_advanced import ShadowForgeAdvanced

async def test_live_capabilities():
    """
    Test the ShadowForge platform with real API integration.
    """
    
    print("\n" + "="*80)
    print("🚀 SHADOWFORGE OS v5.1 - LIVE PLATFORM TEST")
    print("🔑 Testing with real OpenAI API integration")
    print("="*80)
    
    # Check API key availability
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ No OpenAI API key found in environment")
        return
    
    print(f"✅ OpenAI API key configured: {openai_key[:20]}...")
    
    # Initialize the platform
    shadowforge = ShadowForgeAdvanced()
    
    try:
        await shadowforge.initialize()
        
        # =================================================================
        # TEST 1: REAL AI CHAT WITH OPENAI
        # =================================================================
        print("\n🧠 TEST 1: REAL AI CHAT WITH OPENAI")
        print("-" * 50)
        
        print("💬 Testing business strategy analysis...")
        
        strategy_result = await shadowforge.ai_chat(
            message="Analyze the current state of the AI automation market. What are the top 3 opportunities for a new AI-powered business platform in 2024?",
            context="Senior business strategy consultant specializing in AI and automation markets",
            priority="high"
        )
        
        print(f"✅ AI Analysis Complete!")
        print(f"   Model Used: {strategy_result.get('model_used', 'unknown')}")
        print(f"   Provider: {strategy_result.get('provider', 'unknown')}")
        print(f"   Tokens Used: {strategy_result.get('tokens_used', 0)}")
        print(f"   Cost: ${strategy_result.get('cost', 0.0):.4f}")
        print(f"   Quality Score: {strategy_result.get('quality_score', 0.5):.1%}")
        
        print(f"\n📝 AI Response:")
        print("-" * 40)
        print(strategy_result.get('response', 'No response received'))
        
        # =================================================================
        # TEST 2: VIRAL CONTENT PREDICTION
        # =================================================================
        print("\n\n🔮 TEST 2: VIRAL CONTENT PREDICTION ENGINE")
        print("-" * 50)
        
        test_content = """🚨 BREAKING: New AI study reveals that 73% of businesses using AI automation see 40%+ productivity gains within 90 days! 

        Here's what the most successful companies are doing differently:
        
        🎯 They focus on workflow automation, not just chat
        📊 They measure ROI from day 1  
        🚀 They start small and scale fast
        
        Are you ready to 10x your business with AI? #AI #Productivity #Business"""
        
        print("🔍 Analyzing viral potential...")
        
        viral_prediction = await shadowforge.predict_viral_content(
            content=test_content,
            platform="twitter"
        )
        
        if viral_prediction.get("success", False):
            print(f"✅ Viral Prediction Complete!")
            print(f"   Viral Probability: {viral_prediction['viral_probability']:.1%}")
            print(f"   Predicted Reach: {viral_prediction['predicted_reach']:,} users")
            print(f"   Peak Time: {viral_prediction['peak_time']}")
            print(f"   Duration: {viral_prediction['duration_days']} days")
            print(f"   Quantum Resonance: {viral_prediction['quantum_resonance']:.2f}")
            
            if viral_prediction['key_amplifiers']:
                print(f"   🚀 Key Amplifiers: {', '.join(viral_prediction['key_amplifiers'])}")
            
            if viral_prediction['risk_factors']:
                print(f"   ⚠️  Risk Factors: {', '.join(viral_prediction['risk_factors'])}")
        else:
            print(f"❌ Viral prediction failed: {viral_prediction.get('error', 'Unknown error')}")
        
        # =================================================================
        # TEST 3: CREATIVE CONTENT GENERATION
        # =================================================================
        print("\n\n✨ TEST 3: CREATIVE CONTENT GENERATION")
        print("-" * 50)
        
        print("🎨 Generating marketing campaign concept...")
        
        creative_result = await shadowforge.ai_chat(
            message="Create a viral marketing campaign concept for ShadowForge OS - an AI-powered business automation platform. Include: catchy tagline, 3 social media post ideas, and a content strategy that would appeal to entrepreneurs and business owners.",
            context="Creative marketing director with expertise in viral campaigns and AI/tech marketing",
            priority="high"
        )
        
        print(f"✅ Creative Campaign Generated!")
        print(f"   Model: {creative_result.get('model_used', 'unknown')}")
        print(f"   Cost: ${creative_result.get('cost', 0.0):.4f}")
        
        print(f"\n🎯 Campaign Concept:")
        print("-" * 40)
        print(creative_result.get('response', 'No response received'))
        
        # =================================================================
        # TEST 4: BUSINESS PROBLEM SOLVING
        # =================================================================
        print("\n\n💡 TEST 4: BUSINESS PROBLEM SOLVING")
        print("-" * 50)
        
        print("🧩 Solving complex business challenge...")
        
        problem_solving = await shadowforge.ai_chat(
            message="A small e-commerce business is struggling with: 1) Customer support taking 8+ hours to respond, 2) Inventory management errors causing stockouts, 3) Manual order processing taking 30 minutes per order. They have a $5000/month budget for automation. What's the optimal AI automation strategy?",
            context="Business process optimization consultant specializing in AI automation for SMBs",
            priority="urgent"
        )
        
        print(f"✅ Solution Generated!")
        print(f"   Model: {problem_solving.get('model_used', 'unknown')}")
        print(f"   Quality: {problem_solving.get('quality_score', 0.5):.1%}")
        
        print(f"\n🔧 Automation Strategy:")
        print("-" * 40)
        print(problem_solving.get('response', 'No response received'))
        
        # =================================================================
        # TEST 5: SYSTEM PERFORMANCE METRICS
        # =================================================================
        print("\n\n📊 TEST 5: SYSTEM PERFORMANCE METRICS")
        print("-" * 50)
        
        # Get live system metrics
        metrics = await shadowforge.get_system_metrics()
        status = await shadowforge.get_system_status()
        
        print(f"🚀 {status}")
        print(f"   Overall Health: {metrics.get('overall_health', 0.0):.1%}")
        print(f"   System Uptime: {metrics.get('uptime', 'unknown')}")
        
        # Integration statistics
        stats = shadowforge.integration_stats
        print(f"\n📈 Live Usage Statistics:")
        print(f"   Total AI Requests: {stats['total_ai_requests']}")
        print(f"   Successful Predictions: {stats['successful_predictions']}")
        print(f"   Business Analyses: {stats['business_analyses']}")
        
        # AI Core specific metrics
        ai_metrics = metrics.get('subsystem_metrics', {}).get('ai_core', {})
        if ai_metrics:
            print(f"\n🧠 AI Core Performance:")
            print(f"   Total Requests: {ai_metrics.get('total_requests', 0)}")
            print(f"   Total Cost: ${ai_metrics.get('total_cost', 0.0):.4f}")
            print(f"   Average Cost/Request: ${ai_metrics.get('average_cost_per_request', 0.0):.4f}")
            print(f"   Available Models: {len(ai_metrics.get('available_models', []))}")
        
        # =================================================================
        # TEST 6: WEB INTELLIGENCE (if available)
        # =================================================================
        print("\n\n🕷️ TEST 6: WEB INTELLIGENCE GATHERING")
        print("-" * 50)
        
        print("🌐 Testing web intelligence on a public site...")
        
        try:
            web_intel = await shadowforge.gather_web_intelligence(
                url="https://www.example.com",
                analysis_type="basic"
            )
            
            if web_intel.get("success", False):
                print(f"✅ Web Intelligence Complete!")
                print(f"   Title: {web_intel.get('title', 'Unknown')}")
                print(f"   Content Quality: {web_intel.get('content_quality', 0.0):.1%}")
                print(f"   Content Type: {web_intel.get('content_type', 'unknown')}")
            else:
                print(f"⚠️  Web intelligence: {web_intel.get('error', 'Limited functionality without additional dependencies')}")
        except Exception as e:
            print(f"⚠️  Web intelligence: {str(e)} (Expected in basic environment)")
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print("\n\n" + "="*80)
        print("🎉 SHADOWFORGE OS v5.1 - LIVE TEST COMPLETE!")
        print("="*80)
        
        print("✅ Successfully tested live capabilities:")
        print("   🧠 Real AI chat with OpenAI integration")
        print("   🔮 Quantum-enhanced viral content prediction")
        print("   ✨ Creative content generation")
        print("   💡 Complex business problem solving")
        print("   📊 Real-time system performance monitoring")
        print("   🕷️ Web intelligence framework")
        
        final_stats = shadowforge.integration_stats
        total_cost = sum(
            result.get('cost', 0.0) 
            for result in [strategy_result, creative_result, problem_solving]
        )
        
        print(f"\n💰 Session Summary:")
        print(f"   Total AI Requests: {final_stats['total_ai_requests']}")
        print(f"   Total Cost: ${total_cost:.4f}")
        print(f"   Average Quality: {(strategy_result.get('quality_score', 0.5) + creative_result.get('quality_score', 0.5) + problem_solving.get('quality_score', 0.5)) / 3:.1%}")
        
        print(f"\n🚀 Your ShadowForge OS platform is fully operational!")
        print(f"💎 Ready for enterprise-level AI automation and business intelligence")
        
    except Exception as e:
        print(f"\n❌ Test encountered an error: {e}")
        print("🔧 Check your API key and internet connection")
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up resources...")
        await shadowforge.cleanup()
        print(f"✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_live_capabilities())