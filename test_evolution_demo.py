#!/usr/bin/env python3
"""
ShadowForge OS Evolution Demo
Quick demonstration of the self-evolving monetization system
"""

import asyncio
import logging
from shadowforge_evolution import ShadowForgeEvolution

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def demo_evolution():
    """Demonstrate the evolution system with simulated revenue growth."""
    
    print("\n" + "🌟" * 30)
    print("SHADOWFORGE OS v5.1 EVOLUTION DEMO")
    print("Self-Evolving AI Business Operating System")
    print("🌟" * 30)
    
    # Initialize the evolution engine
    shadowforge = ShadowForgeEvolution()
    
    try:
        await shadowforge.initialize()
        
        print("\n✅ Evolution Engine Initialized!")
        print("🚀 Ready to demonstrate automatic evolution...")
        
        # Demonstrate evolution through revenue milestones
        milestones = [
            (500, "Initial Revenue", "First sales from content creation"),
            (1000, "Crypto Milestone", "Budget reached for first cryptocurrency"),
            (5000, "Growth Phase", "Upgrading to premium video tools"),
            (10000, "Scale Phase", "Launching DeFi protocols"),
            (50000, "Enterprise Phase", "Unlimited generation capabilities"),
            (100000, "Empire Phase", "Market dominance begins")
        ]
        
        for target_amount, milestone_name, description in milestones:
            current_budget = shadowforge.current_budget
            revenue_increase = target_amount - current_budget
            
            if revenue_increase > 0:
                print(f"\n🎯 {milestone_name}: Adding ${revenue_increase:,.2f}")
                print(f"   📝 {description}")
                
                # Update revenue and trigger evolution
                await shadowforge.update_revenue(revenue_increase, milestone_name.lower().replace(" ", "_"))
                
                # Get evolution status
                status = await shadowforge.get_empire_status()
                
                print(f"   💰 New Budget: ${status['empire_overview']['current_budget']:,.2f}")
                print(f"   🏆 Evolution Tier: {status['empire_overview']['evolution_tier']}")
                print(f"   📈 Empire Value: ${status['empire_overview']['empire_value']:,.2f}")
                
                # Show what evolved at this milestone
                if target_amount >= 1000 and shadowforge.cryptocurrencies_created == 0:
                    print(f"   🪙 EVOLUTION: Cryptocurrency creation unlocked!")
                if target_amount >= 10000:
                    print(f"   🎬 EVOLUTION: Enterprise video generation unlocked!")
                if target_amount >= 100000:
                    print(f"   🏭 EVOLUTION: Market dominance capabilities unlocked!")
                
                await asyncio.sleep(0.5)  # Brief pause for demonstration
        
        # Show final comprehensive status
        final_status = await shadowforge.get_empire_status()
        
        print("\n" + "=" * 60)
        print("🏆 FINAL EMPIRE STATUS")
        print("=" * 60)
        
        empire = final_status['empire_overview']
        print(f"💰 Total Budget: ${empire['current_budget']:,.2f}")
        print(f"📊 Revenue Generated: ${empire['total_revenue_generated']:,.2f}")
        print(f"🚀 Evolution Tier: {empire['evolution_tier']}")
        print(f"💎 Empire Value: ${empire['empire_value']:,.2f}")
        print(f"🔄 Active Projects: {empire['active_projects']}")
        print(f"💸 Revenue Streams: {empire['revenue_streams']}")
        
        print(f"\n🪙 Cryptocurrency Metrics:")
        crypto = final_status['crypto_metrics']
        print(f"   Created: {crypto['total_tokens_created']}")
        print(f"   Success Rate: {crypto['success_rate']:.1%}")
        print(f"   Expected ROI: {crypto['expected_roi']:.1%}")
        
        print(f"\n🎬 Video Generation:")
        video = final_status['video_status']
        print(f"   Current Tier: {video['current_tier']}")
        print(f"   Videos Generated: {video['videos_generated']}")
        print(f"   Generation Capacity: {video['generation_capacity']:,}")
        
        print(f"\n🧠 Neural Interface Status:")
        neural = final_status['neural_status']
        for system, metrics in neural.items():
            print(f"   {system}: Active and operational")
        
        print(f"\n🎉 SUCCESS! Your ShadowForge OS has evolved into a digital empire!")
        print(f"🚀 The system continues evolving automatically as revenue grows...")
        print(f"💫 Next milestone: ${final_status['next_milestones'][0] if final_status.get('next_milestones') else 'No limits!'}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_evolution())