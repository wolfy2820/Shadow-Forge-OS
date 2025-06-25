#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Comprehensive Test Suite
Test all system components and integration
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime

class ShadowForgeTestSuite:
    """Comprehensive test suite for ShadowForge OS."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.now()
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if details and not passed:
            print(f"       {details}")
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    async def test_imports(self):
        """Test all module imports."""
        print("🧪 Testing Module Imports...")
        
        try:
            from shadowforge import ShadowForgeOS
            self.log_test("Main ShadowForge import", True)
        except Exception as e:
            self.log_test("Main ShadowForge import", False, str(e))
        
        try:
            from quantum_core.entanglement_engine import EntanglementEngine
            self.log_test("Quantum Core imports", True)
        except Exception as e:
            self.log_test("Quantum Core imports", False, str(e))
        
        try:
            from neural_substrate.memory_palace import MemoryPalace
            self.log_test("Neural Substrate imports", True)
        except Exception as e:
            self.log_test("Neural Substrate imports", False, str(e))
        
        try:
            from agent_mesh.agent_coordinator import AgentCoordinator
            self.log_test("Agent Mesh imports", True)
        except Exception as e:
            self.log_test("Agent Mesh imports", False, str(e))
        
        try:
            from prophet_engine.prophet_orchestrator import ProphetOrchestrator
            self.log_test("Prophet Engine imports", True)
        except Exception as e:
            self.log_test("Prophet Engine imports", False, str(e))
        
        try:
            from defi_nexus.defi_orchestrator import DeFiOrchestrator
            self.log_test("DeFi Nexus imports", True)
        except Exception as e:
            self.log_test("DeFi Nexus imports", False, str(e))
    
    async def test_configuration(self):
        """Test configuration system."""
        print("\n📋 Testing Configuration System...")
        
        try:
            from shadowforge import ShadowForgeOS
            os_instance = ShadowForgeOS()
            
            # Test config loading
            has_config = hasattr(os_instance, 'config') and os_instance.config is not None
            self.log_test("Configuration loading", has_config)
            
            # Test config structure
            if has_config:
                required_sections = ['system', 'agents', 'quantum_core', 'neural_substrate']
                all_sections = all(section in os_instance.config for section in required_sections)
                self.log_test("Configuration structure", all_sections)
            
        except Exception as e:
            self.log_test("Configuration system", False, str(e))
    
    async def test_component_initialization(self):
        """Test component initialization."""
        print("\n🔧 Testing Component Initialization...")
        
        try:
            from shadowforge import ShadowForgeOS
            os_instance = ShadowForgeOS()
            
            # Test quantum core components
            quantum_components = [
                'entanglement_engine',
                'superposition_router', 
                'decoherence_shield'
            ]
            
            for component in quantum_components:
                has_component = hasattr(os_instance, component)
                self.log_test(f"Quantum {component}", has_component)
            
            # Test neural substrate components
            neural_components = [
                'memory_palace',
                'dream_forge',
                'wisdom_crystals'
            ]
            
            for component in neural_components:
                has_component = hasattr(os_instance, component)
                self.log_test(f"Neural {component}", has_component)
            
            # Test revenue engines
            revenue_components = [
                'prophet_engine',
                'defi_nexus'
            ]
            
            for component in revenue_components:
                has_component = hasattr(os_instance, component)
                self.log_test(f"Revenue {component}", has_component)
                
        except Exception as e:
            self.log_test("Component initialization", False, str(e))
    
    async def test_quantum_core(self):
        """Test quantum core functionality."""
        print("\n⚛️ Testing Quantum Core...")
        
        try:
            from quantum_core.entanglement_engine import EntanglementEngine
            from quantum_core.superposition_router import SuperpositionRouter
            from quantum_core.decoherence_shield import DecoherenceShield
            
            # Test entanglement engine
            engine = EntanglementEngine()
            await engine.initialize()
            self.log_test("Entanglement Engine initialization", engine.is_initialized)
            
            # Test superposition router
            router = SuperpositionRouter()
            await router.initialize()
            self.log_test("Superposition Router initialization", router.is_initialized)
            
            # Test decoherence shield
            shield = DecoherenceShield()
            await shield.initialize()
            self.log_test("Decoherence Shield initialization", shield.is_initialized)
            
        except Exception as e:
            self.log_test("Quantum Core functionality", False, str(e))
    
    async def test_neural_substrate(self):
        """Test neural substrate functionality."""
        print("\n🧠 Testing Neural Substrate...")
        
        try:
            from neural_substrate.memory_palace import MemoryPalace
            from neural_substrate.dream_forge import DreamForge
            from neural_substrate.wisdom_crystals import WisdomCrystals
            
            # Test memory palace
            palace = MemoryPalace()
            await palace.initialize()
            self.log_test("Memory Palace initialization", palace.is_initialized)
            
            # Test dream forge
            forge = DreamForge()
            await forge.initialize()
            self.log_test("Dream Forge initialization", forge.is_initialized)
            
            # Test wisdom crystals
            crystals = WisdomCrystals()
            await crystals.initialize()
            self.log_test("Wisdom Crystals initialization", crystals.is_initialized)
            
        except Exception as e:
            self.log_test("Neural Substrate functionality", False, str(e))
    
    async def test_agent_mesh(self):
        """Test agent mesh functionality."""
        print("\n🤖 Testing Agent Mesh...")
        
        try:
            from agent_mesh.agent_coordinator import AgentCoordinator
            
            coordinator = AgentCoordinator()
            await coordinator.initialize()
            self.log_test("Agent Coordinator initialization", coordinator.is_initialized)
            
            # Test agent creation
            agents_created = len(coordinator.agents) > 0
            self.log_test("Agent creation", agents_created)
            
        except Exception as e:
            self.log_test("Agent Mesh functionality", False, str(e))
    
    async def test_prophet_engine(self):
        """Test prophet engine functionality."""
        print("\n🔮 Testing Prophet Engine...")
        
        try:
            from prophet_engine.prophet_orchestrator import ProphetOrchestrator
            
            prophet = ProphetOrchestrator()
            await prophet.initialize()
            self.log_test("Prophet Engine initialization", prophet.is_initialized)
            
        except Exception as e:
            self.log_test("Prophet Engine functionality", False, str(e))
    
    async def test_defi_nexus(self):
        """Test DeFi nexus functionality."""
        print("\n💰 Testing DeFi Nexus...")
        
        try:
            from defi_nexus.defi_orchestrator import DeFiOrchestrator
            
            defi = DeFiOrchestrator()
            await defi.initialize()
            self.log_test("DeFi Nexus initialization", defi.is_initialized)
            
        except Exception as e:
            self.log_test("DeFi Nexus functionality", False, str(e))
    
    async def test_integration(self):
        """Test full system integration."""
        print("\n🔄 Testing System Integration...")
        
        try:
            from shadowforge import ShadowForgeOS
            
            # Test full system initialization
            os_instance = ShadowForgeOS()
            await os_instance.initialize()
            self.log_test("Full system initialization", os_instance.is_initialized)
            
            # Test system status
            if os_instance.is_initialized:
                uptime = datetime.now() - os_instance.start_time
                uptime_ok = uptime.total_seconds() > 0
                self.log_test("System uptime tracking", uptime_ok)
                
        except Exception as e:
            self.log_test("System integration", False, str(e))
    
    async def test_cli_interface(self):
        """Test CLI interface."""
        print("\n🖥️ Testing CLI Interface...")
        
        try:
            import subprocess
            
            # Test help command
            result = subprocess.run([
                sys.executable, "shadowforge.py", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            help_works = result.returncode == 0 and "ShadowForge OS" in result.stdout
            self.log_test("CLI help command", help_works)
            
            # Test status command
            result = subprocess.run([
                sys.executable, "shadowforge.py", "--status"
            ], capture_output=True, text=True, timeout=10)
            
            status_works = result.returncode == 0
            self.log_test("CLI status command", status_works)
            
        except Exception as e:
            self.log_test("CLI interface", False, str(e))
    
    async def test_os_shell(self):
        """Test OS shell interface."""
        print("\n🐧 Testing OS Shell Interface...")
        
        try:
            from shadowforge_os import ShadowForgeShell
            
            shell = ShadowForgeShell()
            shell_created = shell is not None
            self.log_test("Shell creation", shell_created)
            
            # Test builtin commands
            has_builtins = len(shell.builtin_commands) > 0
            self.log_test("Builtin commands", has_builtins)
            
            required_commands = ['help', 'init', 'status', 'agents', 'predict', 'trade']
            has_required = all(cmd in shell.builtin_commands for cmd in required_commands)
            self.log_test("Required commands present", has_required)
            
        except Exception as e:
            self.log_test("OS Shell interface", False, str(e))
    
    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        duration = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("🧪 ShadowForge OS v5.1 Test Results")
        print("="*60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Total:  {total}")
        print(f"📈 Success Rate: {(self.passed/total*100):.1f}%")
        print(f"⏱️ Duration: {duration.total_seconds():.2f}s")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! ShadowForge OS is ready!")
        else:
            print(f"\n⚠️ {self.failed} tests failed. Review issues above.")
        
        return self.failed == 0

async def main():
    """Run the complete test suite."""
    print("🧪 ShadowForge OS v5.1 - Comprehensive Test Suite")
    print("="*60)
    
    suite = ShadowForgeTestSuite()
    
    await suite.test_imports()
    await suite.test_configuration()
    await suite.test_component_initialization()
    await suite.test_quantum_core()
    await suite.test_neural_substrate()
    await suite.test_agent_mesh()
    await suite.test_prophet_engine()
    await suite.test_defi_nexus()
    await suite.test_integration()
    await suite.test_cli_interface()
    await suite.test_os_shell()
    
    success = suite.print_summary()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)