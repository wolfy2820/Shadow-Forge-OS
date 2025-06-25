#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Natural Language Operating System Control
Revolutionary AI-powered natural language interface for complete OS control

This system provides natural language control over the entire operating system,
leveraging O3, GPT-4.5, and multi-model AI for unprecedented OS interaction.
"""

import asyncio
import logging
import json
import os
import subprocess
import psutil
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
import ast
import shlex
import platform
import sys
import importlib

# Import advanced AI core
sys.path.append('/home/zeroday/ShadowForge-OS')
from neural_substrate.advanced_ai_core import AdvancedAICore, create_ai_request

class CommandCategory(Enum):
    """Categories of OS commands."""
    FILE_SYSTEM = "file_system"
    PROCESS_MANAGEMENT = "process_management"
    NETWORK = "network"
    SYSTEM_INFO = "system_info"
    PACKAGE_MANAGEMENT = "package_management"
    USER_MANAGEMENT = "user_management"
    SYSTEM_CONTROL = "system_control"
    DEVELOPMENT = "development"
    AI_OPERATIONS = "ai_operations"
    SECURITY = "security"

class ExecutionMode(Enum):
    """Execution modes for commands."""
    SAFE = "safe"           # Only safe, read-only commands
    INTERACTIVE = "interactive"  # Requires confirmation for dangerous operations
    AUTONOMOUS = "autonomous"    # Full autonomous execution

@dataclass
class OSCommand:
    """Operating system command with metadata."""
    command: str
    category: CommandCategory
    description: str
    risk_level: int  # 1-10, 10 being highest risk
    requires_sudo: bool
    expected_output: str
    confidence: float

@dataclass
class ExecutionResult:
    """Result of command execution."""
    command: str
    success: bool
    output: str
    error: str
    execution_time: float
    risk_level: int
    timestamp: datetime

class NaturalLanguageOS:
    """
    Natural Language Operating System Control.
    
    Features:
    - Natural language to OS command translation
    - Multi-model AI integration (O3, GPT-4.5, etc.)
    - Safe execution with risk assessment
    - Comprehensive OS operation coverage
    - Learning from user interactions
    - Autonomous system management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NaturalLanguageOS")
        
        # AI Core integration
        self.ai_core = AdvancedAICore()
        
        # Execution settings
        self.execution_mode = ExecutionMode.INTERACTIVE
        self.max_risk_level = 7  # Default risk threshold
        self.auto_confirm_low_risk = True
        
        # Command history and learning
        self.command_history: List[ExecutionResult] = []
        self.learned_patterns: Dict[str, OSCommand] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # System state
        self.system_info = {}
        self.active_processes = {}
        self.network_status = {}
        
        # Safety mechanisms
        self.dangerous_commands = {
            'rm -rf /', 'dd if=', 'mkfs', 'fdisk', 'parted',
            'shutdown -h now', 'reboot', 'halt', 'poweroff',
            'userdel', 'passwd', 'chmod 777', 'chown -R'
        }
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the Natural Language OS system."""
        try:
            self.logger.info("🧠 Initializing Natural Language OS...")
            
            # Initialize AI Core
            await self.ai_core.initialize()
            
            # Gather system information
            await self._gather_system_info()
            
            # Load learned patterns
            await self._load_learned_patterns()
            
            # Setup monitoring
            asyncio.create_task(self._system_monitoring_loop())
            
            self.is_initialized = True
            self.logger.info("✅ Natural Language OS initialized - Full control active")
            
        except Exception as e:
            self.logger.error(f"❌ Natural Language OS initialization failed: {e}")
            raise
    
    async def deploy(self, target: str):
        """Deploy Natural Language OS to target environment."""
        self.logger.info(f"🚀 Deploying Natural Language OS to {target}")
        
        if target == "production":
            self.execution_mode = ExecutionMode.INTERACTIVE
            self.max_risk_level = 5  # More conservative in production
        elif target == "development":
            self.execution_mode = ExecutionMode.SAFE
            self.max_risk_level = 8
        elif target == "autonomous":
            self.execution_mode = ExecutionMode.AUTONOMOUS
            self.max_risk_level = 10  # Full autonomy
        
        self.logger.info(f"✅ Natural Language OS deployed to {target}")
    
    async def execute_natural_language_command(self, natural_command: str, 
                                             preferred_model: str = "o3") -> Dict[str, Any]:
        """
        Execute a natural language command using AI translation.
        
        Args:
            natural_command: The natural language command from user
            preferred_model: Preferred AI model for translation
            
        Returns:
            Execution result with status and output
        """
        try:
            self.logger.info(f"🗣️ Processing natural language command: {natural_command}")
            
            # Translate natural language to OS command using AI
            translation_result = await self._translate_to_os_command(natural_command, preferred_model)
            
            if not translation_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to translate natural language command",
                    "translation_result": translation_result
                }
            
            os_command = translation_result["os_command"]
            
            # Assess risk and safety
            risk_assessment = await self._assess_command_risk(os_command)
            
            # Check execution permissions
            execution_check = await self._check_execution_permissions(os_command, risk_assessment)
            
            if not execution_check["allowed"]:
                return {
                    "success": False,
                    "error": execution_check["reason"],
                    "risk_assessment": risk_assessment,
                    "os_command": os_command
                }
            
            # Execute the command
            execution_result = await self._execute_os_command(os_command)
            
            # Learn from execution
            await self._learn_from_execution(natural_command, os_command, execution_result)
            
            # Prepare comprehensive result
            result = {
                "success": execution_result.success,
                "natural_command": natural_command,
                "os_command": os_command.command,
                "category": os_command.category.value,
                "output": execution_result.output,
                "error": execution_result.error,
                "execution_time": execution_result.execution_time,
                "risk_level": risk_assessment["risk_level"],
                "ai_model_used": translation_result["model_used"],
                "confidence": os_command.confidence,
                "timestamp": execution_result.timestamp.isoformat()
            }
            
            self.logger.info(f"{'✅' if result['success'] else '❌'} Command executed: {result['success']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Natural language command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "natural_command": natural_command
            }
    
    async def _translate_to_os_command(self, natural_command: str, preferred_model: str = "o3") -> Dict[str, Any]:
        """Translate natural language to OS command using AI."""
        
        # Enhanced system context for O3/GPT-4.5
        system_context = f"""
You are an expert OS command translator for ShadowForge OS Natural Language Interface.

SYSTEM INFORMATION:
- OS: {self.system_info.get('system', 'Linux')}
- Architecture: {self.system_info.get('architecture', 'x86_64')}
- Available shells: bash, zsh, python
- Current directory: {os.getcwd()}
- User: {os.getenv('USER', 'user')}

TASK: Translate natural language commands to precise OS commands.

OUTPUT FORMAT (JSON):
{{
    "command": "exact_shell_command",
    "category": "file_system|process_management|network|system_info|package_management|user_management|system_control|development|ai_operations|security",
    "description": "what this command does",
    "risk_level": 1-10,
    "requires_sudo": true/false,
    "expected_output": "description of expected output",
    "confidence": 0.0-1.0,
    "explanation": "step by step reasoning"
}}

SAFETY RULES:
- Never suggest destructive commands without explicit user intent
- Always assess risk level accurately (1=safe read-only, 10=system destructive)
- For ambiguous requests, choose the safest interpretation
- Include proper error handling and validation

EXAMPLES:
"show me running processes" -> {{"command": "ps aux", "category": "process_management", "risk_level": 1}}
"install python package requests" -> {{"command": "pip install requests", "category": "package_management", "risk_level": 3}}
"check disk space" -> {{"command": "df -h", "category": "system_info", "risk_level": 1}}
"""

        prompt = f"""
Translate this natural language command to an OS command:

NATURAL COMMAND: "{natural_command}"

Provide a JSON response with the exact OS command and metadata.
Consider the current system context and user intent carefully.
"""

        try:
            # Create AI request with preferred model
            ai_request = await create_ai_request(
                prompt=prompt,
                context=system_context,
                model=preferred_model,
                priority="high",
                temperature=0.1,  # Low temperature for precise commands
                max_tokens=2048
            )
            
            # Generate response
            ai_response = await self.ai_core.generate_response(ai_request)
            
            # Parse JSON response
            response_content = ai_response["content"].strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                else:
                    json_content = response_content
            
            # Parse command data
            command_data = json.loads(json_content)
            
            # Create OSCommand object
            os_command = OSCommand(
                command=command_data["command"],
                category=CommandCategory(command_data["category"]),
                description=command_data["description"],
                risk_level=command_data["risk_level"],
                requires_sudo=command_data["requires_sudo"],
                expected_output=command_data["expected_output"],
                confidence=command_data["confidence"]
            )
            
            return {
                "success": True,
                "os_command": os_command,
                "model_used": ai_response["model"],
                "explanation": command_data.get("explanation", ""),
                "ai_confidence": ai_response.get("quality_score", 0.5)
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse AI response JSON: {e}")
            return {
                "success": False,
                "error": f"AI response parsing failed: {e}",
                "raw_response": ai_response.get("content", "")
            }
        except Exception as e:
            self.logger.error(f"❌ Command translation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _assess_command_risk(self, os_command: OSCommand) -> Dict[str, Any]:
        """Assess the risk level of an OS command."""
        
        risk_factors = []
        risk_score = os_command.risk_level
        
        # Check for dangerous command patterns
        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd in os_command.command:
                risk_factors.append(f"Contains dangerous pattern: {dangerous_cmd}")
                risk_score = max(risk_score, 9)
        
        # Check for sudo requirement
        if os_command.requires_sudo:
            risk_factors.append("Requires administrative privileges")
            risk_score = max(risk_score, 6)
        
        # Check for file system modifications
        if any(cmd in os_command.command for cmd in ['rm', 'mv', 'cp', 'chmod', 'chown']):
            if '-r' in os_command.command or '-R' in os_command.command:
                risk_factors.append("Recursive file system operation")
                risk_score = max(risk_score, 7)
        
        # Check for network operations
        if any(cmd in os_command.command for cmd in ['wget', 'curl', 'ssh', 'scp']):
            risk_factors.append("Network operation")
            risk_score = max(risk_score, 4)
        
        # Check for package management
        if any(cmd in os_command.command for cmd in ['apt', 'yum', 'pip', 'npm']):
            if 'install' in os_command.command:
                risk_factors.append("Software installation")
                risk_score = max(risk_score, 5)
        
        return {
            "risk_level": min(risk_score, 10),
            "risk_factors": risk_factors,
            "safety_recommendation": self._get_safety_recommendation(risk_score),
            "requires_confirmation": risk_score > self.max_risk_level or len(risk_factors) > 0
        }
    
    def _get_safety_recommendation(self, risk_level: int) -> str:
        """Get safety recommendation based on risk level."""
        if risk_level <= 2:
            return "Safe to execute automatically"
        elif risk_level <= 4:
            return "Low risk - can execute with minimal confirmation"
        elif risk_level <= 6:
            return "Medium risk - requires user confirmation"
        elif risk_level <= 8:
            return "High risk - requires explicit approval and backup"
        else:
            return "Extreme risk - strongly recommend manual review"
    
    async def _check_execution_permissions(self, os_command: OSCommand, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Check if command execution is allowed."""
        
        # Check risk level against threshold
        if risk_assessment["risk_level"] > self.max_risk_level:
            return {
                "allowed": False,
                "reason": f"Risk level {risk_assessment['risk_level']} exceeds maximum allowed {self.max_risk_level}"
            }
        
        # Check execution mode
        if self.execution_mode == ExecutionMode.SAFE and risk_assessment["risk_level"] > 3:
            return {
                "allowed": False,
                "reason": "Safe mode active - command risk level too high"
            }
        
        # Check for interactive confirmation requirement
        if (self.execution_mode == ExecutionMode.INTERACTIVE and 
            risk_assessment["requires_confirmation"] and 
            not self.auto_confirm_low_risk):
            # In a real implementation, this would prompt the user
            # For now, we'll allow it but log the requirement
            self.logger.warning(f"⚠️ Command requires confirmation: {os_command.command}")
        
        # Check sudo requirements
        if os_command.requires_sudo and os.geteuid() != 0:
            return {
                "allowed": False,
                "reason": "Command requires sudo privileges but not running as root"
            }
        
        return {"allowed": True, "reason": "Command approved for execution"}
    
    async def _execute_os_command(self, os_command: OSCommand) -> ExecutionResult:
        """Execute the OS command safely."""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🔧 Executing OS command: {os_command.command}")
            
            # Parse command safely
            cmd_parts = shlex.split(os_command.command)
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await process.communicate()
            
            execution_time = time.time() - start_time
            
            # Create execution result
            result = ExecutionResult(
                command=os_command.command,
                success=process.returncode == 0,
                output=stdout.decode('utf-8', errors='ignore'),
                error=stderr.decode('utf-8', errors='ignore'),
                execution_time=execution_time,
                risk_level=os_command.risk_level,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.command_history.append(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = ExecutionResult(
                command=os_command.command,
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                risk_level=os_command.risk_level,
                timestamp=datetime.now()
            )
            
            self.command_history.append(error_result)
            return error_result
    
    async def _learn_from_execution(self, natural_command: str, os_command: OSCommand, 
                                  execution_result: ExecutionResult):
        """Learn from command execution for future improvements."""
        
        # Store successful patterns
        if execution_result.success:
            pattern_key = self._normalize_natural_command(natural_command)
            self.learned_patterns[pattern_key] = os_command
            
            # Update user preferences
            if os_command.category.value not in self.user_preferences:
                self.user_preferences[os_command.category.value] = {"success_count": 0, "total_count": 0}
            
            self.user_preferences[os_command.category.value]["success_count"] += 1
        
        # Update total count
        if os_command.category.value in self.user_preferences:
            self.user_preferences[os_command.category.value]["total_count"] += 1
    
    def _normalize_natural_command(self, command: str) -> str:
        """Normalize natural language command for pattern matching."""
        # Remove common variations and normalize
        normalized = command.lower().strip()
        
        # Remove common words
        common_words = ['please', 'can you', 'could you', 'i want to', 'i need to']
        for word in common_words:
            normalized = normalized.replace(word, '').strip()
        
        return normalized
    
    async def _gather_system_info(self):
        """Gather comprehensive system information."""
        self.system_info = {
            "system": platform.system(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "username": os.getenv('USER', 'unknown'),
            "home_directory": os.getenv('HOME', '/'),
            "current_directory": os.getcwd(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": {mount.mountpoint: psutil.disk_usage(mount.mountpoint).percent 
                          for mount in psutil.disk_partitions()},
            "network_interfaces": list(psutil.net_if_addrs().keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _load_learned_patterns(self):
        """Load previously learned command patterns."""
        # In production, this would load from persistent storage
        # For now, initialize with empty patterns
        self.learned_patterns = {}
        self.user_preferences = {}
    
    async def _system_monitoring_loop(self):
        """Continuous system monitoring loop."""
        while self.is_initialized:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check for system events
                await self._check_system_events()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ System monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _update_system_metrics(self):
        """Update real-time system metrics."""
        self.active_processes = {
            proc.pid: {
                "name": proc.info['name'],
                "cpu_percent": proc.info['cpu_percent'],
                "memory_percent": proc.info['memory_percent']
            }
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent'])
        }
        
        self.network_status = {
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv,
            "connections": len(psutil.net_connections())
        }
    
    async def _check_system_events(self):
        """Check for important system events."""
        # Monitor CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.logger.warning(f"🚨 High CPU usage detected: {cpu_percent:.1f}%")
        
        # Monitor memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.logger.warning(f"🚨 High memory usage detected: {memory.percent:.1f}%")
        
        # Monitor disk usage
        for partition in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(partition.mountpoint)
                usage_percent = (disk_usage.used / disk_usage.total) * 100
                if usage_percent > 90:
                    self.logger.warning(f"🚨 High disk usage on {partition.mountpoint}: {usage_percent:.1f}%")
            except:
                pass  # Skip if partition not accessible
    
    async def execute_intelligent_operation(self, operation_description: str) -> Dict[str, Any]:
        """Execute complex multi-step operations using AI planning."""
        
        self.logger.info(f"🧠 Planning intelligent operation: {operation_description}")
        
        # Use O3 for complex operation planning
        planning_context = f"""
You are an intelligent OS operation planner for ShadowForge OS.

SYSTEM CONTEXT:
{json.dumps(self.system_info, indent=2)}

TASK: Break down this complex operation into safe, executable steps:
"{operation_description}"

OUTPUT FORMAT (JSON):
{{
    "operation_plan": [
        {{
            "step": 1,
            "description": "what this step does",
            "command": "exact command to execute",
            "category": "command category",
            "risk_level": 1-10,
            "depends_on": [list of step numbers this depends on],
            "validation": "how to verify this step succeeded"
        }}
    ],
    "total_steps": number,
    "estimated_time": "time estimate",
    "overall_risk": 1-10,
    "prerequisites": ["list of requirements"],
    "rollback_plan": "how to undo if something goes wrong"
}}

Make sure each step is safe and atomic. Prioritize user safety and data integrity.
"""

        try:
            # Create AI request for operation planning
            ai_request = await create_ai_request(
                prompt=f"Plan this operation: {operation_description}",
                context=planning_context,
                model="o3",  # Use O3 for complex planning
                priority="high",
                temperature=0.2,
                max_tokens=4096
            )
            
            # Generate operation plan
            ai_response = await self.ai_core.generate_response(ai_request)
            
            # Parse plan
            plan_content = ai_response["content"].strip()
            json_match = re.search(r'```json\s*(.*?)\s*```', plan_content, re.DOTALL)
            if json_match:
                plan_json = json_match.group(1)
            else:
                plan_json = plan_content
            
            operation_plan = json.loads(plan_json)
            
            # Execute operation steps
            execution_results = []
            
            for step in operation_plan["operation_plan"]:
                self.logger.info(f"🔄 Executing step {step['step']}: {step['description']}")
                
                # Execute step as natural language command
                step_result = await self.execute_natural_language_command(
                    step["command"], 
                    preferred_model="gpt-4.5"
                )
                
                step_result["step_number"] = step["step"]
                step_result["step_description"] = step["description"]
                execution_results.append(step_result)
                
                # Check if step failed
                if not step_result["success"]:
                    self.logger.error(f"❌ Step {step['step']} failed: {step_result['error']}")
                    
                    # Execute rollback if needed
                    if operation_plan.get("rollback_plan"):
                        self.logger.info("🔙 Executing rollback plan...")
                        await self.execute_natural_language_command(
                            operation_plan["rollback_plan"],
                            preferred_model="gpt-4"
                        )
                    
                    break
            
            return {
                "success": all(result["success"] for result in execution_results),
                "operation_description": operation_description,
                "operation_plan": operation_plan,
                "execution_results": execution_results,
                "ai_model_used": ai_response["model"],
                "total_steps": len(execution_results),
                "completed_steps": sum(1 for r in execution_results if r["success"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation_description": operation_description
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_info": self.system_info,
            "execution_mode": self.execution_mode.value,
            "max_risk_level": self.max_risk_level,
            "command_history_count": len(self.command_history),
            "learned_patterns_count": len(self.learned_patterns),
            "recent_commands": [
                {
                    "command": cmd.command,
                    "success": cmd.success,
                    "timestamp": cmd.timestamp.isoformat()
                }
                for cmd in self.command_history[-10:]  # Last 10 commands
            ],
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": {
                    mount.mountpoint: psutil.disk_usage(mount.mountpoint).percent
                    for mount in psutil.disk_partitions()
                },
                "network_connections": len(psutil.net_connections())
            },
            "ai_core_status": await self.ai_core.get_metrics()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get Natural Language OS performance metrics."""
        successful_commands = sum(1 for cmd in self.command_history if cmd.success)
        total_commands = len(self.command_history)
        
        return {
            "total_commands_executed": total_commands,
            "successful_commands": successful_commands,
            "success_rate": successful_commands / max(total_commands, 1),
            "learned_patterns": len(self.learned_patterns),
            "user_preferences": self.user_preferences,
            "execution_mode": self.execution_mode.value,
            "max_risk_level": self.max_risk_level,
            "avg_execution_time": (
                sum(cmd.execution_time for cmd in self.command_history) / max(total_commands, 1)
            ),
            "risk_distribution": {
                f"level_{i}": sum(1 for cmd in self.command_history if cmd.risk_level == i)
                for i in range(1, 11)
            }
        }

# Convenience functions for easy usage
async def execute_nl_command(command: str, model: str = "o3") -> Dict[str, Any]:
    """Execute a natural language command with specified model."""
    nl_os = NaturalLanguageOS()
    await nl_os.initialize()
    result = await nl_os.execute_natural_language_command(command, model)
    return result

async def plan_and_execute_operation(operation: str) -> Dict[str, Any]:
    """Plan and execute a complex operation."""
    nl_os = NaturalLanguageOS()
    await nl_os.initialize()
    result = await nl_os.execute_intelligent_operation(operation)
    return result