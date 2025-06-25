"""
Agent Coordinator - Central orchestration system for the agent mesh

Manages the quantum-entangled network of 7 specialized AI agents,
coordinating their activities to achieve maximum synergy and efficiency.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI

# Agent imports
from .oracle.oracle_agent import OracleAgent
from .alchemist.alchemist_agent import AlchemistAgent
from .architect.architect_agent import ArchitectAgent
from .guardian.guardian_agent import GuardianAgent
from .merchant.merchant_agent import MerchantAgent
from .scholar.scholar_agent import ScholarAgent
from .diplomat.diplomat_agent import DiplomatAgent

class AgentRole(Enum):
    """Specialized agent roles in the mesh."""
    ORACLE = "oracle"
    ALCHEMIST = "alchemist"
    ARCHITECT = "architect"
    GUARDIAN = "guardian"
    MERCHANT = "merchant"
    SCHOLAR = "scholar"
    DIPLOMAT = "diplomat"

@dataclass
class AgentMetrics:
    """Performance metrics for individual agents."""
    agent_id: str
    tasks_completed: int
    success_rate: float
    average_response_time: float
    cpu_utilization: float
    memory_usage: float
    last_activity: datetime

class AgentCoordinator:
    """
    Central coordinator for the ShadowForge agent mesh.
    
    Orchestrates 7 specialized AI agents in quantum entanglement,
    enabling them to work together as a unified digital organism
    focused on content creation and economic dominance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Agent instances
        self.agents: Dict[str, Any] = {}
        self.crew: Optional[Crew] = None
        
        # Coordination state
        self.is_initialized = False
        self.is_active = False
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        
        # Performance tracking
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.total_tasks_processed = 0
        self.coordination_efficiency = 0.0
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model for agents."""
        try:
            # Try Ollama first (local)
            return Ollama(model="qwen2.5-coder:latest")
        except Exception:
            try:
                # Fallback to OpenAI
                return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
            except Exception as e:
                self.logger.warning(f"⚠️ LLM initialization failed: {e}")
                return None
    
    async def initialize(self, agent_count: int = 7):
        """Initialize the agent mesh with specified number of agents."""
        try:
            self.logger.info(f"🤖 Initializing Agent Mesh with {agent_count} agents...")
            
            # Initialize specialized agents
            await self._initialize_agents()
            
            # Create CrewAI crew for coordination
            await self._create_crew()
            
            # Start monitoring and coordination loops
            asyncio.create_task(self._coordination_loop())
            asyncio.create_task(self._metrics_collector())
            
            self.is_initialized = True
            self.logger.info("✅ Agent Mesh initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Agent Mesh initialization failed: {e}")
            raise
    
    async def deploy(self, target: str):
        """Deploy agent mesh to target environment."""
        self.logger.info(f"🚀 Deploying Agent Mesh to {target}")
        
        # Deploy all agents
        deployment_tasks = []
        for agent in self.agents.values():
            if hasattr(agent, 'deploy'):
                deployment_tasks.append(agent.deploy(target))
        
        if deployment_tasks:
            await asyncio.gather(*deployment_tasks)
        
        self.is_active = True
        self.logger.info(f"✅ Agent Mesh deployed to {target}")
    
    async def execute_task(self, task_description: str, priority: str = "medium",
                         required_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a task using the most appropriate agents.
        
        Args:
            task_description: Description of the task to execute
            priority: Task priority (low, medium, high, critical)
            required_agents: Specific agents required for the task
            
        Returns:
            Dict containing task results and execution metadata
        """
        try:
            self.logger.info(f"📋 Executing task: {task_description[:100]}...")
            
            # Analyze task requirements
            task_analysis = await self._analyze_task(task_description)
            
            # Select optimal agents
            selected_agents = await self._select_agents(task_analysis, required_agents)
            
            # Create CrewAI task
            crew_task = Task(
                description=task_description,
                agent=selected_agents[0] if selected_agents else None,
                expected_output="Comprehensive task completion with detailed results"
            )
            
            # Execute task with selected agents
            start_time = datetime.now()
            result = await self._execute_with_crew(crew_task, selected_agents)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.total_tasks_processed += 1
            await self._update_agent_metrics(selected_agents, execution_time, True)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agents_used": [agent.role for agent in selected_agents],
                "task_id": f"task_{self.total_tasks_processed}"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_description": task_description
            }
    
    async def analyze_and_improve(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze system performance and suggest improvements.
        
        Args:
            metrics: Current system performance metrics
            
        Returns:
            List of improvement suggestions
        """
        try:
            improvements = []
            
            # Analyze agent performance
            for agent_id, agent_metrics in self.agent_metrics.items():
                if agent_metrics.success_rate < 0.8:
                    improvements.append({
                        "component": "agent_mesh",
                        "agent": agent_id,
                        "change": {
                            "type": "performance_optimization",
                            "target": "success_rate",
                            "current": agent_metrics.success_rate,
                            "target_value": 0.9
                        }
                    })
                
                if agent_metrics.average_response_time > 5.0:
                    improvements.append({
                        "component": "agent_mesh",
                        "agent": agent_id,
                        "change": {
                            "type": "latency_optimization",
                            "target": "response_time",
                            "current": agent_metrics.average_response_time,
                            "target_value": 3.0
                        }
                    })
            
            # Analyze coordination efficiency
            if self.coordination_efficiency < 0.85:
                improvements.append({
                    "component": "agent_mesh",
                    "change": {
                        "type": "coordination_optimization",
                        "target": "efficiency",
                        "current": self.coordination_efficiency,
                        "target_value": 0.9
                    }
                })
            
            self.logger.info(f"🔍 Generated {len(improvements)} improvement suggestions")
            return improvements
            
        except Exception as e:
            self.logger.error(f"❌ Analysis and improvement failed: {e}")
            return []
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents."""
        status = {
            "initialized": self.is_initialized,
            "active": self.is_active,
            "total_agents": len(self.agents),
            "tasks_processed": self.total_tasks_processed,
            "coordination_efficiency": self.coordination_efficiency,
            "agents": {}
        }
        
        for agent_id, metrics in self.agent_metrics.items():
            status["agents"][agent_id] = {
                "tasks_completed": metrics.tasks_completed,
                "success_rate": metrics.success_rate,
                "response_time": metrics.average_response_time,
                "cpu_usage": metrics.cpu_utilization,
                "memory_usage": metrics.memory_usage,
                "last_active": metrics.last_activity.isoformat()
            }
        
        return status
    
    async def _initialize_agents(self):
        """Initialize all specialized agents."""
        self.logger.debug("🔧 Initializing specialized agents...")
        
        # Initialize Oracle Agent - Market prediction & trend anticipation
        self.agents["oracle"] = OracleAgent(llm=self.llm)
        await self.agents["oracle"].initialize()
        
        # Initialize Alchemist Agent - Content transformation & fusion
        self.agents["alchemist"] = AlchemistAgent(llm=self.llm)
        await self.agents["alchemist"].initialize()
        
        # Initialize Architect Agent - System design & evolution
        self.agents["architect"] = ArchitectAgent(llm=self.llm)
        await self.agents["architect"].initialize()
        
        # Initialize Guardian Agent - Security & compliance enforcement
        self.agents["guardian"] = GuardianAgent(llm=self.llm)
        await self.agents["guardian"].initialize()
        
        # Initialize Merchant Agent - Revenue optimization & scaling
        self.agents["merchant"] = MerchantAgent(llm=self.llm)
        await self.agents["merchant"].initialize()
        
        # Initialize Scholar Agent - Self-improvement & learning
        self.agents["scholar"] = ScholarAgent(llm=self.llm)
        await self.agents["scholar"].initialize()
        
        # Initialize Diplomat Agent - User interaction & negotiation
        self.agents["diplomat"] = DiplomatAgent(llm=self.llm)
        await self.agents["diplomat"].initialize()
        
        # Initialize metrics for all agents
        for agent_id in self.agents.keys():
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                tasks_completed=0,
                success_rate=1.0,
                average_response_time=0.0,
                cpu_utilization=0.0,
                memory_usage=0.0,
                last_activity=datetime.now()
            )
        
        self.logger.debug(f"✅ {len(self.agents)} specialized agents initialized")
    
    async def _create_crew(self):
        """Create CrewAI crew for agent coordination."""
        if not self.agents:
            raise ValueError("Agents must be initialized before creating crew")
        
        # Convert specialized agents to CrewAI agents
        crew_agents = []
        for agent_name, specialized_agent in self.agents.items():
            if hasattr(specialized_agent, 'get_crewai_agent'):
                crew_agent = specialized_agent.get_crewai_agent()
                crew_agents.append(crew_agent)
        
        # Create the crew with hierarchical process
        self.crew = Crew(
            agents=crew_agents,
            tasks=[],  # Tasks will be added dynamically
            process=Process.hierarchical,
            verbose=True,
            memory=True
        )
        
        self.logger.debug("🚢 CrewAI crew created for coordination")
    
    async def _coordination_loop(self):
        """Main coordination loop for managing agent activities."""
        while self.is_initialized:
            try:
                # Process pending tasks
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    await self._dispatch_task(task)
                
                # Update coordination efficiency
                await self._calculate_coordination_efficiency()
                
                # Sleep between coordination cycles
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"❌ Coordination loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _metrics_collector(self):
        """Collect performance metrics from all agents."""
        while self.is_initialized:
            try:
                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'get_metrics'):
                        metrics = await agent.get_metrics()
                        await self._update_agent_metrics_from_data(agent_id, metrics)
                
                await asyncio.sleep(30.0)  # Collect metrics every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(60.0)
    
    async def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze task requirements to determine optimal agent assignment."""
        analysis = {
            "complexity": "medium",
            "domain": "general",
            "required_skills": [],
            "estimated_time": 5.0,
            "priority": "medium"
        }
        
        # Simple keyword-based analysis (can be enhanced with ML)
        description_lower = task_description.lower()
        
        if any(word in description_lower for word in ["predict", "forecast", "trend", "market"]):
            analysis["domain"] = "prediction"
            analysis["required_skills"].append("oracle")
        
        if any(word in description_lower for word in ["create", "generate", "content", "write"]):
            analysis["domain"] = "content"
            analysis["required_skills"].append("alchemist")
        
        if any(word in description_lower for word in ["design", "architect", "system", "structure"]):
            analysis["domain"] = "architecture"
            analysis["required_skills"].append("architect")
        
        if any(word in description_lower for word in ["secure", "protect", "validate", "compliance"]):
            analysis["domain"] = "security"
            analysis["required_skills"].append("guardian")
        
        if any(word in description_lower for word in ["revenue", "money", "profit", "financial"]):
            analysis["domain"] = "finance"
            analysis["required_skills"].append("merchant")
        
        if any(word in description_lower for word in ["learn", "improve", "optimize", "research"]):
            analysis["domain"] = "learning"
            analysis["required_skills"].append("scholar")
        
        if any(word in description_lower for word in ["communicate", "user", "interface", "explain"]):
            analysis["domain"] = "communication"
            analysis["required_skills"].append("diplomat")
        
        return analysis
    
    async def _select_agents(self, task_analysis: Dict[str, Any], 
                           required_agents: Optional[List[str]] = None) -> List[Any]:
        """Select optimal agents for task execution."""
        if required_agents:
            return [self.agents[agent_id] for agent_id in required_agents if agent_id in self.agents]
        
        selected = []
        required_skills = task_analysis.get("required_skills", [])
        
        # Select agents based on required skills
        for skill in required_skills:
            if skill in self.agents:
                selected.append(self.agents[skill])
        
        # If no specific skills required, use diplomat as default
        if not selected:
            selected.append(self.agents["diplomat"])
        
        return selected
    
    async def _execute_with_crew(self, task: Task, selected_agents: List[Any]) -> Any:
        """Execute task using CrewAI crew."""
        if not self.crew:
            raise ValueError("Crew not initialized")
        
        # Add task to crew
        self.crew.tasks = [task]
        
        # Execute with crew
        result = self.crew.kickoff()
        
        return result
    
    async def _dispatch_task(self, task: Task):
        """Dispatch a task to appropriate agents."""
        # Implementation for task dispatching
        pass
    
    async def _calculate_coordination_efficiency(self):
        """Calculate current coordination efficiency."""
        if not self.agent_metrics:
            self.coordination_efficiency = 0.0
            return
        
        # Simple efficiency calculation based on success rates
        total_success_rate = sum(metrics.success_rate for metrics in self.agent_metrics.values())
        self.coordination_efficiency = total_success_rate / len(self.agent_metrics)
    
    async def _update_agent_metrics(self, agents: List[Any], execution_time: float, 
                                  success: bool):
        """Update metrics for agents involved in task execution."""
        for agent in agents:
            if hasattr(agent, 'agent_id'):
                agent_id = agent.agent_id
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    metrics.tasks_completed += 1
                    metrics.last_activity = datetime.now()
                    
                    # Update success rate
                    if success:
                        metrics.success_rate = (metrics.success_rate * (metrics.tasks_completed - 1) + 1.0) / metrics.tasks_completed
                    else:
                        metrics.success_rate = (metrics.success_rate * (metrics.tasks_completed - 1)) / metrics.tasks_completed
                    
                    # Update response time
                    if metrics.average_response_time == 0.0:
                        metrics.average_response_time = execution_time
                    else:
                        metrics.average_response_time = (metrics.average_response_time + execution_time) / 2
    
    async def _update_agent_metrics_from_data(self, agent_id: str, metrics_data: Dict[str, Any]):
        """Update agent metrics from collected data."""
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.cpu_utilization = metrics_data.get("cpu_usage", 0.0)
            metrics.memory_usage = metrics_data.get("memory_usage", 0.0)