#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Advanced Agent Optimization System
Self-improving AI agents with evolutionary algorithms and performance optimization
"""

import asyncio
import json
import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import statistics

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an AI agent."""
    agent_id: str
    task_success_rate: float
    response_quality_score: float
    processing_speed: float
    resource_efficiency: float
    user_satisfaction: float
    error_rate: float
    learning_rate: float
    adaptability_score: float
    collaboration_score: float
    innovation_score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationGenome:
    """Genetic representation of agent optimization parameters."""
    agent_id: str
    parameters: Dict[str, float]
    fitness_score: float
    generation: int
    parent_genomes: List[str] = field(default_factory=list)
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class LearningPattern:
    """Pattern learned by an agent from experience."""
    pattern_id: str
    agent_id: str
    context: str
    action: str
    outcome: str
    success_rate: float
    confidence: float
    usage_count: int
    last_used: datetime
    effectiveness_score: float
    
class AgentOptimizationStrategy(ABC):
    """Abstract base class for agent optimization strategies."""
    
    @abstractmethod
    async def optimize_agent(self, agent_id: str, metrics: AgentPerformanceMetrics, 
                           historical_data: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Optimize agent based on performance metrics."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this optimization strategy."""
        pass

class GeneticOptimizationStrategy(AgentOptimizationStrategy):
    """Genetic algorithm-based optimization strategy."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.7
        self.elite_percentage = 0.2
        
    async def optimize_agent(self, agent_id: str, metrics: AgentPerformanceMetrics,
                           historical_data: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Optimize agent using genetic algorithms."""
        
        # Create initial population if needed
        population = await self._create_or_evolve_population(agent_id, metrics, historical_data)
        
        # Evaluate fitness of each genome
        fitness_scores = await self._evaluate_population_fitness(population, metrics)
        
        # Select best genomes
        elite_genomes = await self._select_elite_genomes(population, fitness_scores)
        
        # Create next generation through crossover and mutation
        next_generation = await self._create_next_generation(elite_genomes)
        
        # Return optimization recommendations
        best_genome = max(population, key=lambda g: g.fitness_score)
        
        return {
            "strategy": "genetic",
            "best_parameters": best_genome.parameters,
            "fitness_score": best_genome.fitness_score,
            "generation": best_genome.generation,
            "population_diversity": self._calculate_population_diversity(population),
            "improvement_potential": self._calculate_improvement_potential(fitness_scores),
            "next_generation_genomes": [g.parameters for g in next_generation[:5]]
        }
    
    async def _create_or_evolve_population(self, agent_id: str, metrics: AgentPerformanceMetrics,
                                         historical_data: List[AgentPerformanceMetrics]) -> List[OptimizationGenome]:
        """Create initial population or evolve existing one."""
        population = []
        
        # Extract parameter ranges from historical data
        parameter_ranges = self._analyze_parameter_ranges(historical_data)
        
        for i in range(self.population_size):
            # Generate random parameters within reasonable ranges
            parameters = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                parameters[param_name] = random.uniform(min_val, max_val)
            
            # Calculate fitness based on current metrics
            fitness = self._calculate_fitness(parameters, metrics)
            
            genome = OptimizationGenome(
                agent_id=agent_id,
                parameters=parameters,
                fitness_score=fitness,
                generation=1,
                mutation_rate=self.mutation_rate
            )
            
            population.append(genome)
        
        return population
    
    def _analyze_parameter_ranges(self, historical_data: List[AgentPerformanceMetrics]) -> Dict[str, Tuple[float, float]]:
        """Analyze historical data to determine parameter ranges."""
        # Define optimization parameters and their ranges
        return {
            "learning_rate": (0.001, 0.1),
            "temperature": (0.1, 1.0),
            "creativity_factor": (0.0, 1.0),
            "exploration_rate": (0.0, 0.5),
            "collaboration_weight": (0.0, 1.0),
            "risk_tolerance": (0.0, 1.0),
            "response_depth": (0.1, 1.0),
            "context_window": (0.5, 2.0),
            "specialization_factor": (0.0, 1.0),
            "adaptation_speed": (0.1, 1.0)
        }
    
    def _calculate_fitness(self, parameters: Dict[str, float], metrics: AgentPerformanceMetrics) -> float:
        """Calculate fitness score for a set of parameters."""
        # Weighted combination of performance metrics
        weights = {
            "task_success_rate": 0.25,
            "response_quality_score": 0.20,
            "processing_speed": 0.15,
            "resource_efficiency": 0.15,
            "user_satisfaction": 0.15,
            "innovation_score": 0.10
        }
        
        fitness = 0.0
        for metric_name, weight in weights.items():
            metric_value = getattr(metrics, metric_name, 0.5)
            fitness += weight * metric_value
        
        # Apply parameter-based modifiers
        fitness *= (1.0 + parameters.get("adaptation_speed", 0.5) * 0.1)
        fitness *= (1.0 - parameters.get("error_rate", 0.1) * 0.2)
        
        return max(0.0, min(1.0, fitness))
    
    async def _evaluate_population_fitness(self, population: List[OptimizationGenome],
                                         metrics: AgentPerformanceMetrics) -> List[float]:
        """Evaluate fitness for entire population."""
        fitness_scores = []
        
        for genome in population:
            # Simulate performance with these parameters
            simulated_performance = await self._simulate_performance(genome.parameters, metrics)
            fitness = self._calculate_fitness(genome.parameters, simulated_performance)
            genome.fitness_score = fitness
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    async def _simulate_performance(self, parameters: Dict[str, float],
                                  base_metrics: AgentPerformanceMetrics) -> AgentPerformanceMetrics:
        """Simulate performance with given parameters."""
        # Create modified metrics based on parameters
        return AgentPerformanceMetrics(
            agent_id=base_metrics.agent_id,
            task_success_rate=min(1.0, base_metrics.task_success_rate * (1.0 + parameters.get("learning_rate", 0.01))),
            response_quality_score=min(1.0, base_metrics.response_quality_score * (1.0 + parameters.get("creativity_factor", 0.5) * 0.1)),
            processing_speed=base_metrics.processing_speed * (1.0 + parameters.get("adaptation_speed", 0.5) * 0.2),
            resource_efficiency=min(1.0, base_metrics.resource_efficiency * (1.0 + parameters.get("risk_tolerance", 0.3) * 0.1)),
            user_satisfaction=min(1.0, base_metrics.user_satisfaction * (1.0 + parameters.get("response_depth", 0.7) * 0.15)),
            error_rate=max(0.0, base_metrics.error_rate * (1.0 - parameters.get("specialization_factor", 0.5) * 0.1)),
            learning_rate=base_metrics.learning_rate * (1.0 + parameters.get("learning_rate", 0.01)),
            adaptability_score=min(1.0, base_metrics.adaptability_score * (1.0 + parameters.get("exploration_rate", 0.2))),
            collaboration_score=min(1.0, base_metrics.collaboration_score * (1.0 + parameters.get("collaboration_weight", 0.5) * 0.2)),
            innovation_score=min(1.0, base_metrics.innovation_score * (1.0 + parameters.get("creativity_factor", 0.5) * 0.3))
        )
    
    async def _select_elite_genomes(self, population: List[OptimizationGenome],
                                  fitness_scores: List[float]) -> List[OptimizationGenome]:
        """Select elite genomes for breeding."""
        elite_count = max(1, int(len(population) * self.elite_percentage))
        
        # Sort by fitness and select top performers
        sorted_population = sorted(population, key=lambda g: g.fitness_score, reverse=True)
        elite_genomes = sorted_population[:elite_count]
        
        return elite_genomes
    
    async def _create_next_generation(self, elite_genomes: List[OptimizationGenome]) -> List[OptimizationGenome]:
        """Create next generation through crossover and mutation."""
        next_generation = []
        
        # Keep elite genomes
        next_generation.extend(elite_genomes)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            # Select two parents
            parent1 = random.choice(elite_genomes)
            parent2 = random.choice(elite_genomes)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_parameters = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child_parameters = parent1.parameters.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_parameters = self._mutate(child_parameters)
            
            # Create child genome
            child_genome = OptimizationGenome(
                agent_id=parent1.agent_id,
                parameters=child_parameters,
                fitness_score=0.0,  # Will be calculated later
                generation=parent1.generation + 1,
                parent_genomes=[parent1.agent_id, parent2.agent_id]
            )
            
            next_generation.append(child_genome)
        
        return next_generation[:self.population_size]
    
    def _crossover(self, parent1_params: Dict[str, float], parent2_params: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parent parameter sets."""
        child_params = {}
        
        for param_name in parent1_params:
            if param_name in parent2_params:
                # Random blend of parent parameters
                alpha = random.random()
                child_params[param_name] = (alpha * parent1_params[param_name] + 
                                          (1 - alpha) * parent2_params[param_name])
            else:
                child_params[param_name] = parent1_params[param_name]
        
        return child_params
    
    def _mutate(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to parameters."""
        mutated_params = parameters.copy()
        
        for param_name, value in mutated_params.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                # Gaussian mutation
                mutation_strength = 0.1
                noise = random.gauss(0, mutation_strength)
                mutated_params[param_name] = max(0.0, min(1.0, value + noise))
        
        return mutated_params
    
    def _calculate_population_diversity(self, population: List[OptimizationGenome]) -> float:
        """Calculate diversity within the population."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_genome_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_genome_distance(self, genome1: OptimizationGenome, genome2: OptimizationGenome) -> float:
        """Calculate distance between two genomes."""
        distance = 0.0
        param_count = 0
        
        for param_name in genome1.parameters:
            if param_name in genome2.parameters:
                diff = abs(genome1.parameters[param_name] - genome2.parameters[param_name])
                distance += diff * diff
                param_count += 1
        
        return math.sqrt(distance / param_count) if param_count > 0 else 0.0
    
    def _calculate_improvement_potential(self, fitness_scores: List[float]) -> float:
        """Calculate potential for improvement based on fitness distribution."""
        if not fitness_scores:
            return 0.5
        
        max_fitness = max(fitness_scores)
        avg_fitness = statistics.mean(fitness_scores)
        
        # Higher potential if there's a big gap between best and average
        potential = (1.0 - max_fitness) + (max_fitness - avg_fitness)
        return min(1.0, potential)
    
    def get_strategy_name(self) -> str:
        return "genetic_optimization"

class ReinforcementLearningOptimizer(AgentOptimizationStrategy):
    """Reinforcement learning-based optimization strategy."""
    
    def __init__(self, learning_rate: float = 0.01, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.action_history = []
        
    async def optimize_agent(self, agent_id: str, metrics: AgentPerformanceMetrics,
                           historical_data: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Optimize agent using reinforcement learning."""
        
        # Define state based on current metrics
        state = self._encode_state(metrics)
        
        # Choose action (parameter adjustment)
        action = await self._choose_action(state)
        
        # Calculate reward based on performance improvement
        reward = self._calculate_reward(metrics, historical_data)
        
        # Update Q-table
        await self._update_q_table(state, action, reward)
        
        # Generate optimization recommendations
        recommendations = await self._generate_recommendations(state, action)
        
        return {
            "strategy": "reinforcement_learning",
            "state": state,
            "action": action,
            "reward": reward,
            "recommendations": recommendations,
            "exploration_rate": self.exploration_rate,
            "q_table_size": len(self.q_table)
        }
    
    def _encode_state(self, metrics: AgentPerformanceMetrics) -> str:
        """Encode agent metrics into a state representation."""
        # Discretize metrics into bins for state representation
        bins = {
            "success": "high" if metrics.task_success_rate > 0.8 else "medium" if metrics.task_success_rate > 0.5 else "low",
            "quality": "high" if metrics.response_quality_score > 0.8 else "medium" if metrics.response_quality_score > 0.5 else "low",
            "speed": "fast" if metrics.processing_speed > 0.8 else "medium" if metrics.processing_speed > 0.5 else "slow",
            "efficiency": "high" if metrics.resource_efficiency > 0.8 else "medium" if metrics.resource_efficiency > 0.5 else "low"
        }
        
        return f"{bins['success']}_{bins['quality']}_{bins['speed']}_{bins['efficiency']}"
    
    async def _choose_action(self, state: str) -> str:
        """Choose optimization action using epsilon-greedy strategy."""
        available_actions = [
            "increase_learning_rate",
            "decrease_learning_rate",
            "increase_creativity",
            "decrease_creativity",
            "increase_exploration",
            "decrease_exploration",
            "increase_specialization",
            "decrease_specialization",
            "increase_collaboration",
            "decrease_collaboration"
        ]
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(available_actions)
        else:
            # Exploit: choose best known action for this state
            if state in self.q_table:
                return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            else:
                return random.choice(available_actions)
    
    def _calculate_reward(self, current_metrics: AgentPerformanceMetrics,
                         historical_data: List[AgentPerformanceMetrics]) -> float:
        """Calculate reward based on performance improvement."""
        if not historical_data:
            return 0.0
        
        # Compare with recent performance
        recent_metrics = historical_data[-5:]  # Last 5 measurements
        avg_recent_performance = statistics.mean([
            (m.task_success_rate + m.response_quality_score + 
             m.processing_speed + m.resource_efficiency) / 4
            for m in recent_metrics
        ])
        
        current_performance = (
            current_metrics.task_success_rate + current_metrics.response_quality_score +
            current_metrics.processing_speed + current_metrics.resource_efficiency
        ) / 4
        
        # Reward is the improvement over recent average
        improvement = current_performance - avg_recent_performance
        
        # Scale reward to [-1, 1] range
        return max(-1.0, min(1.0, improvement * 10))
    
    async def _update_q_table(self, state: str, action: str, reward: float):
        """Update Q-table with new experience."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update rule
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
        
        # Store action history
        self.action_history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "timestamp": datetime.now()
        })
        
        # Limit history size
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-800:]
    
    async def _generate_recommendations(self, state: str, action: str) -> Dict[str, Any]:
        """Generate parameter adjustment recommendations based on action."""
        recommendations = {
            "parameter_adjustments": {},
            "behavioral_changes": [],
            "monitoring_focus": []
        }
        
        action_mappings = {
            "increase_learning_rate": {
                "parameter_adjustments": {"learning_rate": 0.1},
                "behavioral_changes": ["faster_adaptation", "more_experimentation"],
                "monitoring_focus": ["learning_efficiency", "adaptation_speed"]
            },
            "decrease_learning_rate": {
                "parameter_adjustments": {"learning_rate": -0.05},
                "behavioral_changes": ["more_stability", "less_experimentation"],
                "monitoring_focus": ["stability_metrics", "consistency"]
            },
            "increase_creativity": {
                "parameter_adjustments": {"creativity_factor": 0.1, "exploration_rate": 0.05},
                "behavioral_changes": ["more_innovative_responses", "diverse_approaches"],
                "monitoring_focus": ["innovation_score", "response_diversity"]
            },
            "increase_collaboration": {
                "parameter_adjustments": {"collaboration_weight": 0.1},
                "behavioral_changes": ["better_team_coordination", "shared_learning"],
                "monitoring_focus": ["collaboration_score", "team_performance"]
            }
        }
        
        if action in action_mappings:
            recommendations.update(action_mappings[action])
        
        return recommendations
    
    def get_strategy_name(self) -> str:
        return "reinforcement_learning"

class AdvancedAgentOptimizer:
    """
    Advanced Agent Optimization System using multiple strategies.
    
    Features:
    - Multiple optimization strategies (genetic, reinforcement learning, etc.)
    - Continuous performance monitoring and improvement
    - Self-adapting optimization parameters
    - Cross-agent learning and knowledge sharing
    - Performance prediction and proactive optimization
    - Evolutionary agent breeding and selection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AgentOptimizer")
        self.is_initialized = False
        
        # Optimization strategies
        self.strategies = {
            "genetic": GeneticOptimizationStrategy(),
            "reinforcement_learning": ReinforcementLearningOptimizer()
        }
        
        # Performance tracking
        self.agent_metrics_history = {}
        self.optimization_history = {}
        self.learning_patterns = {}
        self.agent_genomes = {}
        
        # Optimization configuration
        self.optimization_interval = 3600  # 1 hour
        self.performance_window = 100  # Track last 100 metrics
        self.improvement_threshold = 0.05  # 5% improvement threshold
        
        # Cross-agent learning
        self.shared_knowledge_base = {}
        self.successful_patterns = {}
        self.agent_collaboration_matrix = {}
        
        # System metrics
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_improvements": 0,
            "average_improvement": 0.0,
            "strategy_success_rates": {},
            "patterns_learned": 0,
            "knowledge_shared": 0
        }
    
    async def initialize(self):
        """Initialize the agent optimization system."""
        self.logger.info("Initializing Advanced Agent Optimization System...")
        
        try:
            # Initialize optimization strategies
            for strategy_name, strategy in self.strategies.items():
                self.optimization_stats["strategy_success_rates"][strategy_name] = {
                    "attempts": 0,
                    "successes": 0,
                    "rate": 0.0
                }
            
            # Load historical optimization data
            await self._load_optimization_history()
            
            self.is_initialized = True
            self.logger.info("Agent Optimization System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Agent Optimization System: {e}")
            raise
    
    async def track_agent_performance(self, agent_id: str, metrics: AgentPerformanceMetrics):
        """Track performance metrics for an agent."""
        if not self.is_initialized:
            await self.initialize()
        
        # Store metrics in history
        if agent_id not in self.agent_metrics_history:
            self.agent_metrics_history[agent_id] = []
        
        self.agent_metrics_history[agent_id].append(metrics)
        
        # Limit history size
        if len(self.agent_metrics_history[agent_id]) > self.performance_window:
            self.agent_metrics_history[agent_id] = self.agent_metrics_history[agent_id][-self.performance_window:]
        
        # Check if optimization is needed
        if await self._should_optimize_agent(agent_id):
            await self.optimize_agent(agent_id)
    
    async def optimize_agent(self, agent_id: str, strategy_name: str = None) -> Dict[str, Any]:
        """Optimize a specific agent using the best available strategy."""
        if not self.is_initialized:
            await self.initialize()
        
        if agent_id not in self.agent_metrics_history:
            raise ValueError(f"No performance history found for agent {agent_id}")
        
        self.logger.info(f"Optimizing agent: {agent_id}")
        
        try:
            # Get current and historical metrics
            current_metrics = self.agent_metrics_history[agent_id][-1]
            historical_metrics = self.agent_metrics_history[agent_id]
            
            # Select optimization strategy
            if strategy_name and strategy_name in self.strategies:
                selected_strategy = self.strategies[strategy_name]
            else:
                selected_strategy = await self._select_best_strategy(agent_id, historical_metrics)
            
            self.logger.info(f"Using optimization strategy: {selected_strategy.get_strategy_name()}")
            
            # Perform optimization
            optimization_result = await selected_strategy.optimize_agent(
                agent_id, current_metrics, historical_metrics
            )
            
            # Process optimization results
            processed_results = await self._process_optimization_results(
                agent_id, optimization_result, selected_strategy.get_strategy_name()
            )
            
            # Extract and store learning patterns
            await self._extract_learning_patterns(agent_id, optimization_result, historical_metrics)
            
            # Update cross-agent knowledge sharing
            await self._update_shared_knowledge(agent_id, optimization_result)
            
            # Track optimization statistics
            await self._update_optimization_stats(selected_strategy.get_strategy_name(), processed_results)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Failed to optimize agent {agent_id}: {e}")
            raise
    
    async def _should_optimize_agent(self, agent_id: str) -> bool:
        """Determine if an agent should be optimized."""
        if agent_id not in self.agent_metrics_history:
            return False
        
        history = self.agent_metrics_history[agent_id]
        
        # Need at least 5 metrics for meaningful optimization
        if len(history) < 5:
            return False
        
        # Check if performance is declining
        recent_performance = self._calculate_overall_performance(history[-3:])
        earlier_performance = self._calculate_overall_performance(history[-6:-3] if len(history) >= 6 else history[:-3])
        
        performance_decline = earlier_performance - recent_performance
        
        # Optimize if performance declined by more than threshold
        if performance_decline > self.improvement_threshold:
            return True
        
        # Check if it's time for regular optimization
        last_optimization = self.optimization_history.get(agent_id, {}).get("last_optimization")
        if last_optimization:
            time_since_optimization = datetime.now() - last_optimization
            if time_since_optimization.total_seconds() > self.optimization_interval:
                return True
        
        return False
    
    def _calculate_overall_performance(self, metrics_list: List[AgentPerformanceMetrics]) -> float:
        """Calculate overall performance score from metrics list."""
        if not metrics_list:
            return 0.0
        
        total_score = 0.0
        for metrics in metrics_list:
            score = (
                metrics.task_success_rate * 0.25 +
                metrics.response_quality_score * 0.20 +
                metrics.processing_speed * 0.15 +
                metrics.resource_efficiency * 0.15 +
                metrics.user_satisfaction * 0.15 +
                metrics.innovation_score * 0.10
            )
            total_score += score
        
        return total_score / len(metrics_list)
    
    async def _select_best_strategy(self, agent_id: str, historical_metrics: List[AgentPerformanceMetrics]) -> AgentOptimizationStrategy:
        """Select the best optimization strategy for this agent."""
        
        # Analyze agent characteristics
        agent_profile = await self._analyze_agent_profile(agent_id, historical_metrics)
        
        # Choose strategy based on agent profile and past success
        strategy_scores = {}
        
        for strategy_name, strategy in self.strategies.items():
            score = 0.0
            
            # Base score from historical success rate
            success_rate = self.optimization_stats["strategy_success_rates"][strategy_name]["rate"]
            score += success_rate * 0.5
            
            # Adjust based on agent characteristics
            if strategy_name == "genetic" and agent_profile["variability"] > 0.7:
                score += 0.3  # Genetic works well with high variability
            elif strategy_name == "reinforcement_learning" and agent_profile["adaptability"] > 0.7:
                score += 0.3  # RL works well with adaptable agents
            
            # Consider recent performance trends
            if agent_profile["trend"] == "declining":
                score += 0.2  # More aggressive optimization needed
            
            strategy_scores[strategy_name] = score
        
        # Select strategy with highest score
        best_strategy_name = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return self.strategies[best_strategy_name]
    
    async def _analyze_agent_profile(self, agent_id: str, historical_metrics: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Analyze agent characteristics to inform optimization strategy selection."""
        if not historical_metrics:
            return {"variability": 0.5, "adaptability": 0.5, "trend": "stable"}
        
        # Calculate performance variability
        performance_scores = [self._calculate_overall_performance([m]) for m in historical_metrics]
        variability = statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0.0
        
        # Calculate adaptability (how quickly agent responds to changes)
        adaptability = statistics.mean([m.adaptability_score for m in historical_metrics])
        
        # Determine trend
        if len(performance_scores) >= 3:
            recent_avg = statistics.mean(performance_scores[-3:])
            earlier_avg = statistics.mean(performance_scores[:-3])
            if recent_avg > earlier_avg + 0.05:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "variability": min(1.0, variability * 5),  # Scale to 0-1
            "adaptability": adaptability,
            "trend": trend,
            "average_performance": statistics.mean(performance_scores),
            "consistency": 1.0 - variability  # Inverse of variability
        }
    
    async def _process_optimization_results(self, agent_id: str, optimization_result: Dict[str, Any], 
                                          strategy_name: str) -> Dict[str, Any]:
        """Process and enhance optimization results."""
        processed_results = optimization_result.copy()
        
        # Add metadata
        processed_results.update({
            "agent_id": agent_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "strategy_used": strategy_name,
            "optimization_id": self._generate_optimization_id(agent_id, strategy_name)
        })
        
        # Calculate expected improvement
        if "best_parameters" in optimization_result:
            expected_improvement = await self._estimate_improvement(agent_id, optimization_result["best_parameters"])
            processed_results["expected_improvement"] = expected_improvement
        
        # Store optimization history
        if agent_id not in self.optimization_history:
            self.optimization_history[agent_id] = []
        
        self.optimization_history[agent_id].append(processed_results)
        
        # Update last optimization time
        if agent_id not in self.optimization_history:
            self.optimization_history[agent_id] = {}
        self.optimization_history[agent_id]["last_optimization"] = datetime.now()
        
        return processed_results
    
    async def _estimate_improvement(self, agent_id: str, optimized_parameters: Dict[str, float]) -> float:
        """Estimate expected performance improvement from optimization."""
        if agent_id not in self.agent_metrics_history:
            return 0.0
        
        current_performance = self._calculate_overall_performance(self.agent_metrics_history[agent_id][-3:])
        
        # Simple heuristic based on parameter changes
        # In production, this would use more sophisticated modeling
        parameter_impact = sum(abs(v - 0.5) for v in optimized_parameters.values()) / len(optimized_parameters)
        estimated_improvement = parameter_impact * 0.1  # Max 10% improvement estimate
        
        return min(0.2, estimated_improvement)  # Cap at 20% improvement
    
    def _generate_optimization_id(self, agent_id: str, strategy_name: str) -> str:
        """Generate unique optimization ID."""
        timestamp = datetime.now().isoformat()
        id_string = f"{agent_id}_{strategy_name}_{timestamp}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    async def _extract_learning_patterns(self, agent_id: str, optimization_result: Dict[str, Any],
                                       historical_metrics: List[AgentPerformanceMetrics]):
        """Extract and store learning patterns from optimization."""
        if agent_id not in self.learning_patterns:
            self.learning_patterns[agent_id] = []
        
        # Create learning pattern from optimization
        pattern = LearningPattern(
            pattern_id=self._generate_pattern_id(),
            agent_id=agent_id,
            context=f"Optimization with {optimization_result.get('strategy', 'unknown')} strategy",
            action=str(optimization_result.get("best_parameters", {})),
            outcome=f"Expected improvement: {optimization_result.get('expected_improvement', 0.0)}",
            success_rate=0.5,  # Will be updated based on actual results
            confidence=optimization_result.get("fitness_score", 0.5),
            usage_count=1,
            last_used=datetime.now(),
            effectiveness_score=optimization_result.get("expected_improvement", 0.0)
        )
        
        self.learning_patterns[agent_id].append(pattern)
        self.optimization_stats["patterns_learned"] += 1
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID."""
        return hashlib.md5(f"pattern_{datetime.now().isoformat()}_{random.random()}".encode()).hexdigest()[:12]
    
    async def _update_shared_knowledge(self, agent_id: str, optimization_result: Dict[str, Any]):
        """Update shared knowledge base with optimization insights."""
        strategy_name = optimization_result.get("strategy", "unknown")
        
        if strategy_name not in self.shared_knowledge_base:
            self.shared_knowledge_base[strategy_name] = {
                "successful_patterns": [],
                "parameter_ranges": {},
                "success_indicators": []
            }
        
        # Store successful optimization patterns
        if optimization_result.get("expected_improvement", 0.0) > 0.05:  # 5% improvement threshold
            pattern = {
                "agent_id": agent_id,
                "parameters": optimization_result.get("best_parameters", {}),
                "improvement": optimization_result.get("expected_improvement", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            self.shared_knowledge_base[strategy_name]["successful_patterns"].append(pattern)
            self.optimization_stats["knowledge_shared"] += 1
        
        # Limit knowledge base size
        for strategy in self.shared_knowledge_base:
            patterns = self.shared_knowledge_base[strategy]["successful_patterns"]
            if len(patterns) > 100:
                # Keep most recent and most successful patterns
                sorted_patterns = sorted(patterns, key=lambda p: p["improvement"], reverse=True)
                self.shared_knowledge_base[strategy]["successful_patterns"] = sorted_patterns[:50]
    
    async def _update_optimization_stats(self, strategy_name: str, optimization_result: Dict[str, Any]):
        """Update optimization statistics."""
        self.optimization_stats["total_optimizations"] += 1
        
        # Update strategy success rates
        strategy_stats = self.optimization_stats["strategy_success_rates"][strategy_name]
        strategy_stats["attempts"] += 1
        
        expected_improvement = optimization_result.get("expected_improvement", 0.0)
        if expected_improvement > 0.02:  # 2% improvement threshold for success
            strategy_stats["successes"] += 1
            self.optimization_stats["successful_improvements"] += 1
        
        # Update success rate
        strategy_stats["rate"] = strategy_stats["successes"] / strategy_stats["attempts"]
        
        # Update average improvement
        current_avg = self.optimization_stats["average_improvement"]
        total_opts = self.optimization_stats["total_optimizations"]
        self.optimization_stats["average_improvement"] = (
            (current_avg * (total_opts - 1) + expected_improvement) / total_opts
        )
    
    async def _load_optimization_history(self):
        """Load historical optimization data."""
        # Placeholder for loading from persistent storage
        # In production, this would load saved optimization history
        pass
    
    async def get_agent_optimization_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current optimization status for an agent."""
        if agent_id not in self.agent_metrics_history:
            return {"error": f"No data found for agent {agent_id}"}
        
        recent_metrics = self.agent_metrics_history[agent_id][-5:]
        current_performance = self._calculate_overall_performance(recent_metrics)
        
        # Get optimization history
        optimization_history = self.optimization_history.get(agent_id, [])
        last_optimization = optimization_history[-1] if optimization_history else None
        
        # Calculate improvement since last optimization
        improvement_since_last = 0.0
        if last_optimization:
            pre_optimization_performance = self._calculate_overall_performance(
                self.agent_metrics_history[agent_id][-10:-5] if len(self.agent_metrics_history[agent_id]) >= 10 else []
            )
            improvement_since_last = current_performance - pre_optimization_performance
        
        return {
            "agent_id": agent_id,
            "current_performance": current_performance,
            "performance_trend": self._get_performance_trend(agent_id),
            "last_optimization": last_optimization,
            "improvement_since_last": improvement_since_last,
            "optimization_due": await self._should_optimize_agent(agent_id),
            "learned_patterns": len(self.learning_patterns.get(agent_id, [])),
            "metrics_history_size": len(self.agent_metrics_history[agent_id])
        }
    
    def _get_performance_trend(self, agent_id: str) -> str:
        """Get performance trend for an agent."""
        if agent_id not in self.agent_metrics_history:
            return "unknown"
        
        history = self.agent_metrics_history[agent_id]
        if len(history) < 5:
            return "insufficient_data"
        
        recent_performance = self._calculate_overall_performance(history[-3:])
        earlier_performance = self._calculate_overall_performance(history[-6:-3] if len(history) >= 6 else history[:-3])
        
        if recent_performance > earlier_performance + 0.05:
            return "improving"
        elif recent_performance < earlier_performance - 0.05:
            return "declining"
        else:
            return "stable"
    
    async def optimize_all_agents(self) -> Dict[str, Any]:
        """Optimize all tracked agents."""
        optimization_results = {}
        
        for agent_id in self.agent_metrics_history:
            if await self._should_optimize_agent(agent_id):
                try:
                    result = await self.optimize_agent(agent_id)
                    optimization_results[agent_id] = result
                except Exception as e:
                    optimization_results[agent_id] = {"error": str(e)}
        
        return {
            "optimized_agents": list(optimization_results.keys()),
            "total_agents_checked": len(self.agent_metrics_history),
            "optimization_results": optimization_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization system metrics."""
        return {
            "system_stats": self.optimization_stats,
            "tracked_agents": len(self.agent_metrics_history),
            "total_patterns_learned": sum(len(patterns) for patterns in self.learning_patterns.values()),
            "shared_knowledge_entries": sum(len(kb["successful_patterns"]) for kb in self.shared_knowledge_base.values()),
            "optimization_strategies": list(self.strategies.keys()),
            "system_initialized": self.is_initialized,
            "performance_window": self.performance_window,
            "optimization_interval": self.optimization_interval
        }
    
    async def deploy(self, target: str):
        """Deploy agent optimization system to target environment."""
        self.logger.info(f"Deploying Agent Optimization System to {target}")
        
        if not self.is_initialized:
            await self.initialize()
        
        # Adjust configuration based on target
        if target == "production":
            self.optimization_interval = 3600  # 1 hour
            self.improvement_threshold = 0.02  # 2% threshold in production
        elif target == "development":
            self.optimization_interval = 300   # 5 minutes
            self.improvement_threshold = 0.01  # 1% threshold in development
        
        self.logger.info(f"Agent Optimization System deployed to {target}")

# Convenience functions
async def quick_optimize_agent(agent_id: str, metrics: AgentPerformanceMetrics) -> Dict[str, Any]:
    """Quick agent optimization for simple use cases."""
    optimizer = AdvancedAgentOptimizer()
    try:
        await optimizer.initialize()
        await optimizer.track_agent_performance(agent_id, metrics)
        result = await optimizer.optimize_agent(agent_id)
        return result
    finally:
        # Cleanup would go here if needed
        pass