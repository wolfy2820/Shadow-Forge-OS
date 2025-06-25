#!/usr/bin/env python3
"""
ShadowForge OS v5.1 - Advanced AI Core
Multi-model AI integration with intelligent routing and optimization
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import os

# Safe imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AIProvider(Enum):
    """Available AI providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai" 
    ANTHROPIC = "anthropic"
    LOCAL = "local"

@dataclass
class ModelConfig:
    """Configuration for AI model."""
    provider: AIProvider
    model_name: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    quality_score: float
    speed_score: float
    context_length: int

@dataclass
class AIRequest:
    """AI request with metadata."""
    prompt: str
    context: str
    model_preference: Optional[str]
    max_tokens: int
    temperature: float
    priority: str
    use_cache: bool
    timestamp: datetime

class AdvancedAICore:
    """
    Advanced AI Core with multi-model integration and intelligent routing.
    
    Features:
    - Multi-provider support (OpenRouter, OpenAI, Anthropic)
    - Intelligent model selection based on task requirements
    - Response caching and optimization
    - Cost tracking and optimization
    - Quality scoring and adaptive learning
    - Failover and redundancy
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AdvancedAICore")
        self.session = None
        self.is_initialized = False
        
        # Model configurations - Enhanced with latest OpenAI models
        self.models = {
            # OpenAI Premium Models (Pro Access)
            "o3": ModelConfig(
                provider=AIProvider.OPENAI,
                model_name="o3",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.060,  # Premium model pricing
                quality_score=0.99,
                speed_score=0.85,
                context_length=200000
            ),
            "gpt-4.5": ModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-4.5-turbo",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.030,
                quality_score=0.97,
                speed_score=0.90,
                context_length=200000
            ),
            "gpt-4-turbo": ModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-4-turbo-preview",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.01,
                quality_score=0.92,
                speed_score=0.8,
                context_length=128000
            ),
            "gpt-4": ModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-4",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.03,
                quality_score=0.94,
                speed_score=0.75,
                context_length=8192
            ),
            # Anthropic Models
            "claude-3-opus": ModelConfig(
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.015,
                quality_score=0.95,
                speed_score=0.7,
                context_length=200000
            ),
            "claude-3-sonnet": ModelConfig(
                provider=AIProvider.OPENROUTER,
                model_name="anthropic/claude-3-sonnet",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.003,
                quality_score=0.88,
                speed_score=0.85,
                context_length=200000
            ),
            # Alternative Models
            "mixtral-8x7b": ModelConfig(
                provider=AIProvider.OPENROUTER,
                model_name="mistralai/mixtral-8x7b-instruct",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.0005,
                quality_score=0.82,
                speed_score=0.9,
                context_length=32000
            )
        }
        
        # API configurations
        self.api_keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")
        }
        
        # Performance tracking
        self.usage_stats = {}
        self.quality_scores = {}
        self.response_cache = {}
        self.cost_tracking = {"total_cost": 0.0, "requests": 0}
        
        # Advanced features
        self.model_performance_history = {}
        self.adaptive_routing_enabled = True
        self.cache_ttl = 3600  # 1 hour cache
        
    async def initialize(self):
        """Initialize the advanced AI core."""
        self.logger.info("Initializing Advanced AI Core...")
        
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
            
            # Initialize client connections
            await self._initialize_clients()
            
            # Load performance history
            await self._load_performance_history()
            
            self.is_initialized = True
            self.logger.info("Advanced AI Core initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Advanced AI Core: {e}")
            raise
    
    async def _initialize_clients(self):
        """Initialize AI provider clients."""
        if OPENAI_AVAILABLE and self.api_keys["openai"]:
            self.openai_client = openai.AsyncOpenAI(api_key=self.api_keys["openai"])
            self.logger.info("OpenAI client initialized")
        
        if ANTHROPIC_AVAILABLE and self.api_keys["anthropic"]:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_keys["anthropic"])
            self.logger.info("Anthropic client initialized")
    
    async def generate_response(self, request: AIRequest) -> Dict[str, Any]:
        """
        Generate AI response with intelligent model selection and optimization.
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            if request.use_cache:
                cached_response = await self._get_cached_response(request)
                if cached_response:
                    self.logger.info("Returning cached response")
                    return cached_response
            
            # Select optimal model
            selected_model = await self._select_optimal_model(request)
            self.logger.info(f"Selected model: {selected_model}")
            
            # Generate response
            response = await self._generate_with_model(request, selected_model)
            
            # Process and optimize response
            processed_response = await self._process_response(response, request)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            await self._update_performance_metrics(selected_model, processing_time, processed_response)
            
            # Cache response if appropriate
            if request.use_cache:
                await self._cache_response(request, processed_response)
            
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return await self._generate_fallback_response(request, str(e))
    
    async def _select_optimal_model(self, request: AIRequest) -> str:
        """
        Select optimal model based on request requirements and performance history.
        """
        if request.model_preference and request.model_preference in self.models:
            return request.model_preference
        
        if not self.adaptive_routing_enabled:
            return "claude-3-sonnet"  # Default model
        
        # Score models based on requirements
        model_scores = {}
        
        for model_name, config in self.models.items():
            score = 0.0
            
            # Quality requirement
            if request.priority == "high":
                score += config.quality_score * 0.4
            else:
                score += config.quality_score * 0.2
            
            # Speed requirement
            if request.priority == "urgent":
                score += config.speed_score * 0.4
            else:
                score += config.speed_score * 0.2
            
            # Cost efficiency
            cost_efficiency = 1.0 / (config.cost_per_1k_tokens + 0.001)
            score += cost_efficiency * 0.2
            
            # Context length requirement
            prompt_length = len(request.prompt) + len(request.context)
            if prompt_length < config.context_length:
                score += 0.2
            
            # Historical performance
            if model_name in self.model_performance_history:
                performance = self.model_performance_history[model_name]
                score += performance.get("success_rate", 0.8) * 0.1
            
            model_scores[model_name] = score
        
        # Select best scoring model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        return best_model[0]
    
    async def _generate_with_model(self, request: AIRequest, model_name: str) -> Dict[str, Any]:
        """Generate response with specific model."""
        config = self.models[model_name]
        
        if config.provider == AIProvider.OPENROUTER:
            return await self._generate_openrouter(request, config)
        elif config.provider == AIProvider.OPENAI:
            return await self._generate_openai(request, config)
        elif config.provider == AIProvider.ANTHROPIC:
            return await self._generate_anthropic(request, config)
        else:
            return await self._generate_fallback_response(request, "Unsupported provider")
    
    async def _generate_openrouter(self, request: AIRequest, config: ModelConfig) -> Dict[str, Any]:
        """Generate response using OpenRouter."""
        if not self.api_keys["openrouter"]:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openrouter']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://shadowforge-os.ai",
            "X-Title": "ShadowForge OS"
        }
        
        messages = []
        if request.context:
            messages.append({"role": "system", "content": request.context})
        messages.append({"role": "user", "content": request.prompt})
        
        payload = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": min(request.max_tokens, config.max_tokens),
            "temperature": request.temperature,
            "stream": False
        }
        
        async with self.session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": config.model_name,
                    "provider": "openrouter",
                    "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                    "cost": self._calculate_cost(data.get("usage", {}).get("total_tokens", 0), config)
                }
            else:
                error_data = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error_data}")
    
    async def _generate_openai(self, request: AIRequest, config: ModelConfig) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        if not OPENAI_AVAILABLE or not hasattr(self, 'openai_client'):
            raise ValueError("OpenAI client not available")
        
        messages = []
        if request.context:
            messages.append({"role": "system", "content": request.context})
        messages.append({"role": "user", "content": request.prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            max_tokens=min(request.max_tokens, config.max_tokens),
            temperature=request.temperature
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": config.model_name,
            "provider": "openai",
            "tokens_used": response.usage.total_tokens,
            "cost": self._calculate_cost(response.usage.total_tokens, config)
        }
    
    async def _generate_anthropic(self, request: AIRequest, config: ModelConfig) -> Dict[str, Any]:
        """Generate response using Anthropic."""
        if not ANTHROPIC_AVAILABLE or not hasattr(self, 'anthropic_client'):
            raise ValueError("Anthropic client not available")
        
        system_message = request.context if request.context else None
        
        response = await self.anthropic_client.messages.create(
            model=config.model_name,
            max_tokens=min(request.max_tokens, config.max_tokens),
            temperature=request.temperature,
            system=system_message,
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        return {
            "content": response.content[0].text,
            "model": config.model_name,
            "provider": "anthropic",
            "tokens_used": tokens_used,
            "cost": self._calculate_cost(tokens_used, config)
        }
    
    async def _process_response(self, response: Dict[str, Any], request: AIRequest) -> Dict[str, Any]:
        """Process and enhance AI response."""
        processed = response.copy()
        
        # Add metadata
        processed.update({
            "timestamp": datetime.now().isoformat(),
            "request_id": self._generate_request_id(request),
            "processing_time": time.time(),
            "quality_score": await self._assess_response_quality(response["content"], request)
        })
        
        # Enhance content if needed
        if request.priority == "high":
            processed["content"] = await self._enhance_response_quality(processed["content"])
        
        return processed
    
    async def _enhance_response_quality(self, content: str) -> str:
        """Enhance response quality using advanced techniques."""
        # This could include grammar checking, fact verification, etc.
        # For now, return as-is but this is where we'd add enhancements
        return content
    
    async def _assess_response_quality(self, content: str, request: AIRequest) -> float:
        """Assess the quality of the response."""
        # Simple quality assessment based on length and coherence
        score = 0.5  # Base score
        
        # Length check
        if len(content) > 100:
            score += 0.2
        
        # Coherence check (basic)
        if content.count('.') > 0:  # Has sentences
            score += 0.1
        
        # Relevance check (basic keyword matching)
        request_words = set(request.prompt.lower().split())
        response_words = set(content.lower().split())
        overlap = len(request_words.intersection(response_words))
        if overlap > 0:
            score += min(0.2, overlap * 0.05)
        
        return min(1.0, score)
    
    async def _get_cached_response(self, request: AIRequest) -> Optional[Dict[str, Any]]:
        """Get cached response if available and valid."""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached["cached_at"])
            if datetime.now() - cache_time < timedelta(seconds=self.cache_ttl):
                return cached["response"]
        
        return None
    
    async def _cache_response(self, request: AIRequest, response: Dict[str, Any]):
        """Cache response for future use."""
        cache_key = self._generate_cache_key(request)
        
        self.response_cache[cache_key] = {
            "response": response,
            "cached_at": datetime.now().isoformat()
        }
        
        # Limit cache size
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(
                self.response_cache.items(),
                key=lambda x: x[1]["cached_at"]
            )
            for key, _ in sorted_items[:100]:  # Remove oldest 100
                del self.response_cache[key]
    
    def _generate_cache_key(self, request: AIRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.prompt}|{request.context}|{request.temperature}|{request.max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_request_id(self, request: AIRequest) -> str:
        """Generate unique request ID."""
        return hashlib.md5(f"{request.timestamp}{request.prompt}".encode()).hexdigest()[:16]
    
    def _calculate_cost(self, tokens: int, config: ModelConfig) -> float:
        """Calculate cost for token usage."""
        return (tokens / 1000) * config.cost_per_1k_tokens
    
    async def _update_performance_metrics(self, model: str, processing_time: float, response: Dict[str, Any]):
        """Update performance metrics for model."""
        if model not in self.model_performance_history:
            self.model_performance_history[model] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_processing_time": 0.0,
                "total_cost": 0.0,
                "average_quality": 0.0
            }
        
        metrics = self.model_performance_history[model]
        metrics["total_requests"] += 1
        metrics["successful_requests"] += 1
        metrics["total_processing_time"] += processing_time
        metrics["total_cost"] += response.get("cost", 0.0)
        
        # Update average quality
        current_quality = response.get("quality_score", 0.5)
        metrics["average_quality"] = (
            (metrics["average_quality"] * (metrics["successful_requests"] - 1) + current_quality)
            / metrics["successful_requests"]
        )
        
        # Update success rate
        metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        
        # Update global cost tracking
        self.cost_tracking["total_cost"] += response.get("cost", 0.0)
        self.cost_tracking["requests"] += 1
    
    async def _load_performance_history(self):
        """Load performance history from storage."""
        # This would load from persistent storage in production
        pass
    
    async def _generate_fallback_response(self, request: AIRequest, error: str) -> Dict[str, Any]:
        """Generate fallback response when primary methods fail."""
        fallback_content = f"I apologize, but I'm currently experiencing technical difficulties. Error: {error[:100]}..."
        
        return {
            "content": fallback_content,
            "model": "fallback",
            "provider": "local",
            "tokens_used": len(fallback_content.split()),
            "cost": 0.0,
            "timestamp": datetime.now().isoformat(),
            "is_fallback": True,
            "quality_score": 0.1
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "total_requests": self.cost_tracking["requests"],
            "total_cost": self.cost_tracking["total_cost"],
            "average_cost_per_request": (
                self.cost_tracking["total_cost"] / max(1, self.cost_tracking["requests"])
            ),
            "model_performance": self.model_performance_history,
            "cache_size": len(self.response_cache),
            "available_models": list(self.models.keys()),
            "initialized": self.is_initialized
        }
    
    async def optimize_performance(self):
        """Optimize performance based on usage patterns."""
        self.logger.info("Optimizing AI Core performance...")
        
        # Analyze model performance and adjust routing weights
        for model_name, performance in self.model_performance_history.items():
            if model_name in self.models:
                config = self.models[model_name]
                
                # Update quality score based on actual performance
                actual_quality = performance.get("average_quality", config.quality_score)
                config.quality_score = (config.quality_score * 0.7) + (actual_quality * 0.3)
                
                # Update speed score based on processing time
                avg_time = performance.get("total_processing_time", 1.0) / max(1, performance.get("total_requests", 1))
                speed_score = max(0.1, min(1.0, 10.0 / max(1.0, avg_time)))
                config.speed_score = (config.speed_score * 0.7) + (speed_score * 0.3)
        
        self.logger.info("AI Core performance optimization complete")
    
    async def deploy(self, target: str):
        """Deploy AI core to target environment."""
        self.logger.info(f"Deploying Advanced AI Core to {target}")
        
        if not self.is_initialized:
            await self.initialize()
        
        # Perform deployment-specific configurations
        if target == "production":
            self.cache_ttl = 7200  # 2 hours in production
            self.adaptive_routing_enabled = True
        elif target == "development":
            self.cache_ttl = 300   # 5 minutes in development
            self.adaptive_routing_enabled = False
        
        self.logger.info(f"Advanced AI Core deployed to {target}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        
        self.logger.info("Advanced AI Core cleanup complete")

# Convenience functions for easy usage
async def create_ai_request(
    prompt: str,
    context: str = "",
    model: str = None,
    priority: str = "normal",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    use_cache: bool = True
) -> AIRequest:
    """Create an AI request with sensible defaults."""
    return AIRequest(
        prompt=prompt,
        context=context,
        model_preference=model,
        max_tokens=max_tokens,
        temperature=temperature,
        priority=priority,
        use_cache=use_cache,
        timestamp=datetime.now()
    )

async def quick_generate(prompt: str, context: str = "", model: str = None) -> str:
    """Quick generation for simple use cases."""
    ai_core = AdvancedAICore()
    request = await create_ai_request(prompt, context, model)
    response = await ai_core.generate_response(request)
    await ai_core.cleanup()
    return response["content"]