"""
ShadowForge OS v5.1 - Quantum Core Module

The quantum core provides the foundational quantum computing capabilities
that enable the system's self-evolution and parallel processing abilities.
"""

from .entanglement_engine import EntanglementEngine
from .superposition_router import SuperpositionRouter
from .decoherence_shield import DecoherenceShield

__all__ = [
    "EntanglementEngine",
    "SuperpositionRouter", 
    "DecoherenceShield"
]