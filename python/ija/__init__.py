"""
IJA — Immortal Jellyfish Algorithm
===================================

A nature-inspired metaheuristic optimizer modelling the lifecycle of
*Turritopsis dohrnii* and lighthouse light propagation.

Quick start
-----------
>>> from ija import IJA
>>> import numpy as np
>>> result = IJA(n=50, max_iter=500).optimize(lambda x: np.sum(x**2), dim=30, lb=-100, ub=100)
>>> print(result.best_fitness)
"""

from .algorithm import IJA, OptimizeResult
from .lifecycle import (
    PHASE_MEDUSA,
    PHASE_POLYP,
    PHASE_SENESCENCE,
    PHASE_STROBILATION,
    get_phase,
)

__version__ = "1.0.0"
__author__  = "IJA Contributors"
__license__ = "MIT"

__all__ = [
    "IJA",
    "OptimizeResult",
    "get_phase",
    "PHASE_POLYP",
    "PHASE_STROBILATION",
    "PHASE_MEDUSA",
    "PHASE_SENESCENCE",
]
