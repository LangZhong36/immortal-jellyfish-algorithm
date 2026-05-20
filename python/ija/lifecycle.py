"""
Lifecycle phase scheduler for the Immortal Jellyfish Algorithm.

The normalised progress ratio tau = t / T_max partitions [0,1] into four
biologically-motivated phases that control which operators are active.
"""

import numpy as np

PHASE_POLYP        = 0
PHASE_STROBILATION = 1
PHASE_MEDUSA       = 2
PHASE_SENESCENCE   = 3

_BOUNDARIES = (0.15, 0.50, 0.85)


def get_phase(tau: float) -> int:
    """Return the lifecycle phase index (0–3) for a given progress ratio tau."""
    if tau < _BOUNDARIES[0]:
        return PHASE_POLYP
    if tau < _BOUNDARIES[1]:
        return PHASE_STROBILATION
    if tau < _BOUNDARIES[2]:
        return PHASE_MEDUSA
    return PHASE_SENESCENCE


def sigmoid_transition(tau: float) -> float:
    """Smooth sigmoidal blend weight: 0 near start, 1 near end."""
    return 1.0 / (1.0 + np.exp(-12.0 * (tau - 0.5)))


def adaptive_alpha(tau: float,
                   alpha_max: float = 2.0,
                   alpha_min: float = 0.05) -> float:
    """Step-size that decays from alpha_max to alpha_min as tau → 1."""
    w = sigmoid_transition(tau)
    return alpha_max * (1.0 - w) + alpha_min * w


def lighthouse_intensity(distance_sq: float,
                         tau: float,
                         I0: float = 1.0,
                         kappa: float = 0.01,
                         decay_lambda: float = 0.5) -> float:
    """
    Inverse-square light intensity with exponential temporal decay.

    I = I0 / (1 + kappa * d^2) * exp(-lambda * tau)
    """
    return (I0 / (1.0 + kappa * distance_sq)) * np.exp(-decay_lambda * tau)


def adaptive_n_ephyra(tau: float,
                      n_max: int = 5,
                      n_min: int = 2) -> int:
    """
    Number of ephyra offspring produced during strobilation.

    Starts large for diversity, shrinks as tau increases.
    """
    frac = max(0.0, min(1.0, (tau - _BOUNDARIES[0]) / (_BOUNDARIES[1] - _BOUNDARIES[0])))
    return max(n_min, round(n_max - frac * (n_max - n_min)))
