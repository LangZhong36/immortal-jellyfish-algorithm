"""
Standard benchmark functions for continuous optimization.

All functions accept a 1-D numpy array x and return a scalar float.
Global minimum is 0 (or near 0) for all functions unless noted.
"""

from typing import Callable, Dict, NamedTuple

import numpy as np


class BenchmarkSpec(NamedTuple):
    name: str
    func: Callable[[np.ndarray], float]
    lb: float
    ub: float
    dim: int
    global_min: float


def sphere(x: np.ndarray) -> float:
    """F1 — Sphere: sum(x_i^2). Unimodal, separable."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """F2 — Rosenbrock (banana): sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2). Unimodal, non-separable."""
    xi = x[:-1]
    xi1 = x[1:]
    return float(np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1.0 - xi) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """F3 — Rastrigin: 10*d + sum(x_i^2 - 10*cos(2*pi*x_i)). Highly multimodal."""
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """F4 — Ackley: exponential-cosine surface with many local optima."""
    d = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    return float(
        -a * np.exp(-b * np.sqrt(sum_sq / d))
        - np.exp(sum_cos / d)
        + a
        + np.e
    )


def griewank(x: np.ndarray) -> float:
    """F5 — Griewank: product-cosine term creates regular local minima."""
    d = len(x)
    indices = np.arange(1, d + 1, dtype=float)
    sum_sq = np.sum(x ** 2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(indices)))
    return float(sum_sq - prod_cos + 1.0)


def levy(x: np.ndarray) -> float:
    """F6 — Levy: sinusoidal function with global minimum at x=(1,...,1)."""
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term_mid = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    term_end = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(term1 + term_mid + term_end)


def schwefel(x: np.ndarray) -> float:
    """F7 — Schwefel: deceptive function with global minimum far from local minima."""
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def zakharov(x: np.ndarray) -> float:
    """F8 — Zakharov: plate-shaped with polynomial terms."""
    d = len(x)
    i = np.arange(1, d + 1, dtype=float)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * i * x)
    return float(sum1 + sum2 ** 2 + sum2 ** 4)


BENCHMARK_SUITE: Dict[str, BenchmarkSpec] = {
    "sphere":     BenchmarkSpec("Sphere",     sphere,     -100.0,  100.0, 30, 0.0),
    "rosenbrock": BenchmarkSpec("Rosenbrock", rosenbrock, -30.0,    30.0, 30, 0.0),
    "rastrigin":  BenchmarkSpec("Rastrigin",  rastrigin,  -5.12,    5.12, 30, 0.0),
    "ackley":     BenchmarkSpec("Ackley",     ackley,     -32.768, 32.768, 30, 0.0),
    "griewank":   BenchmarkSpec("Griewank",   griewank,   -600.0,  600.0, 30, 0.0),
    "levy":       BenchmarkSpec("Levy",       levy,       -10.0,   10.0,  30, 0.0),
    "schwefel":   BenchmarkSpec("Schwefel",   schwefel,   -500.0,  500.0, 30, 0.0),
    "zakharov":   BenchmarkSpec("Zakharov",   zakharov,   -5.0,     10.0, 30, 0.0),
}
