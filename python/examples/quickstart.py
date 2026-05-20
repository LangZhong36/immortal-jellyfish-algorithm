"""
IJA Quickstart Example
======================

Demonstrates the most common use-cases of the Immortal Jellyfish Algorithm.
Run from the project root:

    cd immortal-jellyfish-algorithm
    python python/examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ija import IJA


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def example_basic():
    print("=" * 50)
    print("Example 1: Basic usage — Sphere D=30")
    print("=" * 50)
    solver = IJA(n=50, max_iter=500, seed=0)
    result = solver.optimize(sphere, dim=30, lb=-100.0, ub=100.0)
    print(f"  Best fitness : {result.best_fitness:.6e}")
    print(f"  Best pos[:5] : {result.best_position[:5]}")
    print(f"  Function evals: {result.n_function_evals}")
    print()


def example_verbose_callback():
    print("=" * 50)
    print("Example 2: Verbose mode + early-stop callback — Rastrigin D=30")
    print("=" * 50)
    milestones = []

    def callback(iteration, best_fit, best_pos):
        if iteration in (100, 200, 300, 400, 500):
            milestones.append((iteration, best_fit))
        return best_fit < 1e-8

    result = IJA(n=50, max_iter=500, seed=42).optimize(
        rastrigin, dim=30, lb=-5.12, ub=5.12,
        callback=callback, verbose=True,
    )
    print(f"\n  Stopped at iteration {result.n_iterations}")
    print(f"  Milestones: {milestones}")
    print()


def example_custom_bounds():
    print("=" * 50)
    print("Example 3: Per-dimension bounds — Rosenbrock D=10")
    print("=" * 50)
    dim = 10
    lb = np.array([-5.0] * 5 + [-2.0] * 5)
    ub = np.array([ 5.0] * 5 + [ 8.0] * 5)
    result = IJA(n=40, max_iter=300, seed=7).optimize(
        rosenbrock, dim=dim, lb=lb, ub=ub
    )
    print(f"  Best fitness : {result.best_fitness:.6e}")
    print()


def example_reproducibility():
    print("=" * 50)
    print("Example 4: Reproducibility with seed")
    print("=" * 50)
    r1 = IJA(n=30, max_iter=200, seed=123).optimize(sphere, 10, -10, 10)
    r2 = IJA(n=30, max_iter=200, seed=123).optimize(sphere, 10, -10, 10)
    assert r1.best_fitness == r2.best_fitness, "Seed not working!"
    print(f"  Run 1 best: {r1.best_fitness:.6e}")
    print(f"  Run 2 best: {r2.best_fitness:.6e}")
    print(f"  Identical: {r1.best_fitness == r2.best_fitness}")
    print()


if __name__ == "__main__":
    example_basic()
    example_verbose_callback()
    example_custom_bounds()
    example_reproducibility()
    print("All examples completed successfully.")
