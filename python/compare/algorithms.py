"""
Baseline algorithms for comparison with IJA.

All implement the same interface as IJA.optimize():
    result = AlgorithmClass(n, max_iter).optimize(objective, dim, lb, ub)

Each returns a simple namespace with .best_fitness, .best_position, .convergence_curve.
"""

from __future__ import annotations

from typing import Callable, List, Optional
from types import SimpleNamespace

import numpy as np


def _make_result(best_fit, best_pos, curve):
    return SimpleNamespace(
        best_fitness=best_fit,
        best_position=best_pos,
        convergence_curve=curve,
    )


class PSO:
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995)."""

    def __init__(self, n: int = 50, max_iter: int = 500,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 seed: Optional[int] = None) -> None:
        self.n, self.max_iter = n, max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.seed = seed

    def optimize(self, objective, dim, lb, ub):
        if self.seed is not None:
            np.random.seed(self.seed)
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)

        pos = lb_arr + np.random.rand(self.n, dim) * (ub_arr - lb_arr)
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pbest_fit = np.array([objective(pos[i]) for i in range(self.n)])
        gbest_idx = int(np.argmin(pbest_fit))
        gbest = pbest[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        curve: List[float] = [gbest_fit]
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.n, dim)
            r2 = np.random.rand(self.n, dim)
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, lb_arr, ub_arr)
            fits = np.array([objective(pos[i]) for i in range(self.n)])
            improved = fits < pbest_fit
            pbest[improved] = pos[improved].copy()
            pbest_fit[improved] = fits[improved]
            cur_best_idx = int(np.argmin(pbest_fit))
            if pbest_fit[cur_best_idx] < gbest_fit:
                gbest = pbest[cur_best_idx].copy()
                gbest_fit = float(pbest_fit[cur_best_idx])
            curve.append(gbest_fit)
        return _make_result(gbest_fit, gbest, curve)


class DE:
    """Differential Evolution — DE/rand/1/bin (Storn & Price, 1997)."""

    def __init__(self, n: int = 50, max_iter: int = 500,
                 F: float = 0.8, CR: float = 0.9,
                 seed: Optional[int] = None) -> None:
        self.n, self.max_iter = n, max_iter
        self.F, self.CR = F, CR
        self.seed = seed

    def optimize(self, objective, dim, lb, ub):
        if self.seed is not None:
            np.random.seed(self.seed)
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)

        pop = lb_arr + np.random.rand(self.n, dim) * (ub_arr - lb_arr)
        fits = np.array([objective(pop[i]) for i in range(self.n)])
        best_idx = int(np.argmin(fits))
        best_fit = float(fits[best_idx])
        best_pos = pop[best_idx].copy()
        curve: List[float] = [best_fit]

        for _ in range(self.max_iter):
            for i in range(self.n):
                candidates = [j for j in range(self.n) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb_arr, ub_arr)
                cross_pts = np.random.rand(dim) < self.CR
                if not cross_pts.any():
                    cross_pts[np.random.randint(dim)] = True
                trial = np.where(cross_pts, mutant, pop[i])
                trial_fit = objective(trial)
                if trial_fit <= fits[i]:
                    pop[i] = trial
                    fits[i] = trial_fit
                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_pos = trial.copy()
            curve.append(best_fit)
        return _make_result(best_fit, best_pos, curve)


class GWO:
    """Grey Wolf Optimizer (Mirjalili et al., 2014)."""

    def __init__(self, n: int = 50, max_iter: int = 500,
                 seed: Optional[int] = None) -> None:
        self.n, self.max_iter = n, max_iter
        self.seed = seed

    def optimize(self, objective, dim, lb, ub):
        if self.seed is not None:
            np.random.seed(self.seed)
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)

        pop = lb_arr + np.random.rand(self.n, dim) * (ub_arr - lb_arr)
        fits = np.array([objective(pop[i]) for i in range(self.n)])
        sorted_idx = np.argsort(fits)
        alpha_pos = pop[sorted_idx[0]].copy()
        alpha_fit = float(fits[sorted_idx[0]])
        beta_pos  = pop[sorted_idx[1]].copy()
        delta_pos = pop[sorted_idx[2]].copy()
        curve: List[float] = [alpha_fit]

        for t in range(1, self.max_iter + 1):
            a = 2.0 - 2.0 * t / self.max_iter
            for i in range(self.n):
                A1 = 2 * a * np.random.rand(dim) - a
                C1 = 2 * np.random.rand(dim)
                D_alpha = np.abs(C1 * alpha_pos - pop[i])
                X1 = alpha_pos - A1 * D_alpha

                A2 = 2 * a * np.random.rand(dim) - a
                C2 = 2 * np.random.rand(dim)
                D_beta = np.abs(C2 * beta_pos - pop[i])
                X2 = beta_pos - A2 * D_beta

                A3 = 2 * a * np.random.rand(dim) - a
                C3 = 2 * np.random.rand(dim)
                D_delta = np.abs(C3 * delta_pos - pop[i])
                X3 = delta_pos - A3 * D_delta

                pop[i] = np.clip((X1 + X2 + X3) / 3.0, lb_arr, ub_arr)
                fits[i] = objective(pop[i])

            sorted_idx = np.argsort(fits)
            if fits[sorted_idx[0]] < alpha_fit:
                alpha_fit = float(fits[sorted_idx[0]])
                alpha_pos = pop[sorted_idx[0]].copy()
            beta_pos  = pop[sorted_idx[1]].copy()
            delta_pos = pop[sorted_idx[2]].copy()
            curve.append(alpha_fit)

        return _make_result(alpha_fit, alpha_pos, curve)


class WOA:
    """Whale Optimization Algorithm (Mirjalili & Lewis, 2016)."""

    def __init__(self, n: int = 50, max_iter: int = 500,
                 seed: Optional[int] = None) -> None:
        self.n, self.max_iter = n, max_iter
        self.seed = seed

    def optimize(self, objective, dim, lb, ub):
        if self.seed is not None:
            np.random.seed(self.seed)
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)

        pop = lb_arr + np.random.rand(self.n, dim) * (ub_arr - lb_arr)
        fits = np.array([objective(pop[i]) for i in range(self.n)])
        best_idx = int(np.argmin(fits))
        best_pos = pop[best_idx].copy()
        best_fit = float(fits[best_idx])
        curve: List[float] = [best_fit]

        for t in range(1, self.max_iter + 1):
            a = 2.0 - 2.0 * t / self.max_iter
            a2 = -1.0 - t / self.max_iter
            for i in range(self.n):
                r = np.random.rand()
                A = 2 * a * np.random.rand(dim) - a
                C = 2 * np.random.rand(dim)
                p = np.random.rand()
                b = 1.0
                l = (a2 - 1.0) * np.random.rand() + 1.0

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        D = np.abs(C * best_pos - pop[i])
                        pop[i] = np.clip(best_pos - A * D, lb_arr, ub_arr)
                    else:
                        rand_idx = np.random.randint(0, self.n)
                        D = np.abs(C * pop[rand_idx] - pop[i])
                        pop[i] = np.clip(pop[rand_idx] - A * D, lb_arr, ub_arr)
                else:
                    D = np.abs(best_pos - pop[i])
                    pop[i] = np.clip(
                        D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos,
                        lb_arr, ub_arr,
                    )
                fits[i] = objective(pop[i])
                if fits[i] < best_fit:
                    best_fit = float(fits[i])
                    best_pos = pop[i].copy()
            curve.append(best_fit)

        return _make_result(best_fit, best_pos, curve)


class SCA:
    """Sine Cosine Algorithm (Mirjalili, 2016)."""

    def __init__(self, n: int = 50, max_iter: int = 500,
                 seed: Optional[int] = None) -> None:
        self.n, self.max_iter = n, max_iter
        self.seed = seed

    def optimize(self, objective, dim, lb, ub):
        if self.seed is not None:
            np.random.seed(self.seed)
        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)

        pop = lb_arr + np.random.rand(self.n, dim) * (ub_arr - lb_arr)
        fits = np.array([objective(pop[i]) for i in range(self.n)])
        best_idx = int(np.argmin(fits))
        best_pos = pop[best_idx].copy()
        best_fit = float(fits[best_idx])
        curve: List[float] = [best_fit]

        for t in range(1, self.max_iter + 1):
            r1 = 2.0 - 2.0 * t / self.max_iter
            for i in range(self.n):
                r2 = 2 * np.pi * np.random.rand(dim)
                r3 = 2 * np.random.rand(dim)
                r4 = np.random.rand(dim)
                mask_sin = r4 < 0.5
                mask_cos = ~mask_sin
                pop[i] = np.where(
                    mask_sin,
                    pop[i] + r1 * np.sin(r2) * np.abs(r3 * best_pos - pop[i]),
                    pop[i] + r1 * np.cos(r2) * np.abs(r3 * best_pos - pop[i]),
                )
                pop[i] = np.clip(pop[i], lb_arr, ub_arr)
                fits[i] = objective(pop[i])
                if fits[i] < best_fit:
                    best_fit = float(fits[i])
                    best_pos = pop[i].copy()
            curve.append(best_fit)

        return _make_result(best_fit, best_pos, curve)


ALGORITHMS = {
    "IJA": None,
    "PSO": PSO,
    "DE":  DE,
    "GWO": GWO,
    "WOA": WOA,
    "SCA": SCA,
}
