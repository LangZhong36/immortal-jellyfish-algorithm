"""
Immortal Jellyfish Algorithm (IJA) — main optimizer.

Inspired by:
  1. The lifecycle of Turritopsis dohrnii (the immortal jellyfish) — the only
     known animal that can revert from adult medusa to juvenile polyp form,
     cycling indefinitely through four distinct life stages.
  2. The inverse-square light-intensity model of a physical lighthouse.

Reference lifecycle phases
--------------------------
  Phase 0  Polyp         [0.00, 0.15) T   Lévy-flight global exploration
  Phase 1  Strobilation  [0.15, 0.50) T   Ephyra budding & diversification
  Phase 2  Medusa        [0.50, 0.85) T   Lighthouse-guided exploitation
  Phase 3  Senescence    [0.85, 1.00] T   Fine-grained local refinement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .archive import EliteArchive
from .lifecycle import (
    PHASE_MEDUSA,
    PHASE_POLYP,
    PHASE_SENESCENCE,
    PHASE_STROBILATION,
    adaptive_alpha,
    adaptive_n_ephyra,
    get_phase,
    lighthouse_intensity,
    sigmoid_transition,
)
from .operators import (
    apply_levy_exploration,
    apply_lighthouse_attraction,
    apply_pulse_swimming,
    archive_bud,
    compute_diversity,
    senescence_refine,
    strobilate,
    transdifferentiate,
)


@dataclass
class OptimizeResult:
    """Container for IJA optimization results."""

    best_fitness: float
    best_position: np.ndarray
    convergence_curve: List[float]
    n_function_evals: int
    n_iterations: int
    phases_completed: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(best_fitness={self.best_fitness:.6e}, "
            f"n_iters={self.n_iterations}, nfev={self.n_function_evals})"
        )


class IJA:
    """
    Immortal Jellyfish Algorithm optimizer.

    Parameters
    ----------
    n : int
        Population size.
    max_iter : int
        Maximum number of iterations.
    top_k : int
        Number of elite lighthouses in the Medusa phase.
    levy_beta : float
        Lévy exponent (must be in (1, 2)).
    archive_size : int
        Capacity of the elite archive.
    age_max : int
        Stagnation threshold; triggers transdifferentiation.
    decay_lambda : float
        Exponential decay rate of lighthouse intensity.
    alpha_max : float
        Maximum adaptive step-size.
    alpha_min : float
        Minimum adaptive step-size.
    diversity_threshold : float
        Normalised diversity below which the diversity guard fires.
    diversity_reset_frac : float
        Fraction of worst agents replaced when diversity guard fires.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n: int = 50,
        max_iter: int = 500,
        top_k: int = 3,
        levy_beta: float = 1.5,
        archive_size: int = 15,
        age_max: int = 25,
        decay_lambda: float = 0.5,
        alpha_max: float = 2.0,
        alpha_min: float = 0.05,
        diversity_threshold: float = 0.005,
        diversity_reset_frac: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.n = n
        self.max_iter = max_iter
        self.top_k = top_k
        self.levy_beta = levy_beta
        self.archive_size = archive_size
        self.age_max = age_max
        self.decay_lambda = decay_lambda
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.diversity_threshold = diversity_threshold
        self.diversity_reset_frac = diversity_reset_frac
        self.seed = seed

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        verbose: bool = False,
    ) -> OptimizeResult:
        """
        Run IJA on the given objective function.

        Parameters
        ----------
        objective : callable
            Scalar-valued function f(x) where x is a 1-D numpy array.
        dim : int
            Problem dimensionality.
        lb : float or array-like
            Lower bounds (scalar broadcast or per-dimension array).
        ub : float or array-like
            Upper bounds.
        callback : callable, optional
            Called each iteration as callback(iteration, best_fitness, best_position).
            Returning True from the callback stops the search early.
        verbose : bool
            Print progress every 50 iterations.

        Returns
        -------
        OptimizeResult
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)

        population, fitnesses, ages, phases = self._initialise(
            objective, dim, lb_arr, ub_arr
        )
        personal_phases = np.random.uniform(0.0, 2.0 * np.pi, self.n)

        archive = EliteArchive(self.archive_size)
        archive.update(population, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        best_pos = population[best_idx].copy()
        best_fit = float(fitnesses[best_idx])

        convergence: List[float] = [best_fit]
        nfev = self.n
        completed_phases: List[int] = []
        prev_phase = get_phase(0.0)

        for iteration in range(1, self.max_iter + 1):
            tau = iteration / self.max_iter
            phase = get_phase(tau)

            if phase != prev_phase:
                completed_phases.append(prev_phase)
                prev_phase = phase

            alpha = adaptive_alpha(tau, self.alpha_max, self.alpha_min)
            n_ephyra = adaptive_n_ephyra(tau)

            lighthouses = self._select_lighthouses(population, fitnesses)

            for i in range(self.n):
                xi = population[i]
                fi = fitnesses[i]

                if phase == PHASE_POLYP:
                    xi_new = apply_levy_exploration(xi, best_pos, lb_arr, ub_arr, self.levy_beta)
                    fi_new = objective(xi_new)
                    nfev += 1

                elif phase == PHASE_STROBILATION:
                    xi_new, fi_new = strobilate(
                        xi, fi, objective, n_ephyra, lb_arr, ub_arr, self.levy_beta
                    )
                    nfev += n_ephyra

                elif phase == PHASE_MEDUSA:
                    gs = self._pick_lighthouse(lighthouses, fitnesses)
                    dist_sq = float(np.sum((xi - gs) ** 2))
                    intensity = lighthouse_intensity(
                        dist_sq, tau, decay_lambda=self.decay_lambda
                    )
                    xi_new = apply_lighthouse_attraction(xi, gs, intensity, alpha, lb_arr, ub_arr)
                    xi_new = apply_pulse_swimming(
                        xi_new, tau, personal_phases[i], lb_arr, ub_arr
                    )
                    if ages[i] >= self.age_max:
                        xi_new, fi_new = transdifferentiate(
                            xi_new, objective(xi_new), objective, lb_arr, ub_arr
                        )
                        nfev += 2
                        ages[i] = 0
                    else:
                        fi_new = objective(xi_new)
                        nfev += 1
                    if archive.is_populated():
                        xi_new, fi_new = archive_bud(
                            xi_new, fi_new, archive.sample(), objective, lb_arr, ub_arr,
                            self.levy_beta,
                        )
                        nfev += 1

                else:
                    xi_new, fi_new = senescence_refine(
                        xi, fi, best_pos, objective, lb_arr, ub_arr, tau
                    )
                    nfev += 1
                    if archive.is_populated():
                        xi_new, fi_new = archive_bud(
                            xi_new, fi_new, archive.sample(), objective, lb_arr, ub_arr,
                            self.levy_beta,
                        )
                        nfev += 1

                if fi_new < fi:
                    population[i] = xi_new
                    fitnesses[i] = fi_new
                    ages[i] = 0
                else:
                    ages[i] += 1

            self._diversity_guard(
                population, fitnesses, lb_arr, ub_arr, objective, nfev
            )

            archive.update(population, fitnesses)

            cur_best_idx = int(np.argmin(fitnesses))
            if fitnesses[cur_best_idx] < best_fit:
                best_fit = float(fitnesses[cur_best_idx])
                best_pos = population[cur_best_idx].copy()

            convergence.append(best_fit)

            if verbose and iteration % 50 == 0:
                print(
                    f"  IJA iter {iteration:>5}/{self.max_iter}  "
                    f"phase={phase}  best={best_fit:.6e}"
                )

            if callback is not None:
                if callback(iteration, best_fit, best_pos) is True:
                    break

        completed_phases.append(phase)

        return OptimizeResult(
            best_fitness=best_fit,
            best_position=best_pos,
            convergence_curve=convergence,
            n_function_evals=nfev,
            n_iterations=iteration,
            phases_completed=completed_phases,
        )

    def _initialise(
        self,
        objective: Callable,
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        population = lb + np.random.rand(self.n, dim) * (ub - lb)
        fitnesses = np.array([objective(population[i]) for i in range(self.n)])
        ages = np.zeros(self.n, dtype=int)
        phases = np.zeros(self.n, dtype=int)
        return population, fitnesses, ages, phases

    def _select_lighthouses(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        k = min(self.top_k, self.n)
        elite_idx = np.argpartition(fitnesses, k - 1)[:k]
        elite_idx = elite_idx[np.argsort(fitnesses[elite_idx])]
        return population[elite_idx]

    def _pick_lighthouse(
        self, lighthouses: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        k = len(lighthouses)
        temperature = 0.5
        ranks = np.arange(1, k + 1, dtype=float)
        weights = np.exp(-ranks / temperature)
        weights /= weights.sum()
        idx = np.random.choice(k, p=weights)
        return lighthouses[idx]

    def _diversity_guard(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: Callable,
        nfev: int,
    ) -> None:
        if compute_diversity(population, lb, ub) >= self.diversity_threshold:
            return
        n_reset = max(1, int(self.n * self.diversity_reset_frac))
        worst_idx = np.argpartition(fitnesses, -n_reset)[-n_reset:]
        for idx in worst_idx:
            population[idx] = lb + np.random.rand(population.shape[1]) * (ub - lb)
            fitnesses[idx] = objective(population[idx])
