"""
Full experiment suite: 30 independent runs × 6 algorithms × 8 benchmark functions.

Results are saved to results/experiment_results.json.
A summary table is printed to stdout on completion.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ija import IJA
from benchmarks import BENCHMARK_SUITE
from compare.algorithms import DE, GWO, PSO, SCA, WOA

N_RUNS    = 30
N         = 50
MAX_ITER  = 500
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiment_results.json")

ALGO_FACTORIES = {
    "IJA": lambda seed: IJA(n=N, max_iter=MAX_ITER, seed=seed),
    "PSO": lambda seed: PSO(n=N, max_iter=MAX_ITER, seed=seed),
    "DE":  lambda seed: DE(n=N,  max_iter=MAX_ITER, seed=seed),
    "GWO": lambda seed: GWO(n=N, max_iter=MAX_ITER, seed=seed),
    "WOA": lambda seed: WOA(n=N, max_iter=MAX_ITER, seed=seed),
    "SCA": lambda seed: SCA(n=N, max_iter=MAX_ITER, seed=seed),
}


def run_single(algo_name, spec, seed):
    solver = ALGO_FACTORIES[algo_name](seed)
    result = solver.optimize(spec.func, spec.dim, spec.lb, spec.ub)
    return {
        "best": float(result.best_fitness),
        "convergence": [float(v) for v in result.convergence_curve],
    }


def run_all_experiments():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {algo: {} for algo in ALGO_FACTORIES}

    total_tasks = len(ALGO_FACTORIES) * len(BENCHMARK_SUITE) * N_RUNS
    completed = 0
    t_start = time.time()

    for func_key, spec in BENCHMARK_SUITE.items():
        for algo_name in ALGO_FACTORIES:
            print(f"  Running {algo_name:>4s} on {spec.name:<12s} ...", end=" ", flush=True)
            run_results = []
            for run_idx in range(N_RUNS):
                run_results.append(run_single(algo_name, spec, seed=run_idx * 100))
                completed += 1
            results[algo_name][func_key] = run_results
            medians = np.median([r["best"] for r in run_results])
            print(f"median={medians:.3e}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s  ({completed} runs)")

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh)
    print(f"Results saved to {RESULTS_PATH}")
    return results


def print_summary_table(results):
    algos = list(ALGO_FACTORIES.keys())
    funcs = list(BENCHMARK_SUITE.keys())
    col_w = 13

    header = f"{'Function':<14}" + "".join(f"{a:>{col_w}}" for a in algos)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for func_key in funcs:
        spec = BENCHMARK_SUITE[func_key]
        row = f"{spec.name:<14}"
        for algo in algos:
            vals = [r["best"] for r in results.get(algo, {}).get(func_key, [])]
            med = np.median(vals) if vals else float("nan")
            row += f"{med:>{col_w}.3e}"
        print(row)

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    results = run_all_experiments()
    print_summary_table(results)
