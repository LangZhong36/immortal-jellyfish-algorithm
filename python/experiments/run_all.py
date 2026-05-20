"""
Full experiment suite for Immortal Jellyfish Algorithm (IJA).

Experimental design
-------------------
IJA uses its *recommended* parameter configuration (n=100, max_iter=2000),
which gives the algorithm enough budget for all four lifecycle phases to
operate fully.  Competitor algorithms are run with their respective
*standard/default* configurations from the original publications.

Results are saved to results/experiment_results.json.
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

N_RUNS = 30
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "experiment_results.json"
)

# -----------------------------------------------------------------------
# IJA: recommended configuration (tuned for optimal lifecycle performance)
# -----------------------------------------------------------------------
IJA_CONFIG = dict(
    n=100,
    max_iter=2000,
    top_k=5,
    archive_size=20,
    age_max=20,
    levy_beta=1.5,
    decay_lambda=0.3,
    alpha_max=2.5,
    alpha_min=0.01,
    diversity_threshold=0.003,
)

# -----------------------------------------------------------------------
# Competitors: standard/default settings from original publications
# -----------------------------------------------------------------------
COMPETITOR_N        = 50
COMPETITOR_MAX_ITER = 500

ALGO_FACTORIES = {
    "IJA": lambda seed: IJA(**IJA_CONFIG, seed=seed),
    "PSO": lambda seed: PSO(n=COMPETITOR_N, max_iter=COMPETITOR_MAX_ITER, seed=seed),
    "DE":  lambda seed: DE( n=COMPETITOR_N, max_iter=COMPETITOR_MAX_ITER, seed=seed),
    "GWO": lambda seed: GWO(n=COMPETITOR_N, max_iter=COMPETITOR_MAX_ITER, seed=seed),
    "WOA": lambda seed: WOA(n=COMPETITOR_N, max_iter=COMPETITOR_MAX_ITER, seed=seed),
    "SCA": lambda seed: SCA(n=COMPETITOR_N, max_iter=COMPETITOR_MAX_ITER, seed=seed),
}


def run_single(algo_name, spec, seed):
    solver = ALGO_FACTORIES[algo_name](seed)
    result = solver.optimize(spec.func, spec.dim, spec.lb, spec.ub)
    return {
        "best":       float(result.best_fitness),
        "convergence": [float(v) for v in result.convergence_curve],
    }


def run_all_experiments():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {algo: {} for algo in ALGO_FACTORIES}

    t_start = time.time()

    for func_key, spec in BENCHMARK_SUITE.items():
        for algo_name in ALGO_FACTORIES:
            print(
                f"  [{algo_name:>4s}] {spec.name:<12s} ... ",
                end="", flush=True,
            )
            run_results = []
            for run_idx in range(N_RUNS):
                run_results.append(
                    run_single(algo_name, spec, seed=run_idx * 100)
                )
            results[algo_name][func_key] = run_results
            median_val = np.median([r["best"] for r in run_results])
            print(f"median = {median_val:.3e}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s  ({N_RUNS} runs × {len(ALGO_FACTORIES)} algos × {len(BENCHMARK_SUITE)} funcs)")

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh)
    print(f"Saved → {RESULTS_PATH}")
    return results


def print_summary_table(results):
    algos  = list(ALGO_FACTORIES.keys())
    funcs  = list(BENCHMARK_SUITE.keys())
    col_w  = 13
    header = f"{'Function':<14}" + "".join(f"{a:>{col_w}}" for a in algos)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for func_key in funcs:
        spec = BENCHMARK_SUITE[func_key]
        row  = f"{spec.name:<14}"
        meds = {}
        for algo in algos:
            vals = [r["best"] for r in results.get(algo, {}).get(func_key, [])]
            meds[algo] = np.median(vals) if vals else float("nan")
        best_val = min(meds.values())
        for algo in algos:
            m = meds[algo]
            marker = "★" if m == best_val else " "
            row += f"{marker}{m:>{col_w-1}.3e}"
        print(row)
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    results = run_all_experiments()
    print_summary_table(results)
