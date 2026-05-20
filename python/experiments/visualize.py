"""
Generate all visualization figures for IJA benchmark results.

Usage:
    python experiments/visualize.py              # all figures
    python experiments/visualize.py --skip-slow  # skip 3D and sensitivity
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "experiment_results.json"
)
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# --- colour palette ---
ALGO_COLORS = {
    "IJA": "#FF6B35",
    "PSO": "#4ECDC4",
    "DE":  "#45B7D1",
    "GWO": "#96CEB4",
    "WOA": "#FFEAA7",
    "SCA": "#DDA0DD",
}
ALGO_MARKERS = {
    "IJA": "o", "PSO": "s", "DE": "^", "GWO": "D", "WOA": "v", "SCA": "P",
}
FUNC_LABELS = {
    "sphere":     "Sphere (F1)",
    "rosenbrock": "Rosenbrock (F2)",
    "rastrigin":  "Rastrigin (F3)",
    "ackley":     "Ackley (F4)",
    "griewank":   "Griewank (F5)",
    "levy":       "Levy (F6)",
    "schwefel":   "Schwefel (F7)",
    "zakharov":   "Zakharov (F8)",
}

BG_DARK  = "#1A1A2E"
BG_MID   = "#16213E"
GRID_CLR = "#2A2A4A"
TICK_CLR = "#AAAAAA"
EDGE_CLR = "#444466"
IJA_GLOW = "#FF6B35"


def _dark_ax(ax):
    ax.set_facecolor(BG_MID)
    ax.tick_params(colors=TICK_CLR, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", zorder=0)


def _save(fig, name):
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def _median_curve(results, func_name):
    curves = {}
    for algo, fdict in results.items():
        runs  = fdict.get(func_name, [])
        all_c = [r["convergence"] for r in runs if r.get("convergence")]
        if not all_c:
            continue
        min_len = min(len(c) for c in all_c)
        arr = np.array([c[:min_len] for c in all_c], dtype=float)
        arr = np.where(arr <= 0, 1e-300, arr)
        curves[algo] = np.median(arr, axis=0)
    return curves


def _normalise_curve(curve, x_out=500):
    """Normalise a convergence curve to x_out points by linear interpolation."""
    n = len(curve)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, x_out)
    return np.interp(x_new, x_old, curve)


# -----------------------------------------------------------------------
# Figure 1 — Convergence curves
# -----------------------------------------------------------------------
def plot_convergence_curves(results):
    func_names = list(FUNC_LABELS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor(BG_DARK)

    for idx, fname in enumerate(func_names):
        ax = axes[idx // 4][idx % 4]
        _dark_ax(ax)
        curves = _median_curve(results, fname)
        order  = list(curves.keys())
        # draw non-IJA first (background), IJA last (foreground)
        for algo in sorted(order, key=lambda a: a == "IJA"):
            curve  = _normalise_curve(curves[algo])
            log_c  = np.log10(np.maximum(curve, 1e-300))
            iters  = np.linspace(1, 500, len(log_c))
            is_ija = algo == "IJA"
            lw     = 3.0  if is_ija else 1.3
            zo     = 12   if is_ija else 4
            alpha  = 1.0  if is_ija else 0.75
            ms     = 6    if is_ija else 3
            ax.plot(
                iters, log_c,
                label=algo,
                color=ALGO_COLORS.get(algo, "#FFF"),
                linewidth=lw, marker=ALGO_MARKERS.get(algo, "o"),
                markevery=50, markersize=ms, zorder=zo, alpha=alpha,
            )
            # glow effect for IJA
            if is_ija:
                ax.plot(iters, log_c, color=IJA_GLOW, linewidth=6,
                        alpha=0.18, zorder=zo - 1)

        ax.set_title(FUNC_LABELS[fname], color="white", fontsize=10, pad=6)
        ax.set_xlabel("Iteration (normalised)", color=TICK_CLR, fontsize=8)
        ax.set_ylabel("log₁₀(Best Fitness)",    color=TICK_CLR, fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=6,
        facecolor=BG_DARK, edgecolor=EDGE_CLR, labelcolor="white",
        fontsize=9, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "IJA vs Competitors — Convergence Curves  (Median of 30 Runs)",
        color="white", fontsize=14, y=0.99, fontweight="bold",
    )
    # footnote
    fig.text(
        0.5, -0.01,
        "IJA: n=100, max_iter=2000 (recommended)  |  Competitors: n=50, max_iter=500 (standard)",
        ha="center", va="bottom", color="#888888", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return _save(fig, "convergence_curves.png")


# -----------------------------------------------------------------------
# Figure 2 — Rank heatmap
# -----------------------------------------------------------------------
def _rank_matrix(results):
    func_names = list(FUNC_LABELS.keys())
    algos      = list(ALGO_COLORS.keys())
    mat        = np.full((len(algos), len(func_names)), np.nan)
    for fi, fname in enumerate(func_names):
        medians = {}
        for algo in algos:
            vals = [r["best"] for r in results.get(algo, {}).get(fname, [])
                    if r.get("best") is not None]
            if vals:
                medians[algo] = np.median(vals)
        if not medians:
            continue
        for rank, algo in enumerate(sorted(medians, key=lambda a: medians[a]), 1):
            mat[algos.index(algo)][fi] = rank
    return mat, algos, func_names


def plot_rank_heatmap(results):
    mat, algos, func_names = _rank_matrix(results)
    n_algos = len(algos)

    fig, ax = plt.subplots(figsize=(15, 5.5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    masked = np.ma.masked_invalid(mat)
    cmap   = cm.get_cmap("RdYlGn_r", n_algos)
    im     = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=1, vmax=n_algos)

    ax.set_xticks(range(len(func_names)))
    ax.set_xticklabels(
        [FUNC_LABELS[f] for f in func_names],
        color="white", fontsize=9, rotation=18, ha="right",
    )
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels(algos, color="white", fontsize=11)

    for i in range(n_algos):
        for j in range(len(func_names)):
            v = mat[i][j]
            if not np.isnan(v):
                iv = int(v)
                text_color = "#000000" if iv <= 2 else "#FFFFFF"
                weight     = "bold"    if iv == 1 else "normal"
                label      = f"★ {iv}" if iv == 1 else str(iv)
                ax.text(j, i, label, ha="center", va="center",
                        color=text_color, fontsize=11, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Rank  (1 = best)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_CLR)

    ax.set_title(
        "Algorithm Ranking Heatmap  (Median Final Fitness, 30 Independent Runs)",
        color="white", fontsize=13, pad=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.03,
        "IJA: n=100, max_iter=2000 (recommended)  |  Competitors: n=50, max_iter=500 (standard)",
        ha="center", va="bottom", color="#888888", fontsize=8,
    )
    fig.tight_layout()
    return _save(fig, "rank_heatmap.png")


# -----------------------------------------------------------------------
# Figure 3 — Average rank bar chart
# -----------------------------------------------------------------------
def plot_average_rank_bar(results):
    mat, algos, _ = _rank_matrix(results)
    avg            = np.nanmean(mat, axis=1)
    order          = np.argsort(avg)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_DARK)
    _dark_ax(ax)

    sorted_algos  = [algos[i] for i in order]
    sorted_avgs   = avg[order]
    colors        = [ALGO_COLORS.get(a, "#888") for a in sorted_algos]
    edge_colors   = ["white" if a == "IJA" else EDGE_CLR for a in sorted_algos]
    edge_widths   = [2.0    if a == "IJA" else 1.0      for a in sorted_algos]

    bars = ax.bar(
        sorted_algos, sorted_avgs,
        color=colors, edgecolor=edge_colors, linewidth=edge_widths, width=0.6,
    )
    for bar, val, algo in zip(bars, sorted_avgs, sorted_algos):
        ypos   = bar.get_height() + 0.05
        weight = "bold" if algo == "IJA" else "normal"
        label  = f"★ {val:.2f}" if algo == "IJA" else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                label, ha="center", va="bottom", color="white",
                fontsize=11, fontweight=weight)

    ax.set_ylabel("Average Rank  (lower = better)", color=TICK_CLR, fontsize=10)
    ax.set_title(
        "Average Ranking Across 8 Benchmark Functions",
        color="white", fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, len(algos) + 0.8)
    fig.tight_layout()
    return _save(fig, "average_rank_bar.png")


# -----------------------------------------------------------------------
# Figure 4 — Box plots
# -----------------------------------------------------------------------
def plot_boxplots(results):
    func_names = list(FUNC_LABELS.keys())
    algos      = list(ALGO_COLORS.keys())

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor(BG_DARK)

    for idx, fname in enumerate(func_names):
        ax = axes[idx // 4][idx % 4]
        _dark_ax(ax)
        data, valid = [], []
        for algo in algos:
            finals = [r["best"] for r in results.get(algo, {}).get(fname, [])
                      if r.get("best") is not None]
            if finals:
                data.append(finals)
                valid.append(algo)
        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data, patch_artist=True, notch=False, widths=0.55,
            medianprops=dict(color="white", linewidth=2.0),
        )
        for patch, algo in zip(bp["boxes"], valid):
            patch.set_facecolor(ALGO_COLORS.get(algo, "#888"))
            patch.set_alpha(0.85)
            if algo == "IJA":
                patch.set_linewidth(2.0)
                patch.set_edgecolor("white")
        for el in ["whiskers", "caps", "fliers"]:
            for item in bp[el]:
                item.set_color(TICK_CLR)

        ax.set_xticks(range(1, len(valid) + 1))
        ax.set_xticklabels(valid, color=TICK_CLR, fontsize=8)
        ax.set_title(FUNC_LABELS[fname], color="white", fontsize=10, pad=6)
        ax.set_ylabel("Final Fitness",   color=TICK_CLR, fontsize=8)
        ax.set_yscale("symlog", linthresh=1e-10)

    fig.suptitle(
        "IJA vs Competitors — Final Fitness Distribution  (30 Independent Runs)",
        color="white", fontsize=14, y=0.99, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    return _save(fig, "boxplots.png")


# -----------------------------------------------------------------------
# Figure 5 — Lifecycle diagram
# -----------------------------------------------------------------------
def plot_lifecycle_diagram():
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    phases = [
        (0.0,  0.15, "#4A90D9", "Phase 0\nPolyp\n[0, 0.15T]",        "Lévy-Flight\nExploration"),
        (0.15, 0.50, "#7B68EE", "Phase 1\nStrobilation\n[0.15, 0.5T]","Adaptive\nEphyra Budding"),
        (0.50, 0.85, "#FF6B35", "Phase 2\nMedusa\n[0.5, 0.85T]",      "Lighthouse\nAttraction"),
        (0.85, 1.00, "#E74C3C", "Phase 3\nSenescence\n[0.85T, T]",    "Gaussian\nRefinement"),
    ]
    for x0, x1, color, label, desc in phases:
        rect = plt.Rectangle(
            (x0 + 0.005, 0.25), (x1 - x0) - 0.01, 0.5,
            facecolor=color, alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3,
        )
        ax.add_patch(rect)
        mx = (x0 + x1) / 2
        ax.text(mx, 0.52, label, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=5, multialignment="center")
        ax.text(mx, 0.18, desc, ha="center", va="center",
                color="#DDDDDD", fontsize=8, zorder=5, multialignment="center")
    for xb in [0.15, 0.50, 0.85]:
        ax.annotate(
            "", xy=(xb + 0.015, 0.50), xytext=(xb - 0.005, 0.50),
            arrowprops=dict(arrowstyle="->", color="white", lw=2), zorder=6,
        )
    ax.text(
        0.5, 0.92,
        "Immortal Jellyfish Algorithm (IJA) — Four-Phase Lifecycle",
        ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=5,
    )
    fig.tight_layout()
    return _save(fig, "lifecycle_diagram.png")


# -----------------------------------------------------------------------
# Figure 6 — 3-D trajectory on Rastrigin (slow)
# -----------------------------------------------------------------------
def plot_3d_trajectory():
    from benchmarks.functions import rastrigin
    from ija import IJA

    x1 = np.linspace(-5.12, 5.12, 100)
    x2 = np.linspace(-5.12, 5.12, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.vectorize(lambda a, b: rastrigin(np.array([a, b])))(X1, X2)

    traj_x, traj_y = [], []

    def tracked(x):
        traj_x.append(float(x[0]))
        traj_y.append(float(x[1]))
        return rastrigin(x)

    IJA(n=30, max_iter=300, seed=42).optimize(tracked, 2, -5.12, 5.12)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_DARK)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_MID)
    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.55, linewidth=0)
    fig.colorbar(surf, ax=ax, fraction=0.025, pad=0.04)
    if traj_x:
        step = max(1, len(traj_x) // 60)
        tx, ty = traj_x[::step], traj_y[::step]
        tz = [rastrigin(np.array([a, b])) for a, b in zip(tx, ty)]
        ax.plot(tx, ty, tz, color="#FF6B35", linewidth=2, zorder=10, label="IJA trajectory")
        ax.scatter(tx[-1], ty[-1], tz[-1], color="red", s=80, zorder=11, label="Final best")
    ax.set_xlabel("x₁", color=TICK_CLR)
    ax.set_ylabel("x₂", color=TICK_CLR)
    ax.set_zlabel("f(x)", color=TICK_CLR)
    ax.tick_params(colors=TICK_CLR, labelsize=7)
    ax.set_title("IJA Trajectory on Rastrigin (2D)", color="white", fontsize=11)
    ax.legend(facecolor=BG_DARK, edgecolor=EDGE_CLR, labelcolor="white", fontsize=8)
    return _save(fig, "trajectory_3d.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def generate_all(skip_slow=False):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run  python python/experiments/run_all.py  first.")
        return
    with open(RESULTS_PATH) as fh:
        results = json.load(fh)
    print("=== Generating visualizations ===")
    plot_convergence_curves(results)
    plot_boxplots(results)
    plot_rank_heatmap(results)
    plot_average_rank_bar(results)
    plot_lifecycle_diagram()
    if not skip_slow:
        print("Generating 3-D trajectory (may take ~30 s) ...")
        plot_3d_trajectory()
    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip 3D trajectory plot")
    args = parser.parse_args()
    generate_all(skip_slow=args.skip_slow)
