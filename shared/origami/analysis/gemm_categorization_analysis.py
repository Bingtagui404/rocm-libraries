#!/usr/bin/env python3
"""
GEMM Categorization Analysis — Arithmetic Intensity Distribution

This script analyzes the 50-category GEMM categorization scheme and produces
visualizations showing how categories map to arithmetic intensity regimes,
providing mathematical justification for the design.

Mathematical background:
  GEMM: C(M,N) = A(M,K) * B(K,N)
  FLOPs = 2*M*N*K
  Bytes  = (M*K + K*N + M*N) * bpe
  AI     = FLOPs / Bytes = 2*M*N*K / ((M*K + K*N + M*N) * bpe)

  Equivalently:  1/AI = (bpe/2) * (1/M + 1/N + 1/K)

  This harmonic relationship means the smallest dimension dominates,
  pulling AI down.  The categorization exploits this structure:
    - 5 M ranges x 5 N ranges capture tile shape / parallelism
    - 2 K ranges  separate memory-bound from compute-bound regimes
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Constants matching categorization.hpp ──────────────────────────────────────

MN_BOUNDS = [64, 256, 1024, 4096, float("inf")]
K_BOUNDS  = [2048, float("inf")]

MN_LABELS = ["tiny\n[1,64]", "small\n[65,256]", "medium\n[257,1024]",
             "large\n[1025,4096]", "xlarge\n[4097,inf)"]
K_LABELS  = ["short_k [1,2048]", "long_k [2049,inf)"]

NUM_MN = len(MN_BOUNDS)
NUM_K  = len(K_BOUNDS)
NUM_SIZE_CATEGORIES = NUM_MN * NUM_MN * NUM_K  # 50 (without batch)
NUM_CATEGORIES = NUM_SIZE_CATEGORIES  # analysis uses size categories only


def ai(m, n, k, bpe=2.0):
    """Compute GEMM arithmetic intensity (ops/byte)."""
    denom = (m * k + k * n + m * n) * bpe
    if denom == 0:
        return 0.0
    return 2.0 * m * n * k / denom


def mn_lower(i):
    return 1 if i == 0 else MN_BOUNDS[i - 1] + 1


def mn_upper(i):
    return MN_BOUNDS[i]


def k_lower(i):
    return 1 if i == 0 else K_BOUNDS[i - 1] + 1


def k_upper(i):
    return K_BOUNDS[i]


def representative(lo, hi, cap=16384.0):
    """Geometric mean, capping infinity at `cap`."""
    h = cap if hi == float("inf") else hi
    return math.sqrt(lo * h)


def cat_id(mi, ni, ki):
    return mi * NUM_MN * NUM_K + ni * NUM_K + ki


# ── Figure 1: Heatmap of representative AI across all 50 categories ──────────

def fig1_ai_heatmap(outdir):
    """
    5x5 heatmap for each K regime showing AI at category centers.
    Side-by-side: short_k vs long_k.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ki, ax in enumerate(axes):
        data = np.zeros((NUM_MN, NUM_MN))
        for mi in range(NUM_MN):
            for ni in range(NUM_MN):
                m = representative(mn_lower(mi), mn_upper(mi))
                n = representative(mn_lower(ni), mn_upper(ni))
                k = representative(k_lower(ki), k_upper(ki))
                data[mi, ni] = ai(m, n, k)

        im = ax.imshow(data, cmap="YlOrRd", norm=mcolors.LogNorm(vmin=1, vmax=5000))
        ax.set_xticks(range(NUM_MN))
        ax.set_xticklabels(MN_LABELS, fontsize=8)
        ax.set_yticks(range(NUM_MN))
        ax.set_yticklabels(MN_LABELS, fontsize=8)
        ax.set_xlabel("N range", fontsize=11)
        ax.set_ylabel("M range", fontsize=11)
        ax.set_title(f"K = {K_LABELS[ki]}", fontsize=12, fontweight="bold")

        for mi in range(NUM_MN):
            for ni in range(NUM_MN):
                val = data[mi, ni]
                color = "white" if val > 100 else "black"
                ax.text(ni, mi, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Arithmetic Intensity (ops/byte, BF16)", shrink=0.8)
    fig.suptitle("Representative Arithmetic Intensity per Category\n"
                 "AI = 2MNK / ((MK+KN+MN)*bpe),  bpe=2 (BF16)",
                 fontsize=14, fontweight="bold")

    path = os.path.join(outdir, "fig1_ai_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: AI distribution histogram across sampled problems ──────────────

def fig2_ai_distribution(outdir):
    """
    Sample many random problems, compute their AI, and show a histogram
    colored by category, demonstrating that categories group similar AI values.
    """
    rng = np.random.default_rng(42)
    n_samples = 50000

    m_vals = np.exp(rng.uniform(np.log(1), np.log(32768), n_samples)).astype(int).clip(1)
    n_vals = np.exp(rng.uniform(np.log(1), np.log(32768), n_samples)).astype(int).clip(1)
    k_vals = np.exp(rng.uniform(np.log(1), np.log(65536), n_samples)).astype(int).clip(1)

    ais = np.array([ai(m, n, k) for m, n, k in zip(m_vals, n_vals, k_vals)])

    def classify_mn_py(d):
        for i, ub in enumerate(MN_BOUNDS):
            if d <= ub:
                return i
        return NUM_MN - 1

    def classify_k_py(d):
        for i, ub in enumerate(K_BOUNDS):
            if d <= ub:
                return i
        return NUM_K - 1

    cat_ids = np.array([cat_id(classify_mn_py(m), classify_mn_py(n), classify_k_py(k))
                        for m, n, k in zip(m_vals, n_vals, k_vals)])

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    bins = np.logspace(np.log10(0.5), np.log10(10000), 60)

    short_mask = np.array([classify_k_py(k) == 0 for k in k_vals])
    ax.hist(ais[short_mask], bins=bins, alpha=0.6, label="short_k [1,2048]",
            color="#2196F3", edgecolor="white", linewidth=0.3)
    ax.hist(ais[~short_mask], bins=bins, alpha=0.6, label="long_k [2049,inf)",
            color="#FF5722", edgecolor="white", linewidth=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops/byte, BF16)", fontsize=12)
    ax.set_ylabel("Number of sampled problems", fontsize=12)
    ax.set_title("Distribution of Arithmetic Intensity across 50K random GEMM problems\n"
                 "(log-uniform sampling of M, N, K)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    ax.axvline(x=120, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(130, ax.get_ylim()[1] * 0.85, "MI250X\n(gfx90a)\n~120",
            fontsize=9, color="green", va="top")
    ax.axvline(x=247, color="purple", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(260, ax.get_ylim()[1] * 0.7, "MI300X\n(gfx942)\n~247",
            fontsize=9, color="purple", va="top")
    ax.axvline(x=288, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(300, ax.get_ylim()[1] * 0.55, "MI350X\n(gfx950)\n~288",
            fontsize=9, color="red", va="top")

    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig2_ai_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: AI vs problem size for square GEMMs ────────────────────────────

def fig3_ai_vs_dimension(outdir):
    """
    Shows how AI varies with dimension size for square GEMMs (M=N=D)
    at different K values.  Demonstrates the 1/AI = bpe/2 * (2/D + 1/K)
    relationship and why K=2048 is a natural boundary.
    """
    D = np.logspace(0, 5, 500)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    k_values = [64, 256, 1024, 2048, 4096, 8192, 32768]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(k_values)))

    for k, color in zip(k_values, colors):
        ai_vals = [ai(d, d, k) for d in D]
        style = "-" if k <= 2048 else "--"
        lw = 2.5 if k == 2048 else 1.5
        ax.plot(D, ai_vals, style, color=color, linewidth=lw, label=f"K={k}")

    ax.axhline(y=120, color="green", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(1.5, 130, "MI250X (gfx90a) roofline ~120", fontsize=9, color="green")
    ax.axhline(y=247, color="purple", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(1.5, 260, "MI300X (gfx942) roofline ~247", fontsize=9, color="purple")
    ax.axhline(y=288, color="red", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(1.5, 310, "MI350X (gfx950) roofline ~288", fontsize=9, color="red")

    for i, ub in enumerate(MN_BOUNDS[:-1]):
        ax.axvline(x=ub, color="gray", linestyle="-.", linewidth=0.8, alpha=0.5)
        ax.text(ub * 1.05, 5, MN_LABELS[i].replace("\n", " "), fontsize=7,
                color="gray", rotation=90, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem dimension D (M = N = D)", fontsize=12)
    ax.set_ylabel("Arithmetic Intensity (ops/byte, BF16)", fontsize=12)
    ax.set_title("AI = 2D²K / ((2DK + D²) * 2)  for square GEMMs (M=N=D)\n"
                 "Solid: short_k regime, Dashed: long_k regime",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, title="K value", loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1, 100000)
    ax.set_ylim(0.5, 20000)

    path = os.path.join(outdir, "fig3_ai_vs_dimension.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Category map — all 50 categories with AI color ─────────────────

def fig4_category_map(outdir):
    """
    Grid visualization of all 50 categories showing:
    - Position = (M_range, N_range)
    - Left/right within cell = short_k / long_k
    - Color = representative AI
    """
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)

    norm = mcolors.LogNorm(vmin=1, vmax=5000)
    cmap = plt.cm.RdYlGn

    cell_w = 1.0
    cell_h = 1.0
    gap = 0.05

    for mi in range(NUM_MN):
        for ni in range(NUM_MN):
            for ki in range(NUM_K):
                m = representative(mn_lower(mi), mn_upper(mi))
                n = representative(mn_lower(ni), mn_upper(ni))
                k = representative(k_lower(ki), k_upper(ki))
                intensity = ai(m, n, k)
                cid = cat_id(mi, ni, ki)

                x = ni * (cell_w + gap) + ki * cell_w / 2
                y = (NUM_MN - 1 - mi) * (cell_h + gap)
                w = cell_w / 2

                color = cmap(norm(intensity))
                rect = plt.Rectangle((x, y), w - 0.02, cell_h, facecolor=color,
                                     edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)

                text_color = "white" if intensity > 200 else "black"
                ax.text(x + w / 2 - 0.01, y + cell_h * 0.65,
                        f"#{cid}", ha="center", va="center",
                        fontsize=6, color=text_color, fontweight="bold")
                ax.text(x + w / 2 - 0.01, y + cell_h * 0.35,
                        f"{intensity:.0f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)

    for ni in range(NUM_MN):
        x_center = ni * (cell_w + gap) + cell_w / 2
        ax.text(x_center, -0.4, MN_LABELS[ni].replace("\n", " "),
                ha="center", va="top", fontsize=9)
    for mi in range(NUM_MN):
        y_center = (NUM_MN - 1 - mi) * (cell_h + gap) + cell_h / 2
        ax.text(-0.3, y_center, MN_LABELS[mi].replace("\n", " "),
                ha="right", va="center", fontsize=9)

    ax.set_xlabel("N dimension range", fontsize=12, labelpad=30)
    ax.set_ylabel("M dimension range", fontsize=12, labelpad=20)
    ax.set_xlim(-0.5, NUM_MN * (cell_w + gap))
    ax.set_ylim(-0.8, NUM_MN * (cell_h + gap))
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor="#2196F3", edgecolor="black", label="Left half: short_k [1,2048]"),
        Patch(facecolor="#FF5722", edgecolor="black", label="Right half: long_k [2049,inf)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              bbox_to_anchor=(0.0, 1.08))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Arithmetic Intensity (ops/byte, BF16)", fontsize=10)

    fig.suptitle("GEMM Category Map — 50 Categories\n"
                 "Each cell = (M_range, N_range); left/right = K regime\n"
                 "Number = category ID, value = representative AI",
                 fontsize=14, fontweight="bold", y=1.02)

    path = os.path.join(outdir, "fig4_category_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: 1/AI decomposition — showing harmonic structure ────────────────

def fig5_reciprocal_ai(outdir):
    """
    Visualizes the additive decomposition 1/AI = bpe/2 * (1/M + 1/N + 1/K)
    as stacked bars for each category, showing which dimension dominates.
    """
    categories = list(range(NUM_CATEGORIES))
    inv_m = []
    inv_n = []
    inv_k = []
    cat_labels = []

    for cid in categories:
        ki = cid % NUM_K
        ni = (cid // NUM_K) % NUM_MN
        mi = cid // (NUM_K * NUM_MN)

        m = representative(mn_lower(mi), mn_upper(mi))
        n = representative(mn_lower(ni), mn_upper(ni))
        k = representative(k_lower(ki), k_upper(ki))

        inv_m.append(1.0 / m)
        inv_n.append(1.0 / n)
        inv_k.append(1.0 / k)
        cat_labels.append(str(cid))

    inv_m = np.array(inv_m)
    inv_n = np.array(inv_n)
    inv_k = np.array(inv_k)

    sort_idx = np.argsort(inv_m + inv_n + inv_k)[::-1]
    inv_m = inv_m[sort_idx]
    inv_n = inv_n[sort_idx]
    inv_k = inv_k[sort_idx]
    cat_labels = [cat_labels[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

    x = np.arange(NUM_CATEGORIES)
    ax.bar(x, inv_m, label="1/M", color="#2196F3", edgecolor="white", linewidth=0.3)
    ax.bar(x, inv_n, bottom=inv_m, label="1/N", color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax.bar(x, inv_k, bottom=inv_m + inv_n, label="1/K", color="#FF9800",
           edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=5, rotation=90)
    ax.set_xlabel("Category ID (sorted by 1/AI descending)", fontsize=11)
    ax.set_ylabel("1/M + 1/N + 1/K  (proportional to 1/AI)", fontsize=11)
    ax.set_title("Reciprocal AI Decomposition by Category\n"
                 "1/AI = (bpe/2) * (1/M + 1/N + 1/K) — the smallest dimension dominates",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(outdir, "fig5_reciprocal_ai.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 6: Roofline regimes per category ──────────────────────────────────

def fig6_roofline_regime(outdir):
    """
    Classify each category as memory-bound, transitional, or compute-bound
    relative to MI300X and MI250X rooflines.
    """
    MI250X_ROOFLINE = 120   # gfx90a: 383 TFLOPS / 3.2 TB/s
    MI300X_ROOFLINE = 247   # gfx942: 1307 TFLOPS / 5.3 TB/s
    MI350X_ROOFLINE = 288   # gfx950: ~2300 TFLOPS / 8.0 TB/s

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)

    for idx, (roofline, hw_name) in enumerate([(MI250X_ROOFLINE, "MI250X (gfx90a)"),
                                                (MI300X_ROOFLINE, "MI300X (gfx942)"),
                                                (MI350X_ROOFLINE, "MI350X (gfx950)")]):
        ax = axes[idx]
        data = np.zeros((NUM_MN, NUM_MN, NUM_K))

        for mi in range(NUM_MN):
            for ni in range(NUM_MN):
                for ki in range(NUM_K):
                    m = representative(mn_lower(mi), mn_upper(mi))
                    n = representative(mn_lower(ni), mn_upper(ni))
                    k = representative(k_lower(ki), k_upper(ki))
                    data[mi, ni, ki] = ai(m, n, k)

        regime_colors = {"Memory\nBound": "#2196F3", "Transitional": "#FFC107",
                         "Compute\nBound": "#4CAF50"}

        bar_data = {r: np.zeros(NUM_MN) for r in regime_colors}
        for mi in range(NUM_MN):
            for ni in range(NUM_MN):
                for ki in range(NUM_K):
                    val = data[mi, ni, ki]
                    if val < roofline * 0.5:
                        bar_data["Memory\nBound"][mi] += 1
                    elif val < roofline * 1.5:
                        bar_data["Transitional"][mi] += 1
                    else:
                        bar_data["Compute\nBound"][mi] += 1

        x = np.arange(NUM_MN)
        bottom = np.zeros(NUM_MN)
        for regime, color in regime_colors.items():
            ax.bar(x, bar_data[regime], bottom=bottom, label=regime, color=color,
                   edgecolor="white", linewidth=0.5)
            bottom += bar_data[regime]

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in MN_LABELS], fontsize=8)
        ax.set_xlabel("M dimension range", fontsize=11)
        ax.set_ylabel("Number of categories", fontsize=11)
        ax.set_title(f"{hw_name}\nRoofline = {roofline} ops/byte (BF16)", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Roofline Regime Distribution across M ranges\n"
                 "(each M range has 5 N ranges x 2 K ranges = 10 categories)",
                 fontsize=14, fontweight="bold")

    path = os.path.join(outdir, "fig6_roofline_regime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Print summary table ──────────────────────────────────────────────────────

def print_summary_table():
    """Print a text summary of all 50 categories with their AI values."""
    print("\n" + "=" * 90)
    print(f"{'ID':>4}  {'M range':>16}  {'N range':>16}  {'K range':>18}  "
          f"{'AI (BF16)':>10}  {'Regime':>12}")
    print("=" * 90)

    for cid in range(NUM_CATEGORIES):
        ki = cid % NUM_K
        ni = (cid // NUM_K) % NUM_MN
        mi = cid // (NUM_K * NUM_MN)

        m = representative(mn_lower(mi), mn_upper(mi))
        n = representative(mn_lower(ni), mn_upper(ni))
        k = representative(k_lower(ki), k_upper(ki))
        intensity = ai(m, n, k)

        m_str = f"[{mn_lower(mi)}, {mn_upper(mi) if mn_upper(mi) != float('inf') else 'inf':>5}]"
        n_str = f"[{mn_lower(ni)}, {mn_upper(ni) if mn_upper(ni) != float('inf') else 'inf':>5}]"
        k_str = f"[{k_lower(ki)}, {k_upper(ki) if k_upper(ki) != float('inf') else 'inf':>5}]"

        if intensity < 60:
            regime = "MEM-BOUND"
        elif intensity < 300:
            regime = "TRANSITIONAL"
        else:
            regime = "COMPUTE-BOUND"

        print(f"{cid:>4}  {m_str:>16}  {n_str:>16}  {k_str:>18}  {intensity:>10.1f}  {regime:>12}")

    print("=" * 90)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    print("GEMM Categorization Analysis")
    print(f"Total categories: {NUM_CATEGORIES}")
    print(f"Output directory:  {outdir}\n")

    print_summary_table()

    print("\nGenerating figures...")
    fig1_ai_heatmap(outdir)
    fig2_ai_distribution(outdir)
    fig3_ai_vs_dimension(outdir)
    fig4_category_map(outdir)
    fig5_reciprocal_ai(outdir)
    fig6_roofline_regime(outdir)

    print("\nDone. All figures saved to:", outdir)
