"""
tier1_metrics.py
-----------------
Tier 1 transfer assessment — Step 3.

For each source-target city pair, computes:
  1. MMD (Maximum Mean Discrepancy) with RBF kernel, bandwidth via median heuristic
  2. Overlap ratio at k=10 and k=20 (fraction of target embeddings covered by
     source k-NN neighbourhood, threshold at 95th percentile of within-source
     pairwise distances)

Outputs:
  - metrics_all_pairs.csv          : long-form table (source, target, mmd, overlap_k10, overlap_k20)
  - matrix_mmd.csv                 : 7×7 pivot table
  - matrix_overlap_k10.csv         : 7×7 pivot table
  - matrix_overlap_k20.csv         : 7×7 pivot table
  - heatmap_mmd.png
  - heatmap_overlap_k10.png
  - heatmap_overlap_k20.png

Usage (defaults):
    python tier1_metrics.py

Usage (override paths):
    python tier1_metrics.py \
        --embeddings_dir  results/tier1/embeddings/ \
        --output_dir      results/tier1/metrics/
"""

from __future__ import annotations

import os
import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Project-level defaults
# ---------------------------------------------------------------------------

TRANSFER_PATH = os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITY_CODES = ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI", "MUM"]

K_VALUES = [10, 20]

# Subsample cap per city to keep MMD tractable (MMD is O(n²))
# Set to None to use all points — fine for ≤2000 images
MMD_MAX_SAMPLES = 1500


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute RBF kernel bandwidth via the median heuristic on the pooled set.
    σ = median of all pairwise distances / sqrt(2).
    Subsample for speed if the combined set is large.
    """
    combined = np.vstack([X, Y])
    n = len(combined)
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=2000, replace=False)
        combined = combined[idx]

    dists = cdist(combined, combined, metric="euclidean")
    # Upper triangle only, excluding diagonal
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = np.median(upper) / np.sqrt(2.0)
    return max(sigma, 1e-6)  # guard against degenerate case


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix K(X, Y)."""
    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD² estimate between source embeddings X and target embeddings Y.

    MMD²(X,Y) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]

    We return the signed value (can be slightly negative due to unbiased
    estimator); callers may clip to 0 if needed.
    """
    # Optional subsampling for speed
    rng = np.random.default_rng(42)
    if MMD_MAX_SAMPLES and len(X) > MMD_MAX_SAMPLES:
        X = X[rng.choice(len(X), size=MMD_MAX_SAMPLES, replace=False)]
    if MMD_MAX_SAMPLES and len(Y) > MMD_MAX_SAMPLES:
        Y = Y[rng.choice(len(Y), size=MMD_MAX_SAMPLES, replace=False)]

    sigma = median_bandwidth(X, Y)

    nx, ny = len(X), len(Y)

    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    # Unbiased estimator: zero out diagonal of Kxx and Kyy
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term1 = Kxx.sum() / (nx * (nx - 1))
    term2 = Kyy.sum() / (ny * (ny - 1))
    term3 = Kxy.mean()

    return float(term1 + term2 - 2.0 * term3)


# ---------------------------------------------------------------------------
# Overlap ratio
# ---------------------------------------------------------------------------

def compute_overlap_ratio(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    k: int,
    coverage_percentile: float = 95.0,
) -> float:
    """
    Fraction of target embeddings that fall within the k-NN neighbourhood
    of the source embeddings.

    Threshold is defined as the `coverage_percentile`-th percentile of
    within-source k-NN distances — i.e. a data-driven radius that covers
    most of the source distribution.

    Args:
        source_emb : (N_src, D)
        target_emb : (N_tgt, D)
        k          : number of neighbours
        coverage_percentile : percentile of within-source distances used as threshold

    Returns:
        overlap ratio in [0, 1]
    """
    k_src = min(k + 1, len(source_emb))  # +1 because point is its own neighbour

    # Within-source k-NN distances (exclude self)
    nn_src = NearestNeighbors(n_neighbors=k_src, metric="euclidean", n_jobs=-1)
    nn_src.fit(source_emb)
    src_dists, _ = nn_src.kneighbors(source_emb)
    src_dists = src_dists[:, 1:]  # exclude self (distance 0)

    # Threshold: Nth percentile of mean k-NN distance per source point
    mean_knn_dists = src_dists.mean(axis=1)
    threshold = np.percentile(mean_knn_dists, coverage_percentile)

    # Target-to-source k-NN distances
    k_tgt = min(k, len(source_emb))
    nn_tgt = NearestNeighbors(n_neighbors=k_tgt, metric="euclidean", n_jobs=-1)
    nn_tgt.fit(source_emb)
    tgt_dists, _ = nn_tgt.kneighbors(target_emb)
    mean_tgt_dists = tgt_dists.mean(axis=1)

    covered = (mean_tgt_dists <= threshold).sum()
    return float(covered) / len(target_emb)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "viridis_r",
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a 7×7 heatmap with consistent dark styling."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        linewidths=0.5,
        linecolor="#2d2d4e",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8, "pad": 0.02},
        annot_kws={"size": 8, "color": "white"},
    )

    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.set_xlabel("Target city", color="#aaaacc", fontsize=10)
    ax.set_ylabel("Source city", color="#aaaacc", fontsize=10)
    ax.tick_params(colors="#aaaacc", labelsize=9)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("#444466")

    # Diagonal (self-transfer) annotation
    n = len(matrix)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color="#2d2d4e", zorder=3))
        ax.text(i + 0.5, i + 0.5, "—", ha="center", va="center",
                color="#666688", fontsize=9, zorder=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    embeddings_dir = Path(args.embeddings_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all embedding parquets into a dict: source_city -> DataFrame
    source_data: dict[str, pd.DataFrame] = {}
    for parquet_path in sorted(embeddings_dir.glob("embeddings_*.parquet")):
        src = parquet_path.stem.replace("embeddings_", "").upper()
        if src in CITY_CODES:
            source_data[src] = pd.read_parquet(parquet_path)
            print(f"Loaded embeddings for source: {src} ({len(source_data[src])} rows)")

    if not source_data:
        raise FileNotFoundError(f"No embedding parquet files found in {embeddings_dir}")

    emb_cols_cache: dict[str, list[str]] = {}

    def get_emb(df: pd.DataFrame, df_key: str) -> np.ndarray:
        if df_key not in emb_cols_cache:
            cols = sorted(
                [c for c in df.columns if c.startswith("emb_")],
                key=lambda x: int(x.split("_")[1])
            )
            emb_cols_cache[df_key] = cols
        return df[emb_cols_cache[df_key]].values.astype(np.float32)

    # ---------------------------------------------------------------------------
    # Compute metrics for all source-target pairs
    # ---------------------------------------------------------------------------
    records = []

    for source_city, target_city in product(CITY_CODES, CITY_CODES):
        if source_city == target_city:
            continue
        if source_city not in source_data:
            print(f"  Skipping {source_city}→{target_city}: no source model data")
            continue

        df = source_data[source_city]

        src_mask = df["city"].values == source_city
        tgt_mask = df["city"].values == target_city

        if not src_mask.any():
            print(f"  Warning: no source images for {source_city} in its own parquet")
            continue
        if not tgt_mask.any():
            print(f"  Warning: no target images for {target_city} in {source_city} parquet")
            continue

        src_emb = get_emb(df[src_mask], source_city)
        tgt_emb = get_emb(df[tgt_mask], source_city)

        print(f"  {source_city} → {target_city}: "
              f"src={len(src_emb)}, tgt={len(tgt_emb)}", end="  ")

        # MMD
        mmd = compute_mmd(src_emb, tgt_emb)

        # Overlap ratios
        overlaps = {}
        for k in K_VALUES:
            overlaps[k] = compute_overlap_ratio(src_emb, tgt_emb, k=k)

        print(f"MMD={mmd:.4f}  "
              + "  ".join(f"overlap_k{k}={overlaps[k]:.3f}" for k in K_VALUES))

        record = {
            "source_city": source_city,
            "target_city": target_city,
            "mmd":         mmd,
        }
        for k in K_VALUES:
            record[f"overlap_k{k}"] = overlaps[k]
        records.append(record)

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(output_dir / "metrics_all_pairs.csv", index=False)
    print(f"\nSaved long-form metrics → metrics_all_pairs.csv")

    # ---------------------------------------------------------------------------
    # Pivot to 7×7 matrices and plot
    # ---------------------------------------------------------------------------
    metric_specs = [
        ("mmd",          "MMD (RBF kernel, median bandwidth)",        "viridis",   ".3f"),
        ("overlap_k10",  "Overlap ratio (k=10, 95th pct threshold)",  "viridis_r", ".2f"),
        ("overlap_k20",  "Overlap ratio (k=20, 95th pct threshold)",  "viridis_r", ".2f"),
    ]

    for col, title, cmap, fmt in metric_specs:
        matrix = df_metrics.pivot(
            index="source_city", columns="target_city", values=col
        ).reindex(index=CITY_CODES, columns=CITY_CODES)

        matrix.to_csv(output_dir / f"matrix_{col}.csv")

        vmin = 0.0 if col.startswith("overlap") else None
        vmax = 1.0 if col.startswith("overlap") else None

        plot_heatmap(
            matrix, title,
            output_dir / f"heatmap_{col}.png",
            cmap=cmap, fmt=fmt, vmin=vmin, vmax=vmax,
        )

    print("\nDone. All metrics and heatmaps saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier 1: MMD and overlap ratio")
    parser.add_argument("--embeddings_dir",
                        default=f"{TRANSFER_PATH}/results/tier1/embeddings",
                        help="Directory with embedding parquet files from Step 1")
    parser.add_argument("--output_dir",
                        default=f"{TRANSFER_PATH}/results/tier1/metrics",
                        help="Output directory for CSVs and heatmap figures")
    args = parser.parse_args()
    main(args)