"""
tier1_validation.py
--------------------
Tier 1 transfer assessment — Step 4: Validation against Tier 2 outcomes.

Correlates Tier 1 metrics (MMD, overlap, coverage category) against:
  - Zero-shot Spearman ρ (primary outcome)
  - Fine-tuning improvement = max(ρ over sample sizes) − zero-shot ρ
    computed separately for informed_t1 and random strategies.

Produces:
  - validation_merged.csv              : merged Tier 1 + Tier 2 data
  - validation_summary.csv             : Spearman correlation table
  - scatter_<metric>_vs_<outcome>.png  : 6 scatter plots (3 metrics × 2 outcomes)
                                         × 2 strategies = 12 scatter plots
  - violin_<outcome>.png               : 3 violin plots (one per outcome)
                                         grouped by coverage category

Usage (defaults):
    python tier1_validation.py

Usage (override paths):
    python tier1_validation.py \
        --metrics_csv   results/tier1/metrics/metrics_all_pairs.csv \
        --flags_csv     results/tier1/rankings/source_flags.csv     \
        --tier2_dir     results/transferLearning/                   \
        --output_dir    results/tier1/validation/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Project-level defaults
# ---------------------------------------------------------------------------

TRANSFER_PATH = "/home/jovyan/private/where_transfer"

CITY_CODES = ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI", "MUM"]
OOD_CITIES = {"MUM"}

CITY_COLORS = {
    "ACC": "#E63946", "NBO": "#457B9D", "BEI": "#2A9D8F",
    "TEM": "#E9C46A", "KIS": "#F4A261", "CHI": "#A8DADC", "MUM": "#9B5DE5",
}

# Coverage category ordinal encoding and colors (colorblind-safe)
CATEGORY_ORDER  = ["Strong match", "Caution", "Poor match"]
CATEGORY_CODES  = {"Strong match": 0, "Caution": 1, "Poor match": 2}
CATEGORY_COLORS = {"Strong match": "#44AA99", "Caution": "#DDAA33", "Poor match": "#BB5566"}

STRATEGIES = ["informed_t1", "random"]

TIER1_METRICS = {
    "mmd":      "MMD (RBF kernel)",
    "overlap":  "Overlap ratio (avg k10/k20)",
    "cat_code": "Coverage category (ordinal)",
}

BG_COLOR   = "#000000"
GRID_COLOR = "#222222"
TICK_COLOR = "#cccccc"
FIG_DPI    = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tier2(tier2_dir: str) -> pd.DataFrame:
    """
    Load all individual experiment CSVs from the two strategy subfolders:
      - informed_t1__layerwise_lr1e-04
      - random__layerwise_lr1e-04

    Returns a long-form DataFrame with columns:
    source, target, strategy, sample_size, seed, spearman.
    Filters to single-city source-target pairs only.
    """
    tier2_path = Path(tier2_dir)

    STRATEGY_FOLDERS = [
        "informed_t1__layerwise_lr1e-04",
        "random__layerwise_lr1e-04",
    ]

    csv_files = []
    for folder in STRATEGY_FOLDERS:
        folder_path = tier2_path / folder
        if not folder_path.exists():
            print(f"  Warning: folder not found: {folder_path}")
            continue
        found = list(folder_path.glob("*.csv"))
        print(f"  {folder}: {len(found)} CSVs")
        csv_files.extend(found)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in expected strategy subfolders under {tier2_dir}"
        )

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {csv_file.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["source"] = combined["source"].str.upper()
    combined["target"] = combined["target"].str.upper()

    # Single-city pairs only
    combined = combined[
        combined["source"].isin(CITY_CODES) &
        combined["target"].isin(CITY_CODES) &
        (combined["source"] != combined["target"])
    ]
    print(f"  Loaded {len(combined)} rows from {len(csv_files)} CSVs")
    return combined


def compute_zeroshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract zero-shot rows (strategy == 'zero_shot', sample_size == 0)
    and average over seeds per source-target pair.
    Zero-shot is strategy-agnostic so we don't split by sampling here.
    Returns: source_city, target_city, zeroshot_rho
    """
    zs = df[
        (df["strategy"] == "zero_shot") & (df["sample_size"] == 0)
    ].copy()

    zs_mean = (
        zs.groupby(["source", "target"])["spearman"]
        .mean()
        .reset_index()
        .rename(columns={
            "source":   "source_city",
            "target":   "target_city",
            "spearman": "zeroshot_rho",
        })
    )
    return zs_mean


def compute_ft_improvement(df: pd.DataFrame, zs_mean: pd.DataFrame) -> pd.DataFrame:
    """
    Fine-tuning improvement per source-target-sampling:
      1. Filter to fine-tuning rows (strategy == 'layerwise', sample_size > 0)
      2. Average spearman over seeds at each sample_size per sampling strategy
      3. Take max over sample_sizes → best_ft_rho
      4. improvement = best_ft_rho - zeroshot_rho

    Args:
        df      : full long-form Tier 2 data
        zs_mean : zero-shot means from compute_zeroshot()

    Returns: source_city, target_city, sampling, improvement
    """
    ft = df[
        (df["strategy"] == "layerwise") & (df["sample_size"] > 0)
    ].copy()

    # Step 1: average over seeds at each sample_size × sampling
    ft_mean = (
        ft.groupby(["source", "target", "sampling", "sample_size"])["spearman"]
        .mean()
        .reset_index()
    )

    # Step 2: max over sample sizes per source-target-sampling
    ft_best = (
        ft_mean.groupby(["source", "target", "sampling"])["spearman"]
        .max()
        .reset_index()
        .rename(columns={"spearman": "best_ft_rho"})
    )

    # Step 3: merge with zero-shot and compute improvement
    merged = ft_best.merge(
        zs_mean.rename(columns={"source_city": "source", "target_city": "target"}),
        on=["source", "target"], how="inner"
    )
    merged["improvement"] = merged["best_ft_rho"] - merged["zeroshot_rho"]

    return merged[["source", "target", "sampling", "improvement"]].rename(
        columns={"source": "source_city", "target": "target_city"}
    )


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame,
    tier1_col: str,
    tier1_label: str,
    outcome_col: str,
    outcome_label: str,
    strategy: str,
    output_path: Path,
):
    """
    Scatter: Tier 1 metric (x) vs Tier 2 outcome (y).
    Points coloured by target city. MUM marked with diamond.
    Linear trend line + Spearman ρ annotation.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=FIG_DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = df[tier1_col].values
    y = df[outcome_col].values

    for _, row in df.iterrows():
        is_ood  = row["target_city"] in OOD_CITIES
        color   = CITY_COLORS.get(row["target_city"], "#cccccc")
        marker  = "D" if is_ood else "o"
        size    = 60  if is_ood else 40
        ax.scatter(
            row[tier1_col], row[outcome_col],
            c=color, s=size, marker=marker,
            alpha=0.85, linewidths=0.5,
            edgecolors="white" if is_ood else "none",
            zorder=3,
        )

    # Linear trend
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 2:
        z      = np.polyfit(x[valid], y[valid], 1)
        p      = np.poly1d(z)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, p(x_line), color="#ffffff", linewidth=1.2,
                alpha=0.4, linestyle="--", zorder=2)

        rho, pval = spearmanr(x[valid], y[valid])
        p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
        ax.text(
            0.05, 0.93,
            f"Spearman ρ = {rho:.3f}  ({p_str})",
            transform=ax.transAxes,
            color="white", fontsize=8.5,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)],
        )

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle=":", zorder=1)

    # Legend
    handles = []
    for city in CITY_CODES:
        if df[df["target_city"] == city].empty:
            continue
        is_ood = city in OOD_CITIES
        h = plt.scatter([], [],
                        c=CITY_COLORS.get(city, "#cccccc"),
                        s=50 if is_ood else 35,
                        marker="D" if is_ood else "o",
                        label=f"{city}{'*' if is_ood else ''}",
                        edgecolors="white" if is_ood else "none",
                        linewidths=0.5)
        handles.append(h)
    legend = ax.legend(
        handles=handles,
        title="Target city", title_fontsize=8,
        fontsize=7.5, facecolor="#111111",
        edgecolor=GRID_COLOR, labelcolor="white",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    legend.get_title().set_color("white")

    strategy_label = {"informed_t1": "Informed (t1)", "random": "Random",
                      "zero_shot": "Zero-shot"}.get(strategy, strategy)
    ax.set_xlabel(tier1_label, color=TICK_COLOR, fontsize=10)
    ax.set_ylabel(outcome_label, color=TICK_COLOR, fontsize=10)
    ax.set_title(
        f"{tier1_label}\nvs {outcome_label} [{strategy_label}]",
        color="white", fontsize=10, pad=8,
    )
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.4, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Violin plots
# ---------------------------------------------------------------------------

def plot_violin(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_label: str,
    strategy: str,
    output_path: Path,
):
    """
    Violin plot: one panel per outcome, grouped by coverage category.
    Individual data points overlaid. Sample size annotated below each violin.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    positions = np.arange(len(CATEGORY_ORDER))
    counts    = {}

    for pos, category in enumerate(CATEGORY_ORDER):
        subset = df[df["category"] == category][outcome_col].dropna().values
        color  = CATEGORY_COLORS[category]
        counts[category] = len(subset)

        if len(subset) < 2:
            # Not enough data for violin — just plot points
            ax.scatter(
                np.full(len(subset), pos), subset,
                color=color, s=50, zorder=4, alpha=0.9,
                edgecolors=BG_COLOR, linewidths=0.8,
            )
        else:
            parts = ax.violinplot(
                subset,
                positions=[pos],
                widths=0.6,
                showmedians=True,
                showextrema=True,
            )
            # Style violin body
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.4)
            for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
                if part_name in parts:
                    parts[part_name].set_edgecolor(color)
                    parts[part_name].set_linewidth(1.5)

            # Overlay individual points with jitter
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(subset))
            ax.scatter(
                pos + jitter, subset,
                color=color, s=35, zorder=4, alpha=0.85,
                edgecolors=BG_COLOR, linewidths=0.6,
            )

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{cat}\n(n={counts.get(cat, 0)})" for cat in CATEGORY_ORDER],
        fontsize=10,
    )
    for tick, category in zip(ax.get_xticklabels(), CATEGORY_ORDER):
        tick.set_color(CATEGORY_COLORS[category])

    strategy_label = {"informed_t1": "Informed (t1)", "random": "Random",
                      "zero_shot": "Zero-shot"}.get(strategy, strategy)
    ax.set_ylabel(outcome_label, color=TICK_COLOR, fontsize=10)
    ax.set_xlabel("Source model coverage category", color=TICK_COLOR,
                  fontsize=10, labelpad=8)
    ax.set_title(
        f"{outcome_label} by source model coverage category\n"
        f"[{strategy_label}]",
        color="white", fontsize=11, pad=10,
    )
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4,
            linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Correlation summary table
# ---------------------------------------------------------------------------

def compute_correlation_table(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    rows = []
    subsets = {
        "All pairs":       df,
        "Excl. OOD (MUM)": df[~df["target_city"].isin(OOD_CITIES)],
    }
    for subset_name, df_sub in subsets.items():
        y = df_sub[outcome_col].values
        for col, label in TIER1_METRICS.items():
            if col not in df_sub.columns:
                continue
            x     = df_sub[col].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 3:
                continue
            rho, pval = spearmanr(x[valid], y[valid])
            rows.append({
                "Outcome":       outcome_col,
                "Subset":        subset_name,
                "Tier 1 metric": label,
                "Spearman rho":  round(rho, 4),
                "p-value":       round(pval, 4),
                "n pairs":       int(valid.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Tier 1 ---
    df_t1 = pd.read_csv(args.metrics_csv)
    df_t1["source_city"] = df_t1["source_city"].str.upper()
    df_t1["target_city"] = df_t1["target_city"].str.upper()
    if "overlap" not in df_t1.columns:
        df_t1["overlap"] = (df_t1["overlap_k10"] + df_t1["overlap_k20"]) / 2.0
    print(f"Loaded Tier 1 metrics: {len(df_t1)} pairs")

    # --- Load coverage flags ---
    df_flags = pd.read_csv(args.flags_csv)
    df_flags["source_city"] = df_flags["source_city"].str.upper()
    df_flags["target_city"] = df_flags["target_city"].str.upper()
    df_flags["cat_code"]    = df_flags["category"].map(CATEGORY_CODES)
    df_t1 = df_t1.merge(
        df_flags[["source_city", "target_city", "category", "cat_code"]],
        on=["source_city", "target_city"], how="left",
    )

    # --- Load Tier 2 ---
    print("\nLoading Tier 2 results...")
    df_t2_raw = load_tier2(args.tier2_dir)

    # --- Zero-shot (strategy-agnostic) ---
    df_zs = compute_zeroshot(df_t2_raw)
    df_zs_merged = df_t1.merge(df_zs, on=["source_city", "target_city"], how="inner")
    print(f"\nZero-shot merged: {len(df_zs_merged)} pairs")
    df_zs_merged.to_csv(output_dir / "validation_merged_zeroshot.csv", index=False)

    # --- Fine-tuning improvement (per sampling strategy) ---
    df_ft = compute_ft_improvement(df_t2_raw, df_zs)

    # --- Zero-shot: scatter plots + single violin ---
    print("\nGenerating zero-shot figures...")
    for t1_col, t1_label in TIER1_METRICS.items():
        if t1_col not in df_zs_merged.columns:
            continue
        plot_scatter(
            df_zs_merged, t1_col, t1_label,
            "zeroshot_rho", "Zero-shot Spearman ρ",
            "zero_shot",
            output_dir / f"scatter_{t1_col}_vs_zeroshot.png",
        )

    # One violin for zero-shot (not split by strategy)
    plot_violin(
        df_zs_merged, "zeroshot_rho", "Zero-shot Spearman ρ",
        "zero_shot",
        output_dir / "violin_zeroshot.png",
    )

    # --- Fine-tuning improvement: scatter plots + violin per sampling ---
    all_corr_rows = []

    for sampling in STRATEGIES:
        print(f"\nGenerating fine-tuning improvement figures [{sampling}]...")
        ft_strat = df_ft[df_ft["sampling"] == sampling][
            ["source_city", "target_city", "improvement"]
        ]
        df_ft_merged = df_t1.merge(ft_strat, on=["source_city", "target_city"], how="inner")
        df_ft_merged = df_ft_merged.merge(
            df_zs, on=["source_city", "target_city"], how="left"
        )
        print(f"  Merged: {len(df_ft_merged)} pairs")
        df_ft_merged.to_csv(
            output_dir / f"validation_merged_ft_{sampling}.csv", index=False
        )

        for t1_col, t1_label in TIER1_METRICS.items():
            if t1_col not in df_ft_merged.columns:
                continue
            plot_scatter(
                df_ft_merged, t1_col, t1_label,
                "improvement", "Fine-tuning improvement (Δρ)",
                sampling,
                output_dir / f"scatter_{t1_col}_vs_improvement_{sampling}.png",
            )

        plot_violin(
            df_ft_merged, "improvement", "Fine-tuning improvement (Δρ)",
            sampling,
            output_dir / f"violin_improvement_{sampling}.png",
        )

        # Correlation rows for both outcomes
        for outcome_col, outcome_label in [
            ("zeroshot_rho", "Zero-shot Spearman ρ"),
            ("improvement",  "Fine-tuning improvement (Δρ)"),
        ]:
            if outcome_col not in df_ft_merged.columns:
                continue
            rows = compute_correlation_table(df_ft_merged, outcome_col)
            rows["sampling"] = sampling
            all_corr_rows.append(rows)

    # Also add zero-shot correlations (not sampling-specific)
    for outcome_col, outcome_label in [("zeroshot_rho", "Zero-shot Spearman ρ")]:
        rows = compute_correlation_table(df_zs_merged, outcome_col)
        rows["sampling"] = "zero_shot"
        all_corr_rows.append(rows)

    # --- Save correlation summary ---
    df_corr = pd.concat(all_corr_rows, ignore_index=True)
    df_corr = df_corr[["sampling", "Outcome", "Subset",
                        "Tier 1 metric", "Spearman rho", "p-value", "n pairs"]]
    df_corr.to_csv(output_dir / "validation_summary.csv", index=False)
    print("\nCorrelation summary:")
    print(df_corr.to_string(index=False))
    print(f"\nSaved → validation_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tier 1: Validation against Tier 2 outcomes"
    )
    parser.add_argument(
        "--metrics_csv",
        default=f"{TRANSFER_PATH}/results/tier1/metrics/metrics_all_pairs.csv",
        help="Path to metrics_all_pairs.csv from tier1_metrics.py",
    )
    parser.add_argument(
        "--flags_csv",
        default=f"{TRANSFER_PATH}/results/tier1/rankings/source_flags.csv",
        help="Path to source_flags.csv from tier1_ranking_figures.py",
    )
    parser.add_argument(
        "--tier2_dir",
        default=f"{TRANSFER_PATH}/results/transferLearning",
        help="Directory containing Tier 2 experiment result CSVs",
    )
    parser.add_argument(
        "--output_dir",
        default=f"{TRANSFER_PATH}/results/tier1/validation",
        help="Output directory for validation figures and tables",
    )
    args = parser.parse_args()
    main(args)