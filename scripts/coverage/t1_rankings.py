"""
tier1_ranking_figures.py
-------------------------
Tier 1 transfer assessment — Source model coverage figures.

Produces three figures:
  1. fig1_ranking_overview.png  : per-target horizontal bar chart (transferability score)
  2. fig2_mmd_vs_overlap.png    : MMD vs overlap scatter per target city
  3. fig3_coverage_flags.png    : source model coverage matrix (fixed thresholds)

Coverage categories (MMD threshold: 0.05, Overlap threshold: 0.90):
  Strong match  low MMD  + high overlap -> proceed
  Caution       high MMD OR low overlap -> proceed with caution
  Poor match    high MMD + low overlap  -> deprioritise

Usage (defaults):
    python tier1_ranking_figures.py

Usage (override paths):
    python tier1_ranking_figures.py \
        --metrics_csv  results/tier1/metrics/metrics_all_pairs.csv \
        --output_dir   results/tier1/rankings/
"""

from __future__ import annotations

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Project-level defaults
# ---------------------------------------------------------------------------

TRANSFER_PATH = os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer")

CITY_CODES = ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI", "MUM"]

CITY_COLORS = {
    "ACC": "#E63946",
    "NBO": "#457B9D",
    "BEI": "#2A9D8F",
    "TEM": "#E9C46A",
    "KIS": "#F4A261",
    "CHI": "#A8DADC",
    "MUM": "#9B5DE5",
}

# (mmd_high, overlap_low) -> (label, color)
# Paul Tol colorblind-safe palette, maintaining teal/amber/red vibe
CATEGORY_MAP = {
    (False, False): ("Strong match", "#44AA99"),  # safe teal
    (True,  False): ("Caution",      "#DDAA33"),  # safe amber
    (False, True):  ("Caution",      "#DDAA33"),
    (True,  True):  ("Poor match",   "#BB5566"),  # safe rose-red
}

MMD_THRESHOLD     = 0.05
OVERLAP_THRESHOLD = 0.90

BG_COLOR    = "#000000"
PANEL_COLOR = "#111111"
GRID_COLOR  = "#222222"
TICK_COLOR  = "#cccccc"
FIG_DPI     = 150


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def style_ax(ax):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    ranked_rows = []
    for tc in CITY_CODES:
        s = df[df["target_city"] == tc].copy()
        if s.empty:
            continue
        s["mmd_rank"]        = s["mmd"].rank(method="min", ascending=True)
        s["overlap_rank"]    = s["overlap"].rank(method="min", ascending=False)
        s["composite_score"] = (s["mmd_rank"] + s["overlap_rank"]) / 2.0
        s["composite_rank"]  = s["composite_score"].rank(method="min", ascending=True)
        ranked_rows.append(s)
    return pd.concat(ranked_rows, ignore_index=True)


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag each source-target pair using fixed thresholds on MMD and overlap.
    High MMD    = MMD > MMD_THRESHOLD
    Low overlap = overlap < OVERLAP_THRESHOLD
    Three categories: Strong match / Caution / Poor match.
    """
    df = df.copy()
    df["mmd_high"]    = df["mmd"]     > MMD_THRESHOLD
    df["overlap_low"] = df["overlap"] < OVERLAP_THRESHOLD

    print(f"\n  Thresholds: MMD > {MMD_THRESHOLD}, Overlap < {OVERLAP_THRESHOLD}")
    print(f"  High MMD:    {df['mmd_high'].sum()} / {len(df)} pairs")
    print(f"  Low overlap: {df['overlap_low'].sum()} / {len(df)} pairs")

    def assign(row):
        key = (bool(row["mmd_high"]), bool(row["overlap_low"]))
        label, color = CATEGORY_MAP[key]
        return pd.Series({"category": label, "category_color": color})

    df[["category", "category_color"]] = df.apply(assign, axis=1)

    print("\n  Category counts:")
    for label, count in df["category"].value_counts().items():
        print(f"    {label}: {count}")

    return df


# ---------------------------------------------------------------------------
# Figure 1 — Per-target overview bar chart
# ---------------------------------------------------------------------------

def fig1_ranking_overview(df: pd.DataFrame, output_path: Path):
    n_sources = len(CITY_CODES) - 1

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=FIG_DPI)
    fig.patch.set_facecolor(BG_COLOR)
    axes_flat = axes.flatten()

    for idx, target_city in enumerate(CITY_CODES):
        ax = axes_flat[idx]
        style_ax(ax)

        subset = df[df["target_city"] == target_city].sort_values(
            "composite_rank", ascending=False
        )
        scores  = 1.0 - (subset["composite_rank"].values - 1) / (n_sources - 1)
        sources = subset["source_city"].values
        y_pos   = np.arange(len(sources))

        for i, src in enumerate(sources):
            ax.barh(y_pos[i], scores[i],
                    color=CITY_COLORS.get(src, "#cccccc"),
                    alpha=0.85, height=0.6, edgecolor="none")

        for i, row in enumerate(subset.itertuples()):
            ax.text(0.02, y_pos[i], f"#{int(row.composite_rank)}",
                    va="center", ha="left",
                    fontsize=7.5, fontweight="bold", color=BG_COLOR)
            ax.text(min(scores[i] + 0.02, 1.02), y_pos[i],
                    f"MMD={row.mmd:.3f}  ov={row.overlap:.2f}",
                    va="center", ha="left", fontsize=6.5, color=TICK_COLOR)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sources, fontsize=8.5)
        for tick, src in zip(ax.get_yticklabels(), sources):
            tick.set_color(CITY_COLORS.get(src, "white"))
        ax.set_xlim(0, 1.55)
        ax.set_xlabel("Transferability score", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_title(f"Target: {target_city}",
                     color=CITY_COLORS.get(target_city, "white"),
                     fontsize=10, fontweight="bold", pad=6)
        ax.axvline(0.5, color=GRID_COLOR, linewidth=0.8,
                   linestyle="--", zorder=0)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5,
                linestyle=":", alpha=0.6)

    axes_flat[-1].set_visible(False)
    fig.suptitle(
        "Source city transferability ranking per target city\n"
        "(bar length = transferability score; rank 1 = best source model coverage)",
        color="white", fontsize=11, y=1.01,
    )
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2 — MMD vs overlap scatter per target
# ---------------------------------------------------------------------------

def fig2_mmd_vs_overlap(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=FIG_DPI)
    fig.patch.set_facecolor(BG_COLOR)
    axes_flat = axes.flatten()

    for idx, target_city in enumerate(CITY_CODES):
        ax = axes_flat[idx]
        style_ax(ax)

        subset  = df[df["target_city"] == target_city].copy()
        x       = subset["mmd"].values
        y       = subset["overlap"].values
        sources = subset["source_city"].values

        med_x = np.median(x)
        med_y = np.median(y)

        ax.axvline(med_x, color=GRID_COLOR, linewidth=0.9,
                   linestyle="--", zorder=1, alpha=0.8)
        ax.axhline(med_y, color=GRID_COLOR, linewidth=0.9,
                   linestyle="--", zorder=1, alpha=0.8)

        quad_kw = dict(fontsize=6, color="#666688", alpha=0.9,
                       transform=ax.transAxes)
        ax.text(0.02, 0.97, "✓ low MMD\nhigh overlap",
                va="top", ha="left", **quad_kw)
        ax.text(0.98, 0.03, "✗ high MMD\nlow overlap",
                va="bottom", ha="right", **quad_kw)

        for i, src in enumerate(sources):
            color = CITY_COLORS.get(src, "#cccccc")
            rank  = int(subset[subset["source_city"] == src]["composite_rank"].values[0])
            ax.scatter(x[i], y[i], c=color, s=70, zorder=3,
                       alpha=0.9, linewidths=0.5, edgecolors="white")
            ax.text(x[i] + 0.003, y[i] + 0.003,
                    f"{src} (#{rank})", fontsize=7, color=color, zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground=PANEL_COLOR)])

        ax.set_xlabel("MMD  (lower = better)", fontsize=8)
        ax.set_ylabel("Overlap  (higher = better)", fontsize=8)
        ax.set_title(f"Target: {target_city}",
                     color=CITY_COLORS.get(target_city, "white"),
                     fontsize=10, fontweight="bold", pad=6)
        ax.grid(color=GRID_COLOR, linewidth=0.4, linestyle=":", alpha=0.5)

    axes_flat[-1].set_visible(False)
    fig.suptitle(
        "MMD vs overlap per target city\n"
        "(top-left = best source model coverage; labels show composite rank)",
        color="white", fontsize=11, y=1.01,
    )
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3 — Source model coverage matrix
# ---------------------------------------------------------------------------

def fig3_coverage_flags(df: pd.DataFrame, output_path: Path):
    """
    Matrix dot plot: source cities on y-axis, target cities on x-axis.
    Dots coloured by source model coverage category:
      Teal   (✓) = Strong match
      Yellow (~) = Caution
      Red    (✗) = Poor match
    Legend placed below the plot. Diagonal marked with ×.
    """
    n        = len(CITY_CODES)
    DOT_SIZE = 2200

    lookup = {
        (row.source_city, row.target_city): (row.category, row.category_color)
        for row in df.itertuples()
    }

    SYMBOLS = {"Strong match": "Strong\nmatch", "Caution": "Caution", "Poor match": "Poor\nmatch"}
    TXT_CLR = {"Strong match": "#000000", "Caution": "#000000", "Poor match": "#ffffff"}

    fig, ax = plt.subplots(figsize=(7, 7), dpi=FIG_DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for yi, src in enumerate(CITY_CODES):
        for xi, tgt in enumerate(CITY_CODES):
            if src == tgt:
                ax.scatter(xi, yi, s=DOT_SIZE * 0.25, marker="x",
                           color="#444466", linewidths=1.5, zorder=2)
                continue
            key = (src, tgt)
            if key not in lookup:
                continue
            category, color = lookup[key]
            ax.scatter(xi, yi, s=DOT_SIZE, color=color, zorder=3,
                       edgecolors=BG_COLOR, linewidths=1.2, alpha=0.92)
            ax.text(xi, yi, SYMBOLS[category],
                    ha="center", va="center",
                    fontsize=6, fontweight="bold",
                    color=TXT_CLR[category], zorder=4,
                    linespacing=1.3)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(CITY_CODES, fontsize=10)
    for tick, tgt in zip(ax.get_xticklabels(), CITY_CODES):
        tick.set_color(CITY_COLORS.get(tgt, TICK_COLOR))

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(CITY_CODES, fontsize=10)
    for tick, src in zip(ax.get_yticklabels(), CITY_CODES):
        tick.set_color(CITY_COLORS.get(src, TICK_COLOR))

    ax.set_xlabel("Target city", color=TICK_COLOR, fontsize=11, labelpad=8)
    ax.set_ylabel("Source city", color=TICK_COLOR, fontsize=11, labelpad=8)
    ax.set_title(
        "Source model coverage per transfer pair\n"
        f"MMD threshold: {MMD_THRESHOLD}  |  Overlap threshold: {OVERLAP_THRESHOLD}",
        color="white", fontsize=11, pad=10,
    )
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()

    for i in range(n):
        ax.axvline(i, color=GRID_COLOR, linewidth=0.5, alpha=0.3, zorder=1)
        ax.axhline(i, color=GRID_COLOR, linewidth=0.5, alpha=0.3, zorder=1)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TICK_COLOR)

    legend_handles = [
        mpatches.Patch(facecolor="#44AA99", edgecolor=BG_COLOR,
                       label=f"Strong match  "
                             f"(MMD ≤ {MMD_THRESHOLD}, overlap ≥ {OVERLAP_THRESHOLD})"),
        mpatches.Patch(facecolor="#DDAA33", edgecolor=BG_COLOR,
                       label="Caution  (high MMD or low overlap, not both)"),
        mpatches.Patch(facecolor="#BB5566", edgecolor=BG_COLOR,
                       label=f"Poor match  "
                             f"(MMD > {MMD_THRESHOLD} and overlap < {OVERLAP_THRESHOLD})"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=1,
        facecolor="#111111",
        edgecolor=GRID_COLOR,
        labelcolor="white",
        title="Source model coverage category",
        title_fontsize=9,
        handlelength=1.2,
        borderpad=0.8,
    )
    legend.get_title().set_color("white")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    df["source_city"] = df["source_city"].str.upper()
    df["target_city"] = df["target_city"].str.upper()
    if "overlap" not in df.columns:
        df["overlap"] = (df["overlap_k10"] + df["overlap_k20"]) / 2.0
    print(f"Loaded {len(df)} source-target pairs")

    df_ranked  = compute_rankings(df)

    print("\nComputing coverage flags...")
    df_flagged = compute_flags(df)
    df_flagged.to_csv(output_dir / "source_flags.csv", index=False)
    print(f"Saved flags → source_flags.csv")

    print("\nGenerating Figure 1: per-target overview bar chart...")
    fig1_ranking_overview(df_ranked, output_dir / "fig1_ranking_overview.png")

    print("Generating Figure 2: MMD vs overlap scatter...")
    fig2_mmd_vs_overlap(df_ranked, output_dir / "fig2_mmd_vs_overlap.png")

    print("Generating Figure 3: source model coverage matrix...")
    fig3_coverage_flags(df_flagged, output_dir / "fig3_coverage_flags.png")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tier 1: Source model coverage figures"
    )
    parser.add_argument(
        "--metrics_csv",
        default=f"{TRANSFER_PATH}/results/tier1/metrics/metrics_all_pairs.csv",
        help="Path to metrics_all_pairs.csv from tier1_metrics.py",
    )
    parser.add_argument(
        "--output_dir",
        default=f"{TRANSFER_PATH}/results/tier1/rankings",
        help="Output directory for figures and flags CSV",
    )
    args = parser.parse_args()
    main(args)