"""
fig2_coverage_vs_disagreement.py  (review round 1; replaces Fig 2)
=================================================================

Fig 2 no longer claims coverage predicts transfer. It shows what coverage does
measure and that it is a separate axis from preference disagreement:

  a  overlap ratio vs Structural Ignorance share (coverage tracks unfamiliarity)
  b  overlap ratio vs preference-disagreement rate among covered comparisons,
     points coloured by zero-shot Spearman (performance varies with disagreement,
     not with coverage)

One point per directed source -> target pair (n = 42). rho and exact QAP p-values
in each panel come from dyadic_inference.qap_spearman.

Usage:
  python fig2_coverage_vs_disagreement.py \
      --validation-zip ../../results/tier1/validation.zip \
      --results-zip ../../results/random__layerwise_lr1e-04.zip \
      --out ../../figures/fig2_coverage_vs_disagreement.png
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import dyadic_inference as di
from alr_inference import load_zero_shot

# chart chrome and sequential blue (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_SEQ = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POINT = "#2a78d6"


def load(validation_zip, results_zip) -> pd.DataFrame:
    zf = zipfile.ZipFile(validation_zip)
    df = pd.read_csv(io.BytesIO(zf.read("validation/validation_merged_zeroshot.csv"))).rename(
        columns={"source_city": "source", "target_city": "target"})
    df = df.merge(load_zero_shot(results_zip), on=["source", "target"])
    df["covered"] = df["consensus"] + df["cultural_disagreement"]
    df["pd_rate"] = df["cultural_disagreement"] / df["covered"]
    return df


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def stat_text(ax, q):
    p = "p < 0.001" if q["p_qap"] < 0.001 else f"p = {q['p_qap']:.2f}"
    ax.text(0.03, 0.04, f"ρ = {q['rho']:.2f}, {p}", transform=ax.transAxes,
            fontsize=9, color=INK2, ha="left", va="bottom")


def main(args):
    df = load(args.validation_zip, args.results_zip)
    q_si = di.qap_spearman(df, "overlap", "structural_ignorance")
    q_pd = di.qap_spearman(df, "overlap", "pd_rate")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": SURFACE})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=300,
                                   gridspec_kw={"wspace": 0.28})

    # a: coverage tracks unfamiliar environments
    style(axA)
    axA.scatter(df["overlap"], 100 * df["structural_ignorance"], s=42, color=POINT,
                edgecolor=SURFACE, linewidth=1.2, zorder=3)
    axA.set_xlabel("Coverage (overlap ratio)", fontsize=10)
    axA.set_ylabel("Structural Ignorance\n(% of test comparisons)", fontsize=10)
    axA.set_title("a", loc="left", fontsize=13, fontweight="bold", x=-0.14)
    stat_text(axA, q_si)

    # b: coverage vs preference disagreement, coloured by zero-shot performance
    style(axB)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    sc = axB.scatter(df["overlap"], 100 * df["pd_rate"], c=df["spearman"], cmap=cmap,
                     vmin=min(0, df["spearman"].min()), vmax=df["spearman"].max(),
                     s=42, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    axB.set_xlabel("Coverage (overlap ratio)", fontsize=10)
    axB.set_ylabel("Preference Disagreement\n(% of covered comparisons)", fontsize=10)
    axB.set_title("b", loc="left", fontsize=13, fontweight="bold", x=-0.14)
    stat_text(axB, q_pd)
    cb = fig.colorbar(sc, ax=axB, fraction=0.05, pad=0.03)
    cb.set_label("Zero-shot Spearman", fontsize=9, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=2)

    # label the least-covered pairs (Mumbai as target or Chimoio as source); in b the
    # cluster near full coverage leaves room only for the isolated lowest pair
    for i, (_, r) in enumerate(df.nsmallest(3, "overlap").iterrows()):
        targets = [(axA, 100 * r["structural_ignorance"])]
        if i == 0:
            targets.append((axB, 100 * r["pd_rate"]))
        for ax, y in targets:
            ax.annotate(f"{r['source']}→{r['target']}", (r["overlap"], y),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}\n  a: rho={q_si['rho']:.3f} p={q_si['p_qap']:.4f}"
          f"\n  b: rho={q_pd['rho']:.3f} p={q_pd['p_qap']:.4f}"
          f"\n  zero-shot vs PD rate: rho={di.qap_spearman(df, 'pd_rate', 'spearman')['rho']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-zip", default="../../results/tier1/validation.zip")
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out", default="../../figures/fig2_coverage_vs_disagreement.png")
    main(ap.parse_args())
