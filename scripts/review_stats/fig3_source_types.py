"""
fig3_source_types.py  (review round 1; rebuilds Fig 3)
======================================================

Single-city sources vs multi-city aggregates, compared within each target city
(R1-M3 / R1-M6):

  a  zero-shot Spearman      b  finetuned Spearman (n = 5,000)      c  improvement

One row per target city; each marker is the mean over that target's sources of
one type (single city, primary/secondary tier aggregate, continental
aggregate), with the target always left out of the aggregate. Random-sampling
runs only (free of the informed-sampling leakage, R1-M1). The text in each
panel is the within-target comparison from aggregation_inference.py: mean
per-target difference single - aggregate, how many targets favour single
sources, and the exact Wilcoxon signed-rank p over the 7 targets.

Not yet shown: the seed-selected single source (decision 2), added once the
source-selection analysis is done.

  python fig3_source_types.py --results-zip ../../results/random__layerwise_lr1e-04.zip \
      --out ../../figures/fig3_source_types.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import fig_style as fs
from aggregation_inference import load, per_target

TYPES = [("single", "Single city", fs.BLUE, "o"),
         ("tier", "Tier aggregate", fs.ORANGE, "s"),
         ("continental", "Continental aggregate", fs.AQUA, "D")]
NAMES = {"ACC": "Accra", "NBO": "Nairobi", "BEI": "Beira", "TEM": "Tema",
         "KIS": "Kisumu", "CHI": "Chimoio", "MUM": "Mumbai"}
PANELS = [("zero_shot", "Zero-shot Spearman"), ("final", "Finetuned Spearman (n = 5,000)"),
          ("improvement", "Improvement (Δ Spearman)")]


def main(args):
    df = load(args.results_zip)
    means = df.groupby(["target", "type"])[[p for p, _ in PANELS]].mean()
    counts = df.groupby(["target", "type"]).size().unstack(fill_value=0)
    # order targets by zero-shot of single-city sources
    order = means.xs("single", level="type")["zero_shot"].sort_values().index.tolist()

    fs.setup()
    fig, axes = fs.plt.subplots(1, 3, figsize=(12, 4.2), dpi=300, sharey=True,
                                gridspec_kw={"wspace": 0.16})
    y = np.arange(len(order))
    for ax, (col, label), letter in zip(axes, PANELS, "abc"):
        fs.style(ax, grid_axis="x")
        for t in order:                       # connector showing the spread within a target
            v = [means.loc[(t, k), col] for k, *_ in TYPES if (t, k) in means.index]
            ax.plot([min(v), max(v)], [order.index(t)] * 2, color=fs.GRID, lw=2, zorder=1)
        for k, name, color, marker in TYPES:
            v = [means.loc[(t, k), col] if (t, k) in means.index else np.nan for t in order]
            ax.scatter(v, y, s=46, color=color, marker=marker, edgecolor=fs.SURFACE,
                       linewidth=1.2, zorder=3, label=name)
        if col == "improvement":
            ax.axvline(0, color=fs.MUTED, lw=0.8)
        d = per_target(df, "single", "aggregate", col)
        p = wilcoxon(d.values, alternative="two-sided", method="exact").pvalue if len(d) else np.nan
        ax.text(0.5, 1.02, f"single − aggregate {d.mean():+.2f}\n"
                           f"{int((d > 0).sum())}/{len(d)} targets; Wilcoxon {fs.p_text(p)}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=fs.INK2)
        ax.set_xlabel(label, fontsize=10)
        fs.panel(ax, letter, x=-0.08 if letter != "a" else -0.30)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAMES[t] for t in order], fontsize=9)
    axes[0].set_ylabel("Target city", fontsize=10)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False,
                   fontsize=9, handletextpad=0.3, columnspacing=1.2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}")
    print("sources per target and type:\n" + counts.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out", default="../../figures/fig3_source_types.png")
    main(ap.parse_args())
