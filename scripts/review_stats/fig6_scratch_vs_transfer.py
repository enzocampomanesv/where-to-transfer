"""
fig5_scratch_vs_transfer.py  (review round 1; new Fig 5, R3-4)
==============================================================

Transfer vs training from scratch on the same target comparisons, with the same
number of epochs and stopping rule for both (layer-wise finetuning, up to 1,000
epochs, early stopping on validation accuracy). One panel per target city plus a
summary panel of transfer minus training from scratch.

  lines: training from scratch; transfer from the city chosen on the seed sample,
         from the average city, and from the best city in hindsight (seed means)
  dashed: training from scratch on the full training pool (reference)

  python fig5_scratch_vs_transfer.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fig_style as fs
from fig3_source_types import NAMES

ORDER = ["CHI", "TEM", "ACC", "BEI", "NBO", "KIS", "MUM"]
ARMS = [("scratch", "Training from scratch", fs.NEUTRAL, "o", 2.4),
        ("average", "Transfer, average city", fs.ORANGE, "s", 1.6),
        ("best", "Transfer, best city (hindsight)", fs.AQUA, "D", 1.6),
        ("selected", "Transfer, city chosen on seed", fs.BLUE, "o", 2.6)]


def main(args):
    pairs = pd.read_csv(args.pairs_csv)
    full = pd.read_csv(args.scratch_csv)
    full = full[full["sample_size"].astype(str) == "full"].groupby("target")["spearman"].mean()
    m = pairs.groupby(["target", "sample_size"])[[a for a, *_ in ARMS]].mean()
    sizes = sorted(pairs["sample_size"].unique())
    x = np.log10(sizes)

    fs.setup()
    fig, axes = fs.plt.subplots(2, 4, figsize=(15, 8.4), dpi=300, gridspec_kw={"hspace": 0.5, "wspace": 0.32})
    for ax, t in zip(axes.flat, ORDER):
        fs.style(ax)
        ax.tick_params(labelsize=12)
        ax.axhline(full[t], color=fs.INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
        for a, name, color, marker, lw in ARMS:
            v = m.loc[t, a].reindex(sizes)
            ax.plot(x, v.values, color=color, lw=lw, marker=marker, ms=7, markeredgecolor=fs.SURFACE,
                    markeredgewidth=0.8, zorder=3 if a == "selected" else 2, label=name)
        ax.set_title(NAMES[t], loc="left", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=12)
        ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test Spearman", fontsize=14)
    for ax in axes[1, :]:
        ax.set_xlabel("Target comparisons", fontsize=13)

    ax = axes[1, 3]
    fs.style(ax)
    ax.tick_params(labelsize=12)
    ax.axhline(0, color=fs.MUTED, lw=0.9)
    for a, name, color, marker, lw in ARMS[1:]:
        d = (m[a] - m["scratch"]).unstack()[sizes]
        for k, n in enumerate(sizes):
            ax.scatter(np.full(len(d), x[k]) + {"average": -0.05, "best": 0.05, "selected": 0}[a], d[n],
                       s=22, color=color, alpha=0.4, edgecolor="none", zorder=2)
        ax.plot(x, d.mean().values, color=color, lw=lw, marker=marker, ms=7, markeredgecolor=fs.SURFACE, zorder=3)
    ax.set_title("Transfer − from scratch", loc="left", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=12)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_ylabel("Δ Spearman", fontsize=14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(fs.plt.Line2D([], [], color=fs.INK2, lw=1.1, ls=(0, (4, 3))))
    labels.append("Training from scratch, full training pool")
    fig.legend(handles[::-1][1:] + [handles[-1]], labels[::-1][1:] + [labels[-1]], loc="upper center",
               ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.04))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    d = pd.DataFrame({a: (m[a] - m["scratch"]).groupby(level="sample_size").mean() for a in ["selected", "average", "best"]})
    print(f"saved {args.out}\ntransfer - scratch (mean over targets):\n{d.round(4)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", default="../../results/review_stats/scratch_vs_transfer_long_layerwise_pairs.csv")
    ap.add_argument("--scratch-csv", default="../../results/from_scratch.csv")
    ap.add_argument("--out", default="../../figures/fig7_scratch_vs_transfer.png")
    main(ap.parse_args())
