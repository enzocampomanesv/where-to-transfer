"""
fig1f_error_rates.py  (review round 1; panel E of Fig 2, the SAT figure)
========================================================================

Zero-shot error rate on covered comparisons (Consensus + Preference Disagreement)
against the error rate on comparisons with an unfamiliar image (Structural or
Relational Ignorance), one point per source-target pair, at the study's
Isolation Forest settings. Points on the 1:1 line = unfamiliarity adds no error.
Pairs with at most 2% unfamiliar comparisons are left out (too few to estimate a
rate); point area scales with the square root of the number of unfamiliar test comparisons,
shown in a size legend.

  python fig1f_error_rates.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fig_style as fs


def pair_rates(shares_csv, dataset_csv, min_unfamiliar=0.02):
    d = pd.read_csv(shares_csv)
    d = d[d["varied"] == "baseline"].copy()
    d["familiar"] = d["consensus"] + d["cultural_disagreement"]
    d["unfamiliar"] = 1 - d["familiar"]
    d["err_familiar"] = d["cultural_disagreement"] / d["familiar"]
    wrong_unfam = d["cultural_disagreement"] * (1 / d["pd_share_of_errors"] - 1)
    d["err_unfamiliar"] = wrong_unfam / d["unfamiliar"]
    n_test = pd.read_csv(dataset_csv).set_index("city")["test_comparisons"]
    d["n_unfamiliar"] = d["unfamiliar"] * d["target"].map(n_test)
    return d[d["unfamiliar"] > min_unfamiliar]


SIZE_K = 8          # marker area = SIZE_K * sqrt(number of unfamiliar comparisons)
LEGEND_N = [100, 500, 2500]


def main(args):
    d = pair_rates(args.shares_csv, args.dataset_csv)
    fs.setup()
    fig, ax = fs.plt.subplots(figsize=(4.6, 4.6), dpi=300)
    fs.style(ax)
    ax.tick_params(labelsize=12)
    lo, hi = 0.20, 0.65
    ax.plot([lo, hi], [lo, hi], color=fs.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(hi - 0.012, hi - 0.04, "equal error rates", ha="right", va="top", fontsize=11, color=fs.INK2,
            rotation=45, rotation_mode="anchor")
    ax.scatter(d["err_familiar"], d["err_unfamiliar"], s=SIZE_K * np.sqrt(d["n_unfamiliar"]), color=fs.BLUE,
               alpha=0.75, edgecolor=fs.SURFACE, linewidth=0.8, zorder=2)
    handles = [ax.scatter([], [], s=SIZE_K * np.sqrt(n), color=fs.BLUE, alpha=0.75, edgecolor=fs.SURFACE)
               for n in LEGEND_N]
    ax.legend(handles, [f"{n:,}" for n in LEGEND_N], title="Unfamiliar\ncomparisons", loc="upper left",
              frameon=False, fontsize=11, title_fontsize=11, labelspacing=1.1, borderpad=0.6, handletextpad=0.8,
              scatterpoints=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = np.arange(0.2, 0.65, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Error rate\n(covered comparisons)", fontsize=13)
    ax.set_ylabel("Error rate\n(unfamiliar comparisons)", fontsize=13)
    if args.letter:
        fs.panel(ax, args.letter, x=-0.25)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"{len(d)} source-target pairs; median familiar {d.err_familiar.median():.3f}, "
          f"unfamiliar {d.err_unfamiliar.median():.3f}; within 0.10 of the 1:1 line: "
          f"{(abs(d.err_unfamiliar - d.err_familiar) <= 0.10).sum()}\n-> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares-csv", default="../../results/review_stats/iso_sensitivity_shares.csv")
    ap.add_argument("--dataset-csv", default="../../results/review_stats/dataset_summary.csv")
    ap.add_argument("--out", default="../../figures/fig3_error_rates_panel.png")
    ap.add_argument("--letter", default="", help="panel letter; empty when the letter is added in the composite")
    main(ap.parse_args())
