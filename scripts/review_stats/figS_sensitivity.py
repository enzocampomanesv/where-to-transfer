"""
figS_sensitivity.py  (review round 1, R1-M7; new supplementary figure)
======================================================================

Sensitivity of the two SAT-based results to the analysis settings, one setting
varied at a time around the study's baseline:

  a  Tier 1: coefficient of Preference Disagreement (PD) in the zero-shot ALR
     regression, with 95% city-bootstrap interval, under alternative
     isolation-forest cutoffs (contamination 0.02-0.20 on the image forest,
     the pair forest or both) and with the pair difference reversed
     (iso_sensitivity.py). The exact MRQAP p is 0.0002 in every row.
  b  Tier 2: PD share of the comparisons selected at n = 1,000 by the oracle and
     the prospective SAT sampler under alternative cutoffs, weights, temperature
     and kNN k (sampling_sensitivity.py, mean over 42 pairs x 5 seeds). The
     dashed line is the pool share, which random sampling matches.

  python figS_sensitivity.py --stats-dir ../../results/review_stats \
      --out ../../figures/figS_sensitivity.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fig_style as fs

N_FOCUS = 1000
GROUP_NAMES = {"image": "Image forest", "pair": "Pair forest", "both": "Both forests"}
WEIGHT_NAMES = {"SF_Misaligned": "SF", "Cultural Disagreement": "PD", "Relational Ignorance": "RI",
                "Consensus": "Consensus", "Structural Ignorance": "SI"}


def tier1_label(r):
    if r["varied"] == "baseline":
        return "Baseline ('auto')"
    if r["varied"] == "orientation":
        return "Pair difference reversed"
    return f"{GROUP_NAMES[r['varied']]} {float(r['setting']):.2f}"


def tier2_label(s):
    if s == "baseline":
        return "Baseline"
    k, v = s.split("=", 1)
    if k == "image":
        return f"Image forest {float(v):.2f}"
    if k == "pair":
        return f"Pair forest {float(v):.2f}"
    if k == "sf":
        return f"Score forest {float(v):.2f}"
    if k == "weights":
        return "Equal weights" if v == "equal" else f"{WEIGHT_NAMES[v[:-2]]} weight ×3"
    if k == "T":
        return f"Temperature {float(v):g}"
    return f"kNN k = {v}"


def main(args):
    alr = pd.read_csv(os.path.join(args.stats_dir, "iso_sensitivity_alr.csv"))
    summ = pd.read_csv(os.path.join(args.stats_dir, "sampling_sensitivity_summary.csv"))
    summ = summ[summ["sample_size"] == N_FOCUS]

    fs.setup()
    fig, (axA, axB) = fs.plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=300,
                                      gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1.1]})

    # a: Tier 1 PD coefficient
    fs.style(axA, grid_axis="x")
    alr = alr.iloc[::-1].reset_index(drop=True)          # baseline on top
    y = np.arange(len(alr))
    base = alr.loc[alr["varied"] == "baseline", "coef_pd"].iloc[0]
    axA.axvline(base, color=fs.GRID, lw=1.5, zorder=1)
    axA.axvline(0, color=fs.MUTED, lw=0.9)
    axA.hlines(y, alr["ci_low_pd"], alr["ci_high_pd"], color=fs.BLUE, lw=1.6, zorder=2)
    colors = [fs.INK if v == "baseline" else fs.BLUE for v in alr["varied"]]
    axA.scatter(alr["coef_pd"], y, s=40, color=colors, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    axA.set_yticks(y)
    axA.set_yticklabels(alr.apply(tier1_label, axis=1), fontsize=8.5)
    axA.set_ylim(-0.7, len(alr) - 0.3)
    axA.set_xlabel("PD coefficient in zero-shot ALR model", fontsize=10)
    axA.text(0.02, 1.01, "MRQAP p = 0.0002 in every row", transform=axA.transAxes,
             fontsize=8.5, color=fs.INK2, va="bottom")
    fs.panel(axA, "a", x=-0.62)

    # b: Tier 2 PD share of the selection
    fs.style(axB, grid_axis="x")
    settings = list(dict.fromkeys(summ.sort_index()["setting"]))
    settings = [s for s in settings if s != "weights=equal"] + ["weights=equal"]
    settings = settings[::-1]
    yy = {s: i for i, s in enumerate(settings)}
    pool = 100 * summ["pool_pd_share"].mean()
    axB.axvline(pool, color=fs.MUTED, lw=1, ls=(0, (4, 3)))
    axB.text(pool, len(settings) - 0.2, " pool = random", fontsize=8, color=fs.INK2, va="bottom")
    for strat, name, color, marker, dy in [("oracle_t1", "Oracle SAT", fs.BLUE, "o", 0.14),
                                           ("sat_prospective", "Prospective SAT", fs.ORANGE, "s", -0.14)]:
        d = summ[summ["sampling"] == strat]
        ys = np.array([yy[s] for s in d["setting"]]) + dy
        axB.scatter(100 * d["pd_share"], ys, s=34, color=color, marker=marker,
                    edgecolor=fs.SURFACE, linewidth=1, zorder=3, label=name)
    axB.set_yticks(range(len(settings)))
    axB.set_yticklabels([tier2_label(s) for s in settings], fontsize=8.5)
    axB.set_ylim(-0.7, len(settings) + 0.3)
    axB.set_xlabel(f"PD share of selection, n = {N_FOCUS:,} (%)", fontsize=10)
    axB.legend(loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.3)
    fs.panel(axB, "b", x=-0.5)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="../../results/review_stats")
    ap.add_argument("--out", default="../../figures/figS_sensitivity.png")
    main(ap.parse_args())
