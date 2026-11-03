"""
fig5_source_types.py  (review round 1; Fig 5 of the revision, replaces fig3_source_types.png)
=============================================================================================

Single-city sources vs multi-city aggregates, compared within each target city, as
differences in Spearman correlation by number of target comparisons (0 = zero-shot):

  a  single city minus the mean of the tier and continental aggregates, for the
     average single city and for the single city chosen on the 500-comparison seed
  b  chosen single city minus the aggregate chosen on the same seed

Dots are target cities, lines are means over the seven targets; in b the labels give
the number of targets where the single city was higher and the exact Wilcoxon p.
Aggregates and the average single city come from the three-seed runs, the chosen
single city from the five-seed runs (random sampling); see fig3_source_types.py.

  python fig5_source_types.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import fig_style as fs
from aggregation_inference import source_type
from fig3_source_types import NAMES, study_values, rerun_values, seed_choices

BUDGETS = [0, 500, 1000, 5000]
TICK, LABEL, NOTE, LEG, LETTER = 13, 14.5, 12, 12.5, 18


def main(args):
    study = study_values(args.results_zip)
    rerun = rerun_values(args.runs_zip)
    picks = seed_choices(args.seed_preds)
    st = study.reset_index()
    st["type"] = st["source"].map(source_type)

    agg_mean = st[st["type"] != "single"].groupby(["target", "type"])[BUDGETS].mean().groupby("target").mean()
    average = st[st["type"] == "single"].groupby("target")[BUDGETS].mean()
    chosen = pd.DataFrame({n: picks.apply(lambda p: rerun.loc[(p["single"], p["target"]), n], axis=1)
                           .groupby(picks["target"]).mean() for n in BUDGETS})
    chosen_agg = pd.DataFrame({n: picks.apply(lambda p: study.loc[(p["aggregate"], p["target"]), n], axis=1)
                               .groupby(picks["target"]).mean() for n in BUDGETS})
    d_avg, d_cho, d_cc = average - agg_mean, chosen - agg_mean, chosen - chosen_agg

    fs.setup()
    fig, (a, b) = fs.plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=300, gridspec_kw={"wspace": 0.3})
    x = np.arange(len(BUDGETS))
    labels = ["0\n(zero-shot)"] + [f"{n:,}" for n in BUDGETS[1:]]
    for ax in (a, b):
        fs.style(ax)
        ax.tick_params(labelsize=TICK)
        ax.axhline(0, color=fs.MUTED, lw=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=TICK)
        ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
        ax.set_xlabel("Target comparisons used for finetuning", fontsize=LABEL)

    for d, name, color, marker, dx in [(d_avg, "Average single city", fs.NEUTRAL, "s", -0.1),
                                       (d_cho, "Single city chosen on seed", fs.BLUE, "o", 0.1)]:
        for k, n in enumerate(BUDGETS):
            a.scatter(np.full(len(d), x[k] + dx), d[n], s=34, color=color, alpha=0.4, edgecolor="none", zorder=2)
        a.plot(x + dx, d.mean().values, color=color, lw=2.6, marker=marker, ms=9, markeredgecolor=fs.SURFACE,
               zorder=3, label=name)
    for t in d_avg[0].nsmallest(2).index:          # Accra and Tema: their continental aggregate holds the other Ghanaian city
        a.annotate(NAMES[t], (x[0] - 0.1, d_avg.loc[t, 0]), xytext=(-8, 0), textcoords="offset points",
                   ha="right", va="center", fontsize=NOTE - 1, color=fs.INK2)
    a.set_ylabel("Single city − mean of aggregates\n(Spearman)", fontsize=LABEL)
    a.legend(loc="upper right", frameon=False, fontsize=LEG)

    for k, n in enumerate(BUDGETS):
        b.scatter(np.full(len(d_cc), x[k]), d_cc[n], s=34, color=fs.BLUE, alpha=0.4, edgecolor="none", zorder=2)
        p = wilcoxon(d_cc[n], method="exact").pvalue
        b.annotate(f"{int((d_cc[n] > 0).sum())}/7\n{fs.p_text(p)}", (x[k], d_cc[n].max()), xytext=(0, 8),
                   textcoords="offset points", ha="center", va="bottom", fontsize=NOTE, color=fs.INK2)
    b.plot(x, d_cc.mean().values, color=fs.BLUE, lw=2.6, marker="o", ms=9, markeredgecolor=fs.SURFACE, zorder=3)
    b.set_ylabel("Chosen single city − chosen aggregate\n(Spearman)", fontsize=LABEL)
    b.set_ylim(min(-0.12, d_cc.min().min() - 0.03), d_cc.max().max() + 0.13)
    lo = min(a.get_ylim()[0], -0.3)
    a.set_ylim(lo, max(a.get_ylim()[1], 0.45))

    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_title(letter, loc="left", fontsize=LETTER, fontweight="bold", x=-0.2, y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}")
    for name, d in [("average - aggregates", d_avg), ("chosen - aggregates", d_cho), ("chosen - chosen aggregate", d_cc)]:
        print(name, {n: (round(d[n].mean(), 3), int((d[n] > 0).sum()), round(wilcoxon(d[n], method="exact").pvalue, 3))
                     for n in BUDGETS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--seed-preds", default="../../results/seed_predictions.csv.gz")
    ap.add_argument("--out", default="../../figures/fig6_source_types.png")
    main(ap.parse_args())
