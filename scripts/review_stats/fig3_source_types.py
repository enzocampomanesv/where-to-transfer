"""
fig3_source_types.py  (review round 1; rebuilds Fig 3)
======================================================

Single-city sources vs multi-city aggregates, compared within each target city
(R1-M3 / R1-M6):

  a  zero-shot Spearman per target          b  finetuned Spearman, n = 1,000
     markers: average single city, single city chosen on the 500-comparison
     seed, tier aggregate, continental aggregate (target always left out)
  c  seed-chosen single city minus seed-chosen aggregate, by finetuning budget:
     one dot per target, line = mean; exact Wilcoxon over the 7 targets

Aggregates exist only in the original three-seed runs (random sampling), so
aggregate values and the average single city come from those runs; the
seed-chosen single city comes from the five-seed leak-free rerun (random
sampling). For single cities the two agree closely (r = 0.95, mean difference
<= 0.01 at every budget).

  python fig3_source_types.py
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import fig_style as fs
from aggregation_inference import CITIES, source_type
from sampling_inference import load_runs
from source_selection import seed_accuracy_table

MARKERS = [("average", "Single city, average", fs.NEUTRAL, "o"),
           ("chosen", "Single city, chosen on seed", fs.BLUE, "o"),
           ("tier", "Tier aggregate", fs.ORANGE, "s"),
           ("continental", "Continental aggregate", fs.AQUA, "D")]
NAMES = {"ACC": "Accra", "NBO": "Nairobi", "BEI": "Beira", "TEM": "Tema",
         "KIS": "Kisumu", "CHI": "Chimoio", "MUM": "Mumbai"}
N_FOCUS = 1000
BUDGETS = [0, 500, 1000, 5000]
SEED_N = 500


def study_values(results_zip):
    """Seed-mean Spearman per source, target and n (0 = zero-shot) from the original runs."""
    zf = zipfile.ZipFile(results_zip)
    d = pd.concat([pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith(".csv")])
    d = d[d["target"].isin(CITIES)]
    zs = d[d["strategy"] == "zero_shot"].groupby(["source", "target"])["spearman"].first().rename(0)
    ft = d[d["strategy"] != "zero_shot"].groupby(["source", "target", "sample_size"])["spearman"].mean().unstack()
    v = pd.concat([zs, ft], axis=1)
    v.columns = [int(c) for c in v.columns]
    return v


def rerun_values(runs_zip):
    runs, _ = load_runs(runs_zip)
    r = runs[runs["sampling"] == "random"]
    ft = r.groupby(["source", "target", "sample_size"])["spearman"].mean().unstack()
    zs = r.groupby(["source", "target"])["zero_spearman"].first().rename(0)
    v = pd.concat([zs, ft], axis=1)
    v.columns = [int(c) for c in v.columns]
    return v


def seed_choices(seed_preds):
    """Per target and seed: the single city and the aggregate most accurate on the 500-comparison seed."""
    acc = seed_accuracy_table(seed_preds)
    acc = acc[acc["seed_n"] == SEED_N].copy()
    acc["type"] = acc["source"].map(source_type)
    rows = []
    for (t, seed), g in acc.groupby(["target", "random_seed"]):
        g = g.set_index("source")
        rows.append({"target": t, "random_seed": seed,
                     "single": g[g["type"] == "single"]["acc"].idxmax(),
                     "aggregate": g[g["type"] != "single"]["acc"].idxmax()})
    return pd.DataFrame(rows)


def main(args):
    study = study_values(args.results_zip)
    rerun = rerun_values(args.runs_zip)
    picks = seed_choices(args.seed_preds)
    study_t = study.reset_index()
    study_t["type"] = study_t["source"].map(source_type)

    def per_target(n):
        out = {}
        out["average"] = study_t[study_t["type"] == "single"].groupby("target")[n].mean()
        out["tier"] = study_t[study_t["type"] == "tier"].groupby("target")[n].mean()
        out["continental"] = study_t[study_t["type"] == "continental"].groupby("target")[n].mean()
        out["chosen"] = picks.apply(lambda p: rerun.loc[(p["single"], p["target"]), n], axis=1).groupby(picks["target"]).mean()
        return pd.DataFrame(out)

    diff = {}
    for n in BUDGETS:
        s = picks.apply(lambda p: rerun.loc[(p["single"], p["target"]), n], axis=1)
        a = picks.apply(lambda p: study.loc[(p["aggregate"], p["target"]), n], axis=1)
        diff[n] = (s - a).groupby(picks["target"]).mean()
    diff = pd.DataFrame(diff)

    fs.setup()
    fig, axes = fs.plt.subplots(1, 3, figsize=(13, 4.3), dpi=300, gridspec_kw={"wspace": 0.38,
                                                                             "width_ratios": [1, 1, 0.9]})
    pa, pb = per_target(0), per_target(N_FOCUS)
    order = pa["average"].sort_values().index.tolist()
    y = np.arange(len(order))
    for ax, m, label, letter in [(axes[0], pa, "Zero-shot Spearman", "a"),
                                 (axes[1], pb, f"Finetuned Spearman, n = {N_FOCUS:,}", "b")]:
        fs.style(ax, grid_axis="x")
        for i, t in enumerate(order):
            ax.plot([m.loc[t].min(), m.loc[t].max()], [i, i], color=fs.GRID, lw=2, zorder=1)
        for k, name, color, marker in MARKERS:
            ax.scatter(m.loc[order, k], y, s=48 if k != "chosen" else 60, color=color, marker=marker,
                       edgecolor=fs.SURFACE, linewidth=1.2, zorder=4 if k == "chosen" else 3, label=name)
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[t] for t in order] if letter == "a" else [], fontsize=9)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel(label, fontsize=10)
        fs.panel(ax, letter, x=-0.30 if letter == "a" else -0.08)
    axes[0].set_ylabel("Target city", fontsize=10)

    ax = axes[2]
    fs.style(ax)
    ax.axhline(0, color=fs.MUTED, lw=0.9)
    xs = np.arange(len(BUDGETS))
    for j, n in enumerate(BUDGETS):
        ax.scatter(np.full(len(diff), j), diff[n], s=14, color=fs.BLUE, alpha=0.35, edgecolor="none", zorder=2)
    ax.plot(xs, diff.mean(), color=fs.BLUE, lw=2, marker="o", ms=6, markeredgecolor=fs.SURFACE, zorder=3)
    for j, n in enumerate(BUDGETS):
        d = diff[n]
        p = wilcoxon(d, alternative="two-sided", method="exact").pvalue
        ax.text(j, ax.get_ylim()[1] if j == 0 else ax.get_ylim()[1], "", fontsize=7)
        ax.annotate(f"{int((d > 0).sum())}/7\n{fs.p_text(p)}", (j, d.max()), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=fs.INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(zero-shot)" if n == 0 else f"{n:,}" for n in BUDGETS], fontsize=8.5)
    ax.set_xlabel("Target comparisons used for finetuning (n)", fontsize=10)
    ax.set_ylabel("Chosen single city − chosen aggregate\n(Spearman)", fontsize=9.5)
    fs.panel(ax, "c", x=-0.24)
    ymax = diff.max().max()
    ax.set_ylim(min(-0.1, diff.min().min() - 0.03), ymax + 0.12)

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.2), ncol=4, frameon=False, fontsize=9,
                   handletextpad=0.3, columnspacing=1.4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)

    def contrast(m, a, b="agg"):
        agg = m[["tier", "continental"]].mean(axis=1) if b == "agg" else m[b]
        d = (m[a] - agg).dropna()
        return d.mean(), int((d > 0).sum()), wilcoxon(d, alternative="two-sided", method="exact").pvalue
    print(f"saved {args.out}")
    for n, m in [(0, pa), (N_FOCUS, pb)]:
        print(f"n={n}: average single - aggregates {contrast(m, 'average')}, chosen single - aggregates {contrast(m, 'chosen')}")
    print("chosen single - chosen aggregate by budget (mean, targets single better, p):")
    for n in BUDGETS:
        d = diff[n]
        print(f"  {n}: {d.mean():+.3f}, {int((d > 0).sum())}/7, p = {wilcoxon(d, method='exact').pvalue:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--seed-preds", default="../../results/seed_predictions.csv.gz")
    ap.add_argument("--out", default="../../figures/fig3_source_types.png")
    main(ap.parse_args())
