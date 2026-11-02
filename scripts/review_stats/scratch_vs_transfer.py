"""
scratch_vs_transfer.py  (review round 1: R3-4)
==============================================

Is transfer worth it? At an equal annotation budget n, the source model
finetuned on n random target comparisons vs the same architecture trained from
scratch on exactly those comparisons (train_from_scratch.py), both tested on the
same held-out split; plus the from-scratch model on the whole training pool as
the full-local reference.

The transfer arm is taken from the leak-free rerun (random sampling) in three
versions, per target and seed:
  selected   the single city most accurate on the seed sample (the guide's rule;
             500-comparison seed for n >= 500, the 100-comparison seed at n = 100)
  average    mean over the six candidate cities
  best       the best city in hindsight (by seed-mean Spearman at that n)

Runs are paired by seed (42-44): random_sample(train_pool, n, seed) gives both
arms the same comparisons. Per target the seed-mean difference transfer -
scratch is computed; the 7 targets are summarised with the exact Wilcoxon
signed-rank test.

  python scratch_vs_transfer.py --scratch-csv ../../results/from_scratch.csv \
      --runs-zip ../../results/sampling_v3.zip --out-dir ../../results/review_stats \
      --fig ../../figures/figS_scratch_vs_transfer.png

Writes scratch_vs_transfer_pairs.csv (per target x seed x n) and
scratch_vs_transfer_summary.csv; draws the figure.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import fig_style as fs
from fig3_source_types import NAMES
from sampling_inference import load_runs

ARMS = [("selected", "Transfer, city chosen on seed", fs.BLUE, "o"),
        ("average", "Transfer, average city", fs.ORANGE, "s"),
        ("best", "Transfer, best city (hindsight)", fs.AQUA, "D")]


def load_scratch(path):
    s = pd.read_csv(path)
    s["sample_size"] = s["sample_size"].astype(str)
    full = s[s["sample_size"] == "full"].groupby("target")["spearman"].mean().rename("full")
    s = s[s["sample_size"] != "full"].assign(sample_size=lambda d: d["sample_size"].astype(int))
    return s.rename(columns={"spearman": "scratch", "accuracy": "scratch_acc"}), full


def transfer_arms(runs_zip, sizes, seeds, long=None):
    """Seed-chosen / average / best transfer arms. With `long` (transfer_long.csv rows of
    one recipe), Spearman comes from those runs; the seed choice still uses the rerun's
    seed-sample accuracy of the zero-shot source model, which the recipe does not touch."""
    runs, _ = load_runs(runs_zip)
    r = runs[(runs["sampling"] == "random") & runs["random_seed"].isin(seeds)]
    acc = r.set_index(["source", "target", "random_seed", "sample_size"])["lab_source_accuracy"]
    if long is not None:
        r = long[long["random_seed"].isin(seeds)]
    sp = r.set_index(["source", "target", "random_seed", "sample_size"])["spearman"]
    seed_mean = r.groupby(["source", "target", "sample_size"])["spearman"].mean()
    rows = []
    for (t, seed), g in r.groupby(["target", "random_seed"]):
        cands = sorted(g["source"].unique())
        sizes_t = [n for n in sizes if n in set(g["sample_size"])]
        for n in sizes_t:
            seed_n = 100 if n < 500 else 500
            chosen = max(cands, key=lambda s: acc[(s, t, seed, seed_n)])
            best = max(cands, key=lambda s: seed_mean[(s, t, n)])
            rows.append({"target": t, "random_seed": seed, "sample_size": n,
                         "selected": sp[(chosen, t, seed, n)], "selected_source": chosen,
                         "average": np.mean([sp[(s, t, seed, n)] for s in cands]),
                         "best": sp[(best, t, seed, n)], "best_source": best})
    return pd.DataFrame(rows)


def summarise(pairs, full):
    out = []
    for n, g in pairs.groupby("sample_size"):
        per_t = g.groupby("target")[["selected", "average", "best", "scratch"]].mean()
        for arm, *_ in ARMS:
            d = per_t[arm] - per_t["scratch"]
            p = wilcoxon(d, alternative="two-sided", method="exact").pvalue if len(d) >= 5 else np.nan
            out.append({"sample_size": n, "arm": arm, "mean_transfer": per_t[arm].mean(),
                        "mean_scratch": per_t["scratch"].mean(), "mean_diff": d.mean(),
                        "targets_transfer_better": f"{int((d > 0).sum())}/{len(d)}", "p_wilcoxon_exact": p,
                        "transfer_share_of_full": (per_t[arm] / full.reindex(per_t.index)).mean(),
                        "scratch_share_of_full": (per_t["scratch"] / full.reindex(per_t.index)).mean()})
    return pd.DataFrame(out)


# long-budget rerun of the seed-chosen arm (transfer_long_budget.py), drawn as hollow markers
LONG = {"layerwise": ("Transfer, chosen city, long budget (layer-wise)", "o", 1.12),
        "match": ("Transfer, chosen city, long budget (full finetuning)", "^", 1.25)}


def figure(pairs, full, path, long_sel=None):
    long_sel = long_sel or {}
    fs.setup()
    fig, axes = fs.plt.subplots(2, 4, figsize=(13, 6.2), dpi=300, sharex=True,
                                gridspec_kw={"wspace": 0.3, "hspace": 0.38})
    m = pairs.groupby(["target", "sample_size"])[["selected", "average", "best", "scratch"]].mean()
    sizes = sorted(pairs["sample_size"].unique())
    targets = m.xs(sizes[-1], level="sample_size")["selected"].sort_values(ascending=False).index
    for ax, t in zip(axes.flat, targets):
        fs.style(ax)
        d = m.xs(t, level="target")
        if t in full.index and pd.notna(full[t]):
            ax.axhline(full[t], color=fs.INK2, lw=1.1, ls=(0, (4, 3)))
        ax.plot(d.index, d["scratch"], color=fs.NEUTRAL, lw=2, marker="o", ms=5,
                markeredgecolor=fs.SURFACE, label="From scratch, same comparisons")
        for arm, name, color, marker in ARMS:
            ax.plot(d.index, d[arm], color=color, lw=2 if arm == "selected" else 1.3, marker=marker, ms=5,
                    markeredgecolor=fs.SURFACE, alpha=1 if arm == "selected" else 0.8, label=name)
        for recipe, (name, marker, dodge) in LONG.items():
            if recipe in long_sel:
                v = long_sel[recipe].xs(t, level="target")
                ax.plot(v.index * dodge, v.values, ls=(0, (2, 2)), lw=1, color=fs.BLUE, marker=marker, ms=6.5,
                        mfc=fs.SURFACE, mec=fs.BLUE, mew=1.6, label=name)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=8)
        ax.minorticks_off()
        ax.set_title(NAMES[t], fontsize=10, loc="left")
    # last panel: mean difference to scratch over targets
    ax = axes.flat[-1]
    fs.style(ax)
    ax.axhline(0, color=fs.MUTED, lw=0.9)
    for k, (arm, name, color, marker) in enumerate(ARMS):
        diff = m[arm] - m["scratch"]
        dodge = 1.09 ** (k - 1)                      # small offset per arm on the log axis
        for n in sizes:                              # one dot per target
            v = diff.xs(n, level="sample_size")
            ax.scatter(np.full(len(v), n * dodge), v, s=10, color=color, alpha=0.35, edgecolor="none")
        mean = diff.groupby(level="sample_size").mean().loc[sizes]
        ax.plot(np.array(sizes) * dodge, mean, color=color, lw=2, marker=marker, ms=5, markeredgecolor=fs.SURFACE)
    for recipe, (name, marker, dodge) in LONG.items():
        if recipe in long_sel:
            v = long_sel[recipe]
            d = v - m["scratch"].reindex(v.index)
            ns = d.index.get_level_values("sample_size")
            ax.scatter(ns * dodge * 1.05, d, s=10, color=fs.BLUE, alpha=0.35, edgecolor="none")
            mean = d.groupby(level="sample_size").mean()
            ax.plot(mean.index * dodge * 1.05, mean.values, ls=(0, (2, 2)), lw=1, color=fs.BLUE, marker=marker,
                    ms=6.5, mfc=fs.SURFACE, mec=fs.BLUE, mew=1.6)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=8)
    ax.minorticks_off()
    ax.set_title("Transfer − scratch (all targets)", fontsize=10, loc="left")
    for a in axes[:, 0]:
        a.set_ylabel("Test Spearman", fontsize=9.5)
    for a in axes[1, :]:
        a.set_xlabel("Target comparisons (n)", fontsize=9.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(fs.plt.Line2D([], [], color=fs.INK2, lw=1.1, ls=(0, (4, 3))))
    labels.append("From scratch, full training pool")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=8.5)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    scratch, full = load_scratch(args.scratch_csv)
    seeds = sorted(scratch["random_seed"].unique())
    sizes = sorted(scratch["sample_size"].unique())
    arms = transfer_arms(args.runs_zip, sizes, seeds)
    pairs = arms.merge(scratch[["target", "random_seed", "sample_size", "scratch", "scratch_acc", "epochs_trained"]],
                       on=["target", "random_seed", "sample_size"])
    missing = len(arms) - len(pairs)
    if missing:
        print(f"note: {missing} transfer rows have no from-scratch partner yet (run incomplete?)")
    summ = summarise(pairs, full)
    pairs.to_csv(os.path.join(args.out_dir, "scratch_vs_transfer_pairs.csv"), index=False)
    summ.to_csv(os.path.join(args.out_dir, "scratch_vs_transfer_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("Full-pool reference (from scratch):\n" + full.round(3).to_string())
    print("\n" + summ.round(3).to_string(index=False))
    long_sel = {}
    if args.long_csv:                       # transfer rerun with the scratch arm's training budget
        long = pd.concat([pd.read_csv(p) for p in args.long_csv], ignore_index=True)
        for recipe, lg in long.groupby("recipe"):
            la = transfer_arms(args.runs_zip, sorted(lg["sample_size"].unique()), seeds, long=lg)
            lp = la.merge(scratch[["target", "random_seed", "sample_size", "scratch"]],
                          on=["target", "random_seed", "sample_size"])
            ls = summarise(lp, full)
            short = pairs.set_index(["target", "random_seed", "sample_size"])["selected"]
            lp["selected_short"] = short.reindex(pd.MultiIndex.from_frame(
                lp[["target", "random_seed", "sample_size"]])).to_numpy()
            lp.to_csv(os.path.join(args.out_dir, f"scratch_vs_transfer_long_{recipe}_pairs.csv"), index=False)
            ls.to_csv(os.path.join(args.out_dir, f"scratch_vs_transfer_long_{recipe}_summary.csv"), index=False)
            n_runs = lg.groupby(["source", "target", "random_seed"]).ngroups
            print(f"\nLong budget, recipe '{recipe}' ({n_runs} runs):\n" + ls.round(3).to_string(index=False))
            per_t = lp.groupby(["sample_size", "target"])[["selected_short", "selected", "best", "scratch"]].mean().round(3)
            print(per_t.assign(sel_minus_scratch=(per_t.selected - per_t.scratch).round(3)).to_string())
            print("epochs trained:", lg["epochs_trained"].describe()[["50%", "min", "max"]].to_dict(),
                  " best epoch median:", lg["best_epoch"].median())
            long_sel[recipe] = lp.groupby(["target", "sample_size"])["selected"].mean()
    figure(pairs, full, args.fig, long_sel)
    print(f"\nsaved {args.fig}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-csv", default="../../results/from_scratch.csv")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--fig", default="../../figures/figS_scratch_vs_transfer.png")
    ap.add_argument("--long-csv", nargs="+", default=None,
                    help="transfer_long_{recipe}.csv file(s) from transfer_long_budget.py")
    main(ap.parse_args())
