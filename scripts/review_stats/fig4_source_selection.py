"""
fig4_source_selection.py  (review round 1; rebuilds Fig 4)
==========================================================

What a seed sample of target comparisons tells a practitioner before any
finetuning (decision 3, R2-3/R2-4). Replaces the
published Fig 4, whose "ceiling-recommended" source used the full target labels.

  a  per target, final Spearman after finetuning on n = 1,000 random comparisons
     for the source chosen by each rule: a random single city (mean over
     candidates), the best-covering city (label-free), the city most accurate on a
     100- or 500-comparison seed, and the best city in hindsight
  b  regret of each rule (best city's final Spearman - chosen city's) by budget,
     mean over 7 targets x 5 seeds; the full-test zero-shot rule (the information
     the published ceiling used) is shown as a retrospective reference
  c  seed accuracy of each source-target pair vs the gain from finetuning on
     n = 1,000, with the exact QAP Spearman test

Outcomes are seed-mean final Spearman from the leak-free rerun (random sampling).

  python fig4_source_selection.py --stats-dir ../../results/review_stats \
      --runs-zip ../../results/sampling_v3.zip --out ../../figures/fig4_source_selection.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import dyadic_inference as di
import fig_style as fs
from fig3_source_types import NAMES
from sampling_inference import load_runs

N_FOCUS = 1000
RULES = [("random", "Random city", fs.NEUTRAL, "o"),
         ("coverage", "Best-covering city (label-free)", fs.AQUA, "D"),
         ("seed_accuracy", "Most accurate on 100-comparison seed", fs.ORANGE, "s"),
         ("seed500_accuracy", "Most accurate on 500-comparison seed", fs.BLUE, "o")]


def main(args):
    ch = pd.read_csv(os.path.join(args.stats_dir, "source_selection_choices.csv"))
    reg = pd.read_csv(os.path.join(args.stats_dir, "source_selection_regret.csv"))
    runs, _ = load_runs(args.runs_zip)
    r = runs[runs["sampling"] == "random"]
    final = r.groupby(["source", "target", "sample_size"])["spearman"].mean().unstack()
    zs = r.groupby(["source", "target"])["zero_spearman"].first()

    fs.setup()
    fig, (axA, axB, axC, axD) = fs.plt.subplots(1, 4, figsize=(18.5, 4.6), dpi=300,
                                                gridspec_kw={"wspace": 0.38, "width_ratios": [1.05, 1, 0.95, 0.95]})

    # a: per target at n = 1,000
    fs.style(axA, grid_axis="x")
    c = ch[ch["sample_size"] == N_FOCUS]
    per_rule = c.groupby(["rule", "target"])["value"].mean().unstack(0)
    per_rule["random"] = final[N_FOCUS].groupby(level="target").mean()
    order = per_rule["best"].sort_values().index.tolist()
    y = np.arange(len(order))
    for i, t in enumerate(order):
        v = per_rule.loc[t, [k for k, *_ in RULES] + ["best"]]
        axA.plot([v.min(), v.max()], [i, i], color=fs.GRID, lw=2, zorder=1)
    for (k, name, color, marker), dy in zip(RULES, [0.2, 0.07, -0.07, -0.2]):   # dodge: values can coincide
        axA.scatter(per_rule.loc[order, k], y + dy, s=40, color=color, marker=marker,
                    edgecolor=fs.SURFACE, linewidth=1.1, zorder=3, label=name)
    axA.scatter(per_rule.loc[order, "best"], y, s=70, facecolor="none", edgecolor=fs.INK,
                linewidth=1.2, zorder=4, label="Best city in hindsight")
    axA.set_yticks(y)
    axA.set_yticklabels([NAMES[t] for t in order], fontsize=9)
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.set_xlabel(f"Finetuned Spearman, n = {N_FOCUS:,}", fontsize=10)
    axA.legend(loc="upper center", bbox_to_anchor=(1.15, 1.25), ncol=3, frameon=False,
               fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
    fs.panel(axA, "a", x=-0.28)

    # b: regret by budget
    fs.style(axB)
    sizes = sorted(reg["sample_size"].unique())
    xpos = {n: i for i, n in enumerate(sizes)}
    for k, name, color, marker in RULES:
        g = reg[reg["rule"] == k].set_index("sample_size")["mean_regret"]
        if k == "seed500_accuracy":
            g = g[g.index >= 500]          # the 500-comparison seed only applies from n = 500
        axB.plot([xpos[n] for n in g.index], g.values, color=color, marker=marker, lw=2, ms=6,
                 markeredgecolor=fs.SURFACE, markeredgewidth=1, zorder=3)
    g = reg[reg["rule"] == "zs_spearman"].set_index("sample_size")["mean_regret"]
    axB.plot([xpos[n] for n in g.index], g.values, color=fs.INK2, lw=1.3, ls=(0, (4, 3)), zorder=2)
    axB.text(0.1, g.values[0] + 0.006, "full-test zero-shot Spearman (retrospective)",
             fontsize=7.5, color=fs.INK2, ha="left", va="bottom")
    axB.set_xticks(range(len(sizes)))
    axB.set_xticklabels(["0\n(zero-shot)"] + [f"{n:,}" for n in sizes[1:]], fontsize=9)
    axB.set_xlabel("Annotated comparisons used for finetuning (n)", fontsize=10)
    axB.set_ylabel("Regret (Spearman lost vs best city)", fontsize=10)
    axB.set_ylim(bottom=0)
    fs.panel(axB, "b", x=-0.2)

    # c: seed accuracy predicts how much finetuning adds
    fs.style(axC)
    seed = r[r["sample_size"] == 500].groupby(["source", "target"])["lab_source_accuracy"].mean()
    d = pd.DataFrame({"acc": seed, "gain": final[N_FOCUS] - zs}).reset_index()
    q = di.qap_spearman(d, "acc", "gain")
    axC.axhline(0, color=fs.MUTED, lw=0.9)
    axC.scatter(100 * d["acc"], d["gain"], s=40, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    axC.text(0.97, 0.96, f"ρ = {q['rho']:.2f}, QAP {fs.p_text(q['p_qap'])}", transform=axC.transAxes,
             fontsize=9, color=fs.INK2, ha="right", va="top")
    for _, row in d.nlargest(2, "acc").iterrows():
        axC.annotate(f"{row['source']}→{row['target']}", (100 * row["acc"], row["gain"]),
                     xytext=(-6, 6), textcoords="offset points", fontsize=8, color=fs.INK2, ha="right")
    axC.set_xlabel("Accuracy on 500-comparison seed (%)", fontsize=10)
    axC.set_ylabel(f"Gain from finetuning, n = {N_FOCUS:,}", fontsize=10)
    fs.panel(axC, "c", x=-0.2)

    # d: seed accuracy -> expected zero-shot Spearman (replaces the published ceiling).
    # Line fitted on all pairs; the band is the leave-one-target-out error, so the
    # reported accuracy is that of predicting a city the line has not seen.
    fs.style(axD)
    d = pd.DataFrame({"acc": seed, "zs": zs}).dropna().reset_index()
    loto = np.zeros(len(d))
    for t in d["target"].unique():
        tr = d["target"] != t
        b, a = np.polyfit(d.loc[tr, "acc"], d.loc[tr, "zs"], 1)
        loto[~tr] = a + b * d.loc[~tr, "acc"]
    mae = np.abs(loto - d["zs"]).mean()
    r_loto = np.corrcoef(loto, d["zs"])[0, 1]
    b, a = np.polyfit(d["acc"], d["zs"], 1)
    xs = np.linspace(d["acc"].min() - 0.01, d["acc"].max() + 0.01, 50)
    axD.fill_between(100 * xs, a + b * xs - mae, a + b * xs + mae, color=fs.BLUE, alpha=0.12, lw=0)
    axD.plot(100 * xs, a + b * xs, color=fs.BLUE, lw=1.6)
    axD.scatter(100 * d["acc"], d["zs"], s=40, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    ref = 0.70                                            # our reference for "sufficient" seed accuracy
    axD.axvline(100 * ref, color=fs.INK2, lw=1, ls=(0, (2, 2)))
    axD.text(100 * ref + 0.4, axD.get_ylim()[0] + 0.02, f"70% → ≈ {a + b * ref:.2f}", fontsize=8,
             color=fs.INK2, ha="left", va="bottom")
    axD.text(0.03, 0.96, f"held-out city: r = {r_loto:.2f}, error ± {mae:.2f}", transform=axD.transAxes,
             fontsize=9, color=fs.INK2, ha="left", va="top")
    axD.set_xlabel("Accuracy on 500-comparison seed (%)", fontsize=10)
    axD.set_ylabel("Zero-shot Spearman (test set)", fontsize=10)
    fs.panel(axD, "d", x=-0.2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}\n  c: rho={q['rho']:.3f} p={q['p_qap']:.4f}"
          f"\n  d: zs = {a:.3f} + {b:.3f} * acc; LOTO r={r_loto:.3f}, MAE={mae:.3f}; at 70%: {a + b * ref:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="../../results/review_stats")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--out", default="../../figures/fig4_source_selection.png")
    main(ap.parse_args())
