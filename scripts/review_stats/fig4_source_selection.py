"""
fig4_source_selection.py  (review round 1; Fig 4 of the revision)
================================================================

What a seed sample of target comparisons tells a practitioner before finetuning
(decision 3, R2-3/R2-4; revision_notes_R1.md §5.1). Two rows, two panels each:

  a  per target city, finetuned Spearman (n = 1,000) of the city chosen on a
     500-comparison seed, against a random city (mean over candidates) and the best
     city in hindsight
  b  shortfall of each rule from the best city in hindsight, by number of target
     comparisons (mean over 7 targets x 5 seeds); dashed: choosing by zero-shot
     Spearman on the full test set (retrospective reference)
  c  seed accuracy vs gain from finetuning on n = 1,000 (exact QAP Spearman)
  d  seed accuracy vs zero-shot Spearman, line fitted on all pairs, band = error
     for a target city left out of the fit

Outcomes are seed-mean final Spearman from the five-seed runs (random sampling).

  python fig4_source_selection.py
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
TICK, LABEL, NOTE, LEG, LETTER = 13, 14.5, 12.5, 12, 18
RULES_B = [("random", "Random city", fs.NEUTRAL, "o"),
           ("coverage", "Best-covering city", fs.AQUA, "D"),
           ("seed_accuracy", "100-comparison seed", fs.ORANGE, "s"),
           ("seed500_accuracy", "500-comparison seed", fs.BLUE, "o")]


def main(args):
    ch = pd.read_csv(os.path.join(args.stats_dir, "source_selection_choices.csv"))
    reg = pd.read_csv(os.path.join(args.stats_dir, "source_selection_regret.csv"))
    runs, _ = load_runs(args.runs_zip)
    r = runs[runs["sampling"] == "random"]
    final = r.groupby(["source", "target", "sample_size"])["spearman"].mean().unstack()
    zs = r.groupby(["source", "target"])["zero_spearman"].first()

    fs.setup()
    fig, axes = fs.plt.subplots(2, 2, figsize=(13, 11.2), dpi=300, gridspec_kw={"hspace": 0.42, "wspace": 0.3})
    (axA, axB), (axC, axD) = axes
    for ax in axes.flat:
        fs.style(ax)
        ax.tick_params(labelsize=TICK)

    # a: per target, chosen city between a random city and the best city
    c = ch[ch["sample_size"] == N_FOCUS]
    per = c.groupby(["rule", "target"])["value"].mean().unstack(0)
    per["random"] = final[N_FOCUS].groupby(level="target").mean()
    order = per["best"].sort_values().index.tolist()
    y = np.arange(len(order))
    axA.grid(axis="y", visible=False)
    for i, t in enumerate(order):
        lo, hi = per.loc[t, ["random", "seed500_accuracy", "best"]].min(), per.loc[t, ["random", "seed500_accuracy", "best"]].max()
        axA.plot([lo, hi], [i, i], color=fs.GRID, lw=3, zorder=1, solid_capstyle="round")
    axA.scatter(per.loc[order, "random"], y, s=90, color=fs.NEUTRAL, edgecolor=fs.SURFACE, lw=1.2, zorder=3,
                label="Random city (mean)")
    axA.scatter(per.loc[order, "seed500_accuracy"], y, s=90, color=fs.BLUE, edgecolor=fs.SURFACE, lw=1.2, zorder=4,
                label="City chosen on 500-comparison seed")
    axA.scatter(per.loc[order, "best"], y, s=190, facecolor="none", edgecolor=fs.INK, lw=1.5, zorder=5,
                label="Best city in hindsight")
    axA.set_yticks(y)
    axA.set_yticklabels([NAMES[t] for t in order], fontsize=TICK)
    axA.set_ylim(-0.7, len(order) + 1.9)
    axA.set_xlabel(f"Finetuned Spearman (n = {N_FOCUS:,})", fontsize=LABEL)
    axA.legend(loc="upper left", frameon=False, fontsize=LEG, handletextpad=0.4, borderaxespad=0.2)

    # b: shortfall by number of target comparisons
    sizes = sorted(reg["sample_size"].unique())
    xpos = {n: i for i, n in enumerate(sizes)}
    for k, name, color, marker in RULES_B:
        g = reg[reg["rule"] == k].set_index("sample_size")["mean_regret"]
        if k == "seed500_accuracy":
            g = g[g.index >= 500]          # a 500-comparison seed needs at least 500 comparisons
        axB.plot([xpos[n] for n in g.index], g.values, color=color, marker=marker, lw=2.4, ms=8,
                 markeredgecolor=fs.SURFACE, markeredgewidth=1, zorder=3, label=name)
    g = reg[reg["rule"] == "zs_spearman"].set_index("sample_size")["mean_regret"]
    axB.plot([xpos[n] for n in g.index], g.values, color=fs.INK2, lw=1.6, ls=(0, (4, 3)), zorder=2,
             label="Zero-shot, full test\nset (retrospective)")
    axB.set_xticks(range(len(sizes)))
    axB.set_xticklabels(["0"] + [f"{n:,}" for n in sizes[1:]], fontsize=TICK)
    axB.set_xlabel("Target comparisons used for finetuning", fontsize=LABEL)
    axB.set_ylabel("Shortfall from best city\n(Spearman)", fontsize=LABEL)
    axB.set_ylim(0, 0.37)
    axB.set_yticks(np.arange(0, 0.26, 0.05))
    axB.legend(loc="upper right", frameon=False, fontsize=LEG - 1, handlelength=1.8, ncol=2, columnspacing=1.0,
               borderaxespad=0.1)

    # c: seed accuracy vs gain from finetuning
    seed = r[r["sample_size"] == 500].groupby(["source", "target"])["lab_source_accuracy"].mean()
    d = pd.DataFrame({"acc": seed, "gain": final[N_FOCUS] - zs}).reset_index()
    q = di.qap_spearman(d, "acc", "gain")
    axC.axhline(0, color=fs.MUTED, lw=0.9)
    axC.scatter(100 * d["acc"], d["gain"], s=60, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    axC.text(0.97, 0.96, f"ρ = {q['rho']:.2f}, {fs.p_text(q['p_qap'])}".replace("-", "−"), transform=axC.transAxes,
             fontsize=NOTE, color=fs.INK2, ha="right", va="top")
    axC.set_xlabel("Accuracy on 500-comparison seed (%)", fontsize=LABEL)
    axC.set_ylabel(f"Gain from finetuning\n(n = {N_FOCUS:,}, Spearman)", fontsize=LABEL)

    # d: seed accuracy -> expected zero-shot Spearman; band = leave-one-target-out error
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
    axD.plot(100 * xs, a + b * xs, color=fs.BLUE, lw=2)
    axD.scatter(100 * d["acc"], d["zs"], s=60, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    axD.text(0.03, 0.96, f"Held-out city: r = {r_loto:.2f}\nmean absolute error = {mae:.2f}", transform=axD.transAxes,
             fontsize=NOTE, color=fs.INK2, ha="left", va="top")
    axD.set_xlabel("Accuracy on 500-comparison seed (%)", fontsize=LABEL)
    axD.set_ylabel("Zero-shot Spearman", fontsize=LABEL)

    for ax, letter in zip(axes.flat, "abcd"):
        ax.set_title(letter, loc="left", fontsize=LETTER, fontweight="bold", x=-0.2 if letter != "a" else -0.24, y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}\n  c: rho={q['rho']:.3f} p={q['p_qap']:.4f}"
          f"\n  d: zs = {a:.3f} + {b:.3f} * acc; LOTO r={r_loto:.3f}, MAE={mae:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="../../results/review_stats")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--out", default="../../figures/fig5_source_selection.png")
    main(ap.parse_args())
