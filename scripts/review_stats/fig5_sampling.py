"""
fig5_sampling.py  (review round 1; rebuilds Fig 5 from the leak-free rerun) -- DRAFT
===================================================================================

Draft pending the co-authors' decision on how to frame the informed-sampling
result. The panels show the data under any framing:

  a  finetuned Spearman of each strategy minus random, by annotation budget:
     seed-averaged paired difference over the 42 pairs, 95% city-bootstrap
     interval (sampling_inference.py)
  b  Preference Disagreement (PD) share of the selected comparisons at n = 1,000,
     per pair (seed mean) and overall mean; the pool share equals random's
  c  oracle gain over random at n = 1,000 vs the pool's PD share, per pair, with
     the exact QAP Spearman test

Replaces the published two-panel figure, which used leaky informed runs, a
0.05 "meaningful improvement" rate and difficulty bands (§5.2-5.3).

  python fig5_sampling.py --runs-zip ../../results/sampling_v3.zip \
      --stats-dir ../../results/review_stats --out ../../figures/fig5_sampling.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import dyadic_inference as di
import fig_style as fs
from sampling_inference import load_runs

STRATS = [("oracle_t1", "Oracle SAT (retrospective)", fs.BLUE, "o"),
          ("sat_prospective", "Prospective SAT", fs.ORANGE, "s"),
          ("uncertainty", "Uncertainty", fs.AQUA, "D")]
N_FOCUS = 1000


def main(args):
    contrasts = pd.read_csv(os.path.join(args.stats_dir, "sampling_contrasts.csv"))
    contrasts = contrasts[contrasts["outcome"] == "spearman"]
    per_pair = pd.read_csv(os.path.join(args.stats_dir, "sampling_per_pair.csv"))
    stage_a = pd.read_csv(os.path.join(args.stats_dir, "sampling_sensitivity_runs.csv"))
    pool_pd = stage_a.groupby(["source", "target"])["pool_pd_share"].first().rename("pool_pd").reset_index()
    runs, _ = load_runs(args.runs_zip)

    fs.setup()
    fig, (axA, axB, axC) = fs.plt.subplots(1, 3, figsize=(13, 4.2), dpi=300,
                                           gridspec_kw={"wspace": 0.32, "width_ratios": [1.15, 0.9, 1]})

    # a: difference to random by budget
    fs.style(axA)
    axA.axhline(0, color=fs.MUTED, lw=0.9)
    sizes = sorted(contrasts["sample_size"].unique())
    for i, (s, name, color, marker) in enumerate(STRATS):
        c = contrasts[contrasts["sampling"] == s].set_index("sample_size").loc[sizes]
        x = np.array(sizes) * (1.12 ** (i - 1))          # small dodge on the log axis
        axA.vlines(x, c["boot_ci_low"], c["boot_ci_high"], color=color, lw=1.6)
        axA.plot(x, c["mean_diff"], color=color, lw=2, marker=marker, ms=6,
                 markeredgecolor=fs.SURFACE, markeredgewidth=1, label=name, zorder=3)
    axA.set_xscale("log")
    axA.set_xticks(sizes)
    axA.set_xticklabels([f"{s:,}" for s in sizes])
    axA.minorticks_off()
    axA.set_xlabel("Annotated comparisons (n)", fontsize=10)
    axA.set_ylabel("Finetuned Spearman − random", fontsize=10)
    axA.legend(loc="upper left", frameon=False, fontsize=8.5, handletextpad=0.4)
    fs.panel(axA, "a", x=-0.2)

    # b: PD share of the selection at n = 1,000
    fs.style(axB, grid_axis="y")
    sel = runs[runs["sample_size"] == N_FOCUS].assign(pd_share=lambda d: d["sel_cultural"] / d["sample_size"])
    sel = sel.groupby(["sampling", "source", "target"])["pd_share"].mean().reset_index()
    order = [("random", "Random", fs.NEUTRAL, "o")] + STRATS
    rng = np.random.RandomState(0)
    for i, (s, name, color, marker) in enumerate(order):
        v = 100 * sel.loc[sel["sampling"] == s, "pd_share"].to_numpy()
        axB.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=color, alpha=0.45,
                    marker=marker, edgecolor="none", zorder=2)
        axB.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=fs.INK, lw=2, zorder=3)
        axB.text(i, 1.01, f"{v.mean():.1f}%", ha="center", va="bottom", fontsize=8.5,
                 color=fs.INK2, transform=axB.get_xaxis_transform())
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels(["Random", "Oracle", "Prospec-\ntive", "Uncer-\ntainty"], fontsize=9)
    axB.set_xlim(-0.6, len(order) - 0.4)
    axB.set_ylabel(f"PD share of selection, n = {N_FOCUS:,} (%)", fontsize=10)
    fs.panel(axB, "b", x=-0.22)

    # c: oracle gain vs pool PD share
    fs.style(axC)
    g = per_pair[(per_pair["outcome"] == "spearman") & (per_pair["sampling"] == "oracle_t1")
                 & (per_pair["sample_size"] == N_FOCUS)].merge(pool_pd, on=["source", "target"])
    q = di.qap_spearman(g, "pool_pd", "diff")
    axC.axhline(0, color=fs.MUTED, lw=0.9)
    axC.scatter(100 * g["pool_pd"], g["diff"], s=40, color=fs.BLUE, edgecolor=fs.SURFACE,
                linewidth=1.2, zorder=3)
    axC.text(0.03, 0.96, f"ρ = {q['rho']:.2f}, QAP {fs.p_text(q['p_qap'])}", transform=axC.transAxes,
             fontsize=9, color=fs.INK2, ha="left", va="top")
    axC.set_xlabel("PD share of the target pool (%)", fontsize=10)
    axC.set_ylabel(f"Oracle − random, n = {N_FOCUS:,}", fontsize=10)
    fs.panel(axC, "c", x=-0.2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}\n  c: rho={q['rho']:.3f} p_QAP={q['p_qap']:.4f}")
    print(sel.groupby("sampling")["pd_share"].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--stats-dir", default="../../results/review_stats")
    ap.add_argument("--out", default="../../figures/fig5_sampling.png")
    main(ap.parse_args())
