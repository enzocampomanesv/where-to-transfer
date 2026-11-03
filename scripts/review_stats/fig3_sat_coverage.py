"""
fig3_sat_coverage.py  (review round 1; Figs 3 and 4 of the revision)
====================================================================

What the SAT shows, one point per source-target pair, as two figures:

  Fig 3 (single panel): error rate on unfamiliar comparisons vs on covered comparisons (21 pairs with
     more than 2% unfamiliar comparisons; point area ~ sqrt(number of unfamiliar comparisons))
  Fig 4a  coverage (overlap ratio) vs Structural Ignorance share (42 pairs)
  Fig 4b  coverage vs Preference Disagreement rate among covered comparisons, colored by
     zero-shot Spearman (42 pairs)

rho and exact QAP p-values from dyadic_inference.qap_spearman.

  python fig3_sat_coverage.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import dyadic_inference as di
import fig_style as fs
from fig1f_error_rates import pair_rates, SIZE_K, LEGEND_N
from fig2_coverage_vs_disagreement import load, BLUE_SEQ

TICK, LABEL, NOTE, LETTER = 12.5, 14, 12.5, 18


def stat(ax, q, x=0.04, y=0.05, ha="left"):
    p = "p < 0.001" if q["p_qap"] < 0.001 else f"p = {q['p_qap']:.2f}"
    ax.text(x, y, f"ρ = {q['rho']:.2f}, {p}".replace("-", "−"), transform=ax.transAxes, fontsize=NOTE, color=fs.INK2, ha=ha, va="bottom")


def main(args):
    er = pair_rates(args.shares_csv, args.dataset_csv)
    cov = load(args.validation_zip, args.results_zip)
    q_si = di.qap_spearman(cov, "overlap", "structural_ignorance")
    q_pd = di.qap_spearman(cov, "overlap", "pd_rate")

    fs.setup()
    fig1, a = fs.plt.subplots(figsize=(5.6, 5.4), dpi=300)
    fig, (b, c) = fs.plt.subplots(1, 2, figsize=(11.6, 5.4), dpi=300, gridspec_kw={"wspace": 0.42})
    for ax in (a, b, c):
        fs.style(ax)
        ax.tick_params(labelsize=TICK)
        ax.set_box_aspect(1)

    # a: error rates
    lo, hi = 0.20, 0.65
    a.plot([lo, hi], [lo, hi], color=fs.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    a.text(hi - 0.012, hi - 0.045, "equal error rates", ha="right", va="top", fontsize=NOTE - 1, color=fs.INK2,
           rotation=45, rotation_mode="anchor")
    a.scatter(er["err_familiar"], er["err_unfamiliar"], s=SIZE_K * np.sqrt(er["n_unfamiliar"]), color=fs.BLUE,
              alpha=0.75, edgecolor=fs.SURFACE, linewidth=0.8, zorder=2)
    handles = [a.scatter([], [], s=SIZE_K * np.sqrt(n), color=fs.BLUE, alpha=0.75, edgecolor=fs.SURFACE)
               for n in LEGEND_N]
    a.legend(handles, [f"{n:,}" for n in LEGEND_N], title="Unfamiliar\ncomparisons", loc="upper left",
             frameon=False, fontsize=NOTE - 1, title_fontsize=NOTE - 1, labelspacing=1.0, borderpad=0.4,
             handletextpad=0.8)
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    ticks = np.arange(0.2, 0.65, 0.1)
    a.set_xticks(ticks)
    a.set_yticks(ticks)
    a.set_xticklabels([f"{t:.0%}" for t in ticks])
    a.set_yticklabels([f"{t:.0%}" for t in ticks])
    a.set_xlabel("Error rate\n(covered comparisons)", fontsize=LABEL)
    a.set_ylabel("Error rate\n(unfamiliar comparisons)", fontsize=LABEL)

    # b: coverage vs Structural Ignorance
    b.scatter(cov["overlap"], 100 * cov["structural_ignorance"], s=55, color=fs.BLUE, edgecolor=fs.SURFACE,
              linewidth=1.2, zorder=3)
    b.set_xlabel("Coverage\n(overlap ratio)", fontsize=LABEL)
    b.set_ylabel("Structural Ignorance\n(% of test comparisons)", fontsize=LABEL)
    stat(b, q_si, x=0.04, y=0.05)
    low = cov.nsmallest(1, "overlap").iloc[0]
    b.annotate(f"{low['source']}→{low['target']}", (low["overlap"], 100 * low["structural_ignorance"]),
               xytext=(8, 0), textcoords="offset points", fontsize=NOTE - 1.5, color=fs.INK2, va="center")

    # c: coverage vs Preference Disagreement, colored by zero-shot Spearman
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    sc = c.scatter(cov["overlap"], 100 * cov["pd_rate"], c=cov["spearman"], cmap=cmap,
                   vmin=min(0, cov["spearman"].min()), vmax=cov["spearman"].max(),
                   s=55, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    c.set_xlabel("Coverage\n(overlap ratio)", fontsize=LABEL)
    c.set_ylabel("Preference Disagreement\n(% of covered comparisons)", fontsize=LABEL)
    stat(c, q_pd, x=0.04, y=0.05)
    cb = fig.colorbar(sc, cax=c.inset_axes([1.05, 0.0, 0.05, 1.0]))
    cb.set_label("Zero-shot Spearman", fontsize=LABEL - 1, color=fs.INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TICK - 1, colors=fs.INK2, length=2)

    for ax, letter in ((b, "a"), (c, "b")):
        ax.set_title(letter, loc="left", fontsize=LETTER, fontweight="bold", x=-0.36, y=1.03)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig1.savefig(args.out_errors, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out_errors} and {args.out}\n  b: rho={q_si['rho']:.3f} p={q_si['p_qap']:.4f}\n  c: rho={q_pd['rho']:.3f} p={q_pd['p_qap']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares-csv", default="../../results/review_stats/iso_sensitivity_shares.csv")
    ap.add_argument("--dataset-csv", default="../../results/review_stats/dataset_summary.csv")
    ap.add_argument("--validation-zip", default="../../results/tier1/validation.zip")
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out-errors", default="../../figures/fig3_error_rates.png")
    ap.add_argument("--out", default="../../figures/fig4_coverage.png")
    main(ap.parse_args())
