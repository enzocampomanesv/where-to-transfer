"""
figS1_sampling.py  (review round 1; Supplementary Fig. S1)
==========================================================

Difference in finetuned Spearman correlation from random sampling, by number of target
comparisons, for SAT-informed sampling, uncertainty sampling, and the oracle (true SAT
categories, an upper bound). Points: mean over the 42 single-city source-target pairs
(five seeds each); bars: city-bootstrap 95% intervals (sampling_inference.py).

  python figS1_sampling.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fig_style as fs

STRATS = [("sat_prospective", "SAT-informed", fs.BLUE, "o", -0.12),
          ("uncertainty", "Uncertainty", fs.AQUA, "D", 0.0),
          ("oracle_t1", "Oracle (true SAT categories)", fs.ORANGE, "s", 0.12)]


def main(args):
    c = pd.read_csv(args.contrasts_csv)
    c = c[(c["outcome"] == "spearman") & (c["vs"] == "random")]
    sizes = sorted(c["sample_size"].unique())
    x = np.arange(len(sizes))
    fs.setup()
    fig, ax = fs.plt.subplots(figsize=(8, 5), dpi=300)
    fs.style(ax)
    ax.tick_params(labelsize=13)
    ax.axhline(0, color=fs.MUTED, lw=1.1)
    for key, name, color, marker, dx in STRATS:
        g = c[c["sampling"] == key].set_index("sample_size").loc[sizes]
        ax.errorbar(x + dx, g["mean_diff"], yerr=[g["mean_diff"] - g["boot_ci_low"], g["boot_ci_high"] - g["mean_diff"]],
                    fmt=marker, color=color, ms=9, mec=fs.SURFACE, elinewidth=1.8, capsize=4, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=13)
    ax.set_xlabel("Target comparisons used for finetuning", fontsize=14)
    ax.set_ylabel("Difference from random sampling\n(finetuned Spearman)", fontsize=14)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts-csv", default="../../results/review_stats/sampling_contrasts.csv")
    ap.add_argument("--out", default="../../figures/figS1_sampling.png")
    main(ap.parse_args())
