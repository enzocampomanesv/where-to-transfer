"""
coverage_thresholds.py  (review round 1: R1-M7, MMD / overlap thresholds)
=========================================================================

Does the coverage reading (coverage tracks Structural Ignorance, i.e. covariate
shift, but not zero-shot performance or the finetuning gain) depend on the arbitrary choices in the Tier 1 measures?

  * overlap ratio at k = 10, k = 20 and their average (the paper's measure), and,
    if --percentile-csv is given (t1_overlap_percentiles.py, needs the embeddings
    on CRIB), at the 90th / 95th / 99th percentile coverage threshold
  * the coverage category under a grid of cut-offs around the paper's
    MMD > 0.05 and overlap < 0.90

For every variant: exact QAP Spearman with the Structural Ignorance share
(construct), the PD rate among in-domain pairs (should stay unrelated), zero-shot
Spearman, and the leak-free finetuning gain at n = 1,000 (random sampling).

  python coverage_thresholds.py --validation-zip ../../results/tier1/validation.zip \
      --results-zip ../../results/random__layerwise_lr1e-04.zip \
      --runs-zip ../../results/sampling_v3.zip --out-dir ../../results/review_stats
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

import dyadic_inference as di
from alr_inference import load_zero_shot
from sampling_inference import load_runs

OUTCOMES = {"structural_ignorance": "Structural Ignorance share",
            "pd_rate_in_domain": "PD rate among in-domain pairs",
            "zero_spearman": "Zero-shot Spearman",
            "gain_1000": "Finetuning gain, n = 1,000"}
MMD_CUTS = [0.03, 0.04, 0.05, 0.06, 0.07]
OVERLAP_CUTS = [0.85, 0.875, 0.90, 0.925, 0.95]


def load_pairs(validation_zip, results_zip, runs_zip):
    zf = zipfile.ZipFile(validation_zip)
    df = pd.read_csv(io.BytesIO(zf.read("validation/validation_merged_zeroshot.csv"))).rename(
        columns={"source_city": "source", "target_city": "target"})
    zs = load_zero_shot(results_zip).rename(columns={"spearman": "zero_spearman"})
    df = df.merge(zs, on=["source", "target"])
    df["pd_rate_in_domain"] = df["cultural_disagreement"] / (df["consensus"] + df["cultural_disagreement"])
    runs, _ = load_runs(runs_zip)
    r = runs[(runs["sampling"] == "random") & (runs["sample_size"] == 1000)]
    gain = (r.groupby(["source", "target"])["spearman"].mean()
            - r.groupby(["source", "target"])["zero_spearman"].first()).rename("gain_1000")
    return df.merge(gain.reset_index(), on=["source", "target"])


def category(df, mmd_cut, overlap_cut, overlap_col="overlap"):
    """The paper's rule: 0 = strong match, 1 = caution (one flag), 2 = poor match (both)."""
    return (df["mmd"] > mmd_cut).astype(int) + (df[overlap_col] < overlap_cut).astype(int)


def correlations(df, x):
    out = {}
    for y in OUTCOMES:
        q = di.qap_spearman(df, x, y)
        out[f"rho_{y}"], out[f"p_{y}"] = q["rho"], q["p_qap"]
    return out


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_pairs(args.validation_zip, args.results_zip, args.runs_zip)
    print(f"{len(df)} pairs")

    # continuous measures (QAP Spearman; for overlap, higher = better coverage)
    measures = {"overlap": "Overlap, avg k10/k20, 95th pct (paper)", "overlap_k10": "Overlap, k = 10, 95th pct",
                "overlap_k20": "Overlap, k = 20, 95th pct", "mmd": "MMD"}
    if args.percentile_csv:
        p = pd.read_csv(args.percentile_csv).rename(columns={"source_city": "source", "target_city": "target"})
        df = df.merge(p, on=["source", "target"], how="left")
        for c in [c for c in p.columns if c.startswith("overlap_p")]:
            measures[c] = f"Overlap, {c.replace('overlap_', '')}"
    rows = [{"variant": label, "kind": "continuous", **correlations(df, col)} for col, label in measures.items()]

    # category under a grid of cut-offs
    for m in MMD_CUTS:
        for o in OVERLAP_CUTS:
            df["_cat"] = category(df, m, o)
            counts = df["_cat"].value_counts().reindex([0, 1, 2], fill_value=0)
            row = {"variant": f"Category, MMD > {m}, overlap < {o}", "kind": "category",
                   "paper": (m, o) == (0.05, 0.90), "n_strong": counts[0], "n_caution": counts[1], "n_poor": counts[2]}
            if df["_cat"].nunique() > 1:
                row.update(correlations(df, "_cat"))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "coverage_thresholds.csv"), index=False)

    pd.set_option("display.width", 250)
    cols = ["variant"] + [c for y in OUTCOMES for c in (f"rho_{y}", f"p_{y}")]
    print("\nContinuous measures (exact QAP):")
    print(out[out.kind == "continuous"][cols].round(3).to_string(index=False))
    cat = out[out.kind == "category"]
    print("\nCategory over the cut-off grid (25 variants; paper = MMD 0.05, overlap 0.90):")
    print(cat[cols + ["n_strong", "n_caution", "n_poor"]].round(3).to_string(index=False))
    s = cat.dropna(subset=["rho_structural_ignorance"])
    print("\nSummary over the grid:")
    for y, label in OUTCOMES.items():
        print(f"  {label:<32} rho {s[f'rho_{y}'].min():+.2f} to {s[f'rho_{y}'].max():+.2f};"
              f" QAP p < 0.05 in {(s[f'p_{y}'] < 0.05).sum()}/{len(s)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-zip", default="../../results/tier1/validation.zip")
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--percentile-csv", default=None, help="from t1_overlap_percentiles.py (CRIB)")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    main(ap.parse_args())
