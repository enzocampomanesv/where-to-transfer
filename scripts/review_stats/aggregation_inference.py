"""
aggregation_inference.py  (review round 1, R1-M3 / R1-M6)
=========================================================

Re-tests "single-city sources outperform multi-city aggregates" (Results:
Mann-Whitney U on pooled source-target pairs) with inference that respects the
grouping by target city.

QAP does not apply (aggregates do not form a complete source x target grid), so:
  1. per-target means: for each target, average the outcome over its single-city
     sources and over each aggregate type, giving 7 paired differences.
     Exact Wilcoxon signed-rank test on those (two-sided min p = 1/64).
  2. stratified permutation: shuffle source-type labels within each target only;
     statistic = mean of per-target differences. Exact up to Monte Carlo error.
  3. crossed random effects: outcome ~ source_type, random intercepts for
     target and for source model.
  4. leave-one-target-out range of the per-target mean difference.

Outcomes (seed-averaged, n = 5,000):
  improvement  finetuned Spearman - zero-shot Spearman (the paper's metric)
  final        finetuned Spearman (what a practitioner ends up with)
  zero_shot    zero-shot Spearman

Uses the random-sampling runs only: they are free of the informed-sampling
label leakage (R1-M1) and need no rerun.

Usage:
  python aggregation_inference.py --results-zip ../../results/random__layerwise_lr1e-04.zip \
      --out-dir ../../results/review_stats
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import dyadic_inference as di

CITIES = ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI", "MUM"]
OUTCOMES = ["improvement", "final", "zero_shot"]
CONTRASTS = [("single", "tier"), ("single", "continental"), ("single", "aggregate")]
N = 5000


def source_type(s: str) -> str:
    if s in CITIES:
        return "single"
    if s.startswith("AFR"):
        return "continental"
    if s.startswith(("PRI", "SEC")):
        return "tier"
    raise ValueError(s)


def load(results_zip: str) -> pd.DataFrame:
    zf = zipfile.ZipFile(results_zip)
    d = pd.concat([pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith(".csv")])
    d = d[d["target"].isin(CITIES)]
    zs = d[d["strategy"] == "zero_shot"].groupby(["source", "target"])["spearman"].first()
    ft = (d[(d["strategy"] != "zero_shot") & (d["sample_size"] == N)]
          .groupby(["source", "target"])["spearman"].mean())
    out = pd.DataFrame({"zero_shot": zs, "final": ft}).dropna().reset_index()
    out["improvement"] = out["final"] - out["zero_shot"]
    out["type"] = out["source"].map(source_type)
    return out


def per_target(df: pd.DataFrame, a: str, b: str, y: str) -> pd.Series:
    """Per-target mean(y | type a) - mean(y | type b); 'aggregate' = tier + continental."""
    t = df["type"].where(df["type"] == "single", "aggregate") if b == "aggregate" else df["type"]
    m = df.assign(_t=t).groupby(["target", "_t"])[y].mean().unstack()
    return (m[a] - m[b]).dropna()


def stratified_perm_p(df, a, b, y, n_perm=20000, seed=0) -> float:
    rng = np.random.RandomState(seed)
    t = df["type"].where(df["type"] == "single", "aggregate") if b == "aggregate" else df["type"]
    sub = df.assign(_t=t)[lambda x: x["_t"].isin([a, b])]
    obs = per_target(sub.assign(type=sub["_t"]), a, b, y).mean()
    groups = [g for _, g in sub.groupby("target")]
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = pd.concat([g.assign(type=rng.permutation(g["_t"].to_numpy())) for g in groups])
        null[i] = per_target(shuffled, a, b, y).mean()
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.results_zip)
    print("Source-target pairs by type:", df["type"].value_counts().to_dict(), "\n")

    rows = []
    for y in OUTCOMES:
        mixed = di.crossed_effects(
            df.assign(type=pd.Categorical(df["type"], ["single", "tier", "continental"])),
            f"{y} ~ C(type)")
        for a, b in CONTRASTS:
            diff = per_target(df, a, b, y)
            loto = [diff.drop(t).mean() for t in diff.index]
            mixed_coef = mixed_p = np.nan
            if b != "aggregate":
                key = f"C(type)[T.{b}]"
                mixed_coef, mixed_p = -mixed.params[key], mixed.pvalues[key]
            rows.append({
                "outcome": y, "contrast": f"{a} - {b}",
                "mean_a": df.loc[df["type"] == a, y].mean(),
                "mean_b": df.loc[df["type"] != "single", y].mean() if b == "aggregate"
                          else df.loc[df["type"] == b, y].mean(),
                "per_target_diff": diff.mean(), "targets_a_better": f"{(diff > 0).sum()}/{len(diff)}",
                "p_wilcoxon_exact": wilcoxon(diff, alternative="two-sided", method="exact").pvalue,
                "p_strat_perm": stratified_perm_p(df, a, b, y, n_perm=args.n_perm),
                "mixed_diff": mixed_coef, "p_mixed": mixed_p,
                "loto_min": min(loto), "loto_max": max(loto)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))

    detail = pd.concat({f"{a}-{b}|{y}": per_target(df, a, b, y)
                        for y in OUTCOMES for a, b in CONTRASTS}, axis=1)
    print("\nPer-target differences:\n", detail.round(3).to_string())
    out.to_csv(os.path.join(args.out_dir, "aggregation_inference.csv"), index=False)
    detail.to_csv(os.path.join(args.out_dir, "aggregation_per_target.csv"))
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--n-perm", type=int, default=20000)
    main(ap.parse_args())
