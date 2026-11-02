"""
alr_inference.py  (review round 1, R1-M3)
=========================================

Re-tests the additive log-ratio regression of zero-shot Spearman on SAT
category shares (Results: CD coefficient -0.91, R2 = 0.95) with inference that
accounts for repeated cities:

  naive OLS p-values (as in the submitted paper)
  exact MRQAP p-values
  city-bootstrap 95% intervals
  leave-one-city-out coefficient range
  crossed source/target random-effects model

Zero-shot rows are identical across sampling runs, so any results archive works.

Usage:
  python alr_inference.py --results-zip ../../results/random__layerwise_lr1e-04.zip \
      --out-dir ../../results/review_stats
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd

import dyadic_inference as di

CITIES = ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI", "MUM"]
SHARES = ["consensus", "cultural_disagreement", "relational_ignorance", "structural_ignorance"]
TERMS = {"alr_cd": "cultural_disagreement", "alr_ri": "relational_ignorance",
         "alr_si": "structural_ignorance"}
ZERO_FILL = 1e-6   # "negligible positive value" for zero shares before the log-ratio


def load_zero_shot(results_zip: str) -> pd.DataFrame:
    zf = zipfile.ZipFile(results_zip)
    d = pd.concat([pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()
                   if n.endswith(".csv")])
    z = d[(d["strategy"] == "zero_shot") & d["source"].isin(CITIES) & d["target"].isin(CITIES)]
    return z.groupby(["source", "target"], as_index=False).first()[["source", "target", "spearman", *SHARES]]


def add_alr(df: pd.DataFrame, zero_fill: float = ZERO_FILL) -> pd.DataFrame:
    s = df[SHARES].clip(lower=zero_fill)
    s = s.div(s.sum(axis=1), axis=0)
    out = df.copy()
    for term, share in TERMS.items():
        out[term] = np.log(s[share] / s["consensus"])
    return out


def alr_coefs(df: pd.DataFrame) -> dict:
    beta, _, r2 = di.ols(di._design(df, list(TERMS)), df["spearman"].to_numpy(float))
    return {**{t: b for t, b in zip(TERMS, beta[1:])}, "r2": r2}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = add_alr(load_zero_shot(args.results_zip), args.zero_fill)
    print(f"{len(df)} source-target pairs, {df['source'].nunique()} cities\n")

    # naive OLS (paper) vs exact MRQAP
    import statsmodels.api as sm
    naive = sm.OLS(df["spearman"], sm.add_constant(df[list(TERMS)])).fit()
    mrqap = di.mrqap_dsp(df, "spearman", list(TERMS))
    mrqap["p_naive"] = naive.pvalues[list(TERMS)].values

    boot = di.city_bootstrap(df, alr_coefs, n_boot=args.n_boot, seed=0)
    loco = di.leave_one_city_out(df, alr_coefs)
    mixed = di.crossed_effects(df, "spearman ~ " + " + ".join(TERMS))

    table = mrqap.set_index("term").join(boot[["ci_low", "ci_high"]])
    table["loco_min"] = loco[list(TERMS)].min().values
    table["loco_max"] = loco[list(TERMS)].max().values
    table["coef_mixed"] = mixed.params[list(TERMS)].values
    table["p_mixed"] = mixed.pvalues[list(TERMS)].values

    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(f"\nR2 = {mrqap.attrs['r2']:.3f}  (city-bootstrap 95% CI "
          f"{boot.loc['r2', 'ci_low']:.3f}-{boot.loc['r2', 'ci_high']:.3f})")
    print(f"Random-effect variances: source={mixed.vcomp[0]:.5f}, target={mixed.vcomp[1]:.5f}, "
          f"residual={mixed.scale:.5f}")
    print("\nLeave-one-city-out:")
    print(loco.round(3).to_string(index=False))

    table.to_csv(os.path.join(args.out_dir, "alr_inference.csv"))
    loco.to_csv(os.path.join(args.out_dir, "alr_loco.csv"), index=False)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--zero-fill", type=float, default=ZERO_FILL)
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args())
