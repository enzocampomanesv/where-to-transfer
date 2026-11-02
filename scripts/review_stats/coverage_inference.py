"""
coverage_inference.py  (review round 1, R1-M3)
==============================================

Re-tests the Tier 1 validation correlations (Results: coverage vs zero-shot
rho = -0.06; coverage vs finetuning improvement rho = 0.31, p = 0.04) with exact
QAP p-values and city-bootstrap intervals, for every Tier 1 measure and both
sampling runs. Reads the merged tables written by t1_validation.py.

Usage:
  python coverage_inference.py --validation-zip ../../results/tier1/validation.zip \
      --out-dir ../../results/review_stats
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import pandas as pd
from scipy.stats import spearmanr

import dyadic_inference as di

MEASURES = {"cat_code": "Coverage category (ordinal, 0 = strong)",
            "overlap": "Overlap ratio (avg k10/k20)",
            "mmd": "MMD (RBF kernel)"}
TABLES = {"zeroshot_rho": ("validation/validation_merged_zeroshot.csv", "zeroshot_rho"),
          "improvement_random": ("validation/validation_merged_ft_random.csv", "improvement"),
          "improvement_informed_t1": ("validation/validation_merged_ft_informed_t1.csv", "improvement")}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    zf = zipfile.ZipFile(args.validation_zip)
    rows = []
    for outcome, (member, ycol) in TABLES.items():
        df = pd.read_csv(io.BytesIO(zf.read(member))).rename(
            columns={"source_city": "source", "target_city": "target"})
        for x, label in MEASURES.items():
            q = di.qap_spearman(df, x, ycol)
            boot = di.city_bootstrap(
                df, lambda d, x=x: {"rho": spearmanr(d[x], d[ycol]).correlation},
                n_boot=args.n_boot, seed=0)
            no_mum = df[(df["source"] != "MUM") & (df["target"] != "MUM")]
            q_nm = di.qap_spearman(no_mum, x, ycol)
            rows.append({"outcome": outcome, "measure": label, "rho": q["rho"],
                         "p_naive": q["p_naive"], "p_qap": q["p_qap"],
                         "ci_low": boot.loc["rho", "ci_low"], "ci_high": boot.loc["rho", "ci_high"],
                         "rho_no_MUM": q_nm["rho"], "p_qap_no_MUM": q_nm["p_qap"]})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(args.out_dir, "coverage_inference.csv"), index=False)

    if args.results_zip:
        construct = coverage_construct(zf, args.results_zip)
        print("\nWhat coverage measures (zero-shot SAT shares on the test set):")
        print(construct.round(3).to_string(index=False))
        construct.to_csv(os.path.join(args.out_dir, "coverage_construct.csv"), index=False)
    print(f"\n-> {args.out_dir}")


def coverage_construct(validation_zf, results_zip) -> pd.DataFrame:
    """Does coverage track covariate shift (Structural Ignorance) and is it separate
    from concept shift (CD rate among in-domain pairs)?

    CD is also expressed as a rate within in-domain pairs, because its share of all
    pairs rises mechanically when fewer pairs are Structural Ignorance.
    """
    from alr_inference import load_zero_shot
    df = pd.read_csv(io.BytesIO(validation_zf.read(TABLES["zeroshot_rho"][0]))).rename(
        columns={"source_city": "source", "target_city": "target"})
    df = df.merge(load_zero_shot(results_zip).drop(columns="spearman"), on=["source", "target"])
    df["in_domain_share"] = df["consensus"] + df["cultural_disagreement"]
    df["cd_rate_in_domain"] = df["cultural_disagreement"] / df["in_domain_share"]
    outcomes = ["structural_ignorance", "relational_ignorance", "in_domain_share",
                "cultural_disagreement", "cd_rate_in_domain"]
    rows = []
    for x, label in MEASURES.items():
        for y in outcomes:
            q = di.qap_spearman(df, x, y)
            rows.append({"measure": label, "vs": y, "rho": q["rho"], "p_qap": q["p_qap"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-zip", default="../../results/tier1/validation.zip")
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip",
                    help="any results archive (zero-shot rows) for the construct check")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args())
