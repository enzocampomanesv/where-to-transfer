"""
sat_error_rates.py  (review round 1: what the SAT decomposition shows)
======================================================================

Is the source model more often wrong on target comparisons that look unfamiliar
(Structural / Relational Ignorance) than on those that look familiar (Consensus +
Preference Disagreement)? If not, the share of errors falling in PD simply equals
the share of familiar comparisons, and the finding is that transfer fails in the
judgements (concept shift), not in the images (covariate shift).

Computed from iso_sensitivity.py's per-pair category shares, for every
isolation-forest setting. PD is a wrong prediction on a familiar comparison, so
  error rate, familiar   = PD / (Consensus + PD)
  wrong, unfamiliar      = PD * (1 / pd_share_of_errors - 1)

  python sat_error_rates.py --shares-csv ../../results/review_stats/iso_sensitivity_shares.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def error_rates(d: pd.DataFrame, min_unfamiliar: float = 0.02) -> pd.DataFrame:
    d = d.assign(familiar=d["consensus"] + d["cultural_disagreement"])
    d["unfamiliar"] = 1 - d["familiar"]
    d["err_familiar"] = d["cultural_disagreement"] / d["familiar"]
    d["wrong_unfamiliar"] = d["cultural_disagreement"] * (1 / d["pd_share_of_errors"] - 1)
    rows = []
    for (varied, setting), g in d.groupby(["varied", "setting"], sort=False):
        m = g["unfamiliar"] > min_unfamiliar
        rows.append({"varied": varied, "setting": setting,
                     "familiar_share": g["familiar"].mean(),
                     "pd_share_of_errors": g["pd_share_of_errors"].mean(),
                     "error_rate_familiar": g["err_familiar"].mean(),
                     "error_rate_unfamiliar": (g["wrong_unfamiliar"][m] / g["unfamiliar"][m]).mean(),
                     "error_rate_unfamiliar_pooled": g["wrong_unfamiliar"].sum() / g["unfamiliar"].sum(),
                     "n_pairs_unfamiliar": int(m.sum())})
    return pd.DataFrame(rows)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out = error_rates(pd.read_csv(args.shares_csv), args.min_unfamiliar)
    out.to_csv(os.path.join(args.out_dir, "sat_error_rates.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(f"\n(per-pair unfamiliar rate over pairs with > {args.min_unfamiliar:.0%} unfamiliar comparisons)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares-csv", default="../../results/review_stats/iso_sensitivity_shares.csv")
    ap.add_argument("--min-unfamiliar", type=float, default=0.02)
    ap.add_argument("--out-dir", default="../../results/review_stats")
    main(ap.parse_args())
