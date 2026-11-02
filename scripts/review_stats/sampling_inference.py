"""
sampling_inference.py  (review round 1: R1-M1, R3-m3, R1-m4)
============================================================

Leak-free sampling rerun (sampling_v3): does SAT-informed selection of target
annotations beat random selection, and how much of the oracle's advantage
survives when only already-collected labels are used?

Per (source, target, sample size) the five seeds are averaged first, so each
directed pair contributes one paired difference strategy - random. Pairs share
cities, so uncertainty comes from the city bootstrap and a crossed
source/target random-intercept model (dyadic_inference.py), not from treating
the 42 pairs as independent.

The smoke test left 8 duplicate ACC->TEM rows (seed 42, n = 100/500); the later
full-run row is kept. The duplicates repeat the same selection, so their spread
measures training nondeterminism alone and is reported as a noise floor.

  python sampling_inference.py --results-zip ../../results/sampling_v3.zip --out-dir ../../results/review_stats

Writes sampling_contrasts.csv, sampling_per_pair.csv, sampling_noise.csv.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from dyadic_inference import city_bootstrap, crossed_effects

KEY = ["source", "target", "sampling", "random_seed", "sample_size"]
OUTCOMES = ["spearman", "accuracy"]
BASELINE = "random"


def load_runs(path: str):
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.endswith("_sampling_v3.csv")]
    raw = pd.concat([pd.read_csv(io.BytesIO(z.read(n))) for n in names], ignore_index=True)
    dup = raw[raw.duplicated(KEY, keep=False)].sort_values(KEY)
    return raw.drop_duplicates(KEY, keep="last").reset_index(drop=True), dup


def noise_floor(dup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in dup.groupby(KEY):
        same_selection = g[["sel_consensus", "sel_cultural", "sel_relational",
                            "sel_structural", "n_unique_locations"]].nunique().max() == 1
        rows.append({**dict(zip(KEY, k)), "same_selection": same_selection,
                     **{f"abs_diff_{o}": float(np.ptp(g[o])) for o in OUTCOMES}})
    return pd.DataFrame(rows)


def contrasts(runs: pd.DataFrame, n_boot: int):
    pair_means = (runs.groupby(["source", "target", "sample_size", "sampling"])[OUTCOMES]
                  .mean().reset_index())
    wide = pair_means.pivot_table(index=["source", "target", "sample_size"],
                                  columns="sampling", values=OUTCOMES)
    rows, per_pair = [], []
    for outcome in OUTCOMES:
        for strat in [s for s in wide[outcome].columns if s != BASELINE]:
            diff = (wide[(outcome, strat)] - wide[(outcome, BASELINE)]).rename("diff").reset_index()
            diff["outcome"], diff["sampling"] = outcome, strat
            per_pair.append(diff)
            for n, g in diff.groupby("sample_size"):
                g = g.reset_index(drop=True)
                stat = lambda df: {"mean_diff": df["diff"].mean(), "median_diff": df["diff"].median()}
                boot = city_bootstrap(g, stat, n_boot=n_boot)
                fit = crossed_effects(g, "diff ~ 1")
                rows.append({
                    "outcome": outcome, "sampling": strat, "vs": BASELINE, "sample_size": n,
                    "n_pairs": len(g),
                    "mean_diff": boot.loc["mean_diff", "est"],
                    "boot_ci_low": boot.loc["mean_diff", "ci_low"],
                    "boot_ci_high": boot.loc["mean_diff", "ci_high"],
                    "median_diff": boot.loc["median_diff", "est"],
                    "share_pairs_better": float((g["diff"] > 0).mean()),
                    "mixed_intercept": float(fit.params["Intercept"]),
                    "mixed_se": float(fit.bse["Intercept"]),
                    "mixed_p": float(fit.pvalues["Intercept"]),
                })
    return pd.DataFrame(rows), pd.concat(per_pair, ignore_index=True)


def seed_spread(runs: pd.DataFrame) -> pd.DataFrame:
    """SD over the five seeds within each pair x size x strategy, summarised."""
    sd = runs.groupby(["source", "target", "sample_size", "sampling"])["spearman"].std()
    return sd.groupby(["sample_size", "sampling"]).median().rename("median_seed_sd").reset_index()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    runs, dup = load_runs(args.results_zip)
    counts = runs.groupby(["source", "target"]).size()
    assert (counts == 80).all() and len(counts) == 42, "incomplete rerun"

    noise = noise_floor(dup)
    table, per_pair = contrasts(runs, args.n_boot)
    spread = seed_spread(runs)

    table.to_csv(os.path.join(args.out_dir, "sampling_contrasts.csv"), index=False)
    per_pair.to_csv(os.path.join(args.out_dir, "sampling_per_pair.csv"), index=False)
    noise.to_csv(os.path.join(args.out_dir, "sampling_noise.csv"), index=False)

    pd.set_option("display.width", 250)
    print("Duplicate runs (same seed, same settings):")
    print(noise.round(3).to_string(index=False))
    print("\nMedian SD of Spearman over seeds:")
    print(spread.pivot(index="sample_size", columns="sampling", values="median_seed_sd").round(3))
    print("\nStrategy - random, seed-averaged per pair:")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args())
