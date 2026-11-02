"""
test_gate.py
============

Validates gate.py against the REAL uploaded metrics CSVs plus synthetic edge cases:
band construction from three seeds, the tol=max(2*sd, 0.02) rule, per-metric and
overall pass/fail, zero-shot lookup, and missing-cell -> None (hard error upstream).

Run:  python test_gate.py
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd

import gate

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

METRICS_DIR = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning")   # contains informed_t1/ and random/


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def test_load_real_metrics():
    print("Test 1: load real metrics CSVs")
    df = gate.load_metrics(METRICS_DIR)
    ok = True
    ok &= check("loaded rows", len(df) > 0, f"{len(df)} rows")
    ok &= check("has key + metric columns",
                {"source", "target", "sample_size", "sampling",
                 "spearman", "accuracy"}.issubset(df.columns))
    ok &= check("both arms present",
                set(df["sampling"].dropna().unique()) >= {"random", "informed_t1"},
                f"{sorted(df['sampling'].dropna().unique())}")
    return ok, df


def test_band_from_real(df):
    print("Test 2: three-seed band for a real cell (PRI_NBO->TEM, 5000, random)")
    band = gate.band_for(df, "PRI_NBO", "TEM", 5000, "random")
    ok = True
    ok &= check("band found", band is not None)
    if band:
        ok &= check("spearman band has 3 seeds", band["spearman"]["n"] == 3,
                    f"n={band['spearman']['n']}")
        ok &= check("min <= mean <= max (spearman)",
                    band["spearman"]["min"] <= band["spearman"]["mean"] <= band["spearman"]["max"])
        ok &= check("accuracy band present", "accuracy" in band)
    return ok


def test_informed_alias(df):
    print("Test 3: 'informed' alias resolves to informed_t1")
    # find any informed_t1 cell that exists
    inf = df[(df["sampling"] == "informed_t1") & (df["sample_size"] == 5000)]
    ok = True
    if len(inf):
        s, t = inf.iloc[0]["source"], inf.iloc[0]["target"]
        b1 = gate.band_for(df, s, t, 5000, "informed")
        b2 = gate.band_for(df, s, t, 5000, "informed_t1")
        ok &= check("alias == explicit", b1 == b2, f"{s}->{t}")
    else:
        ok &= check("informed_t1 cells exist", False)
    return ok


def test_pass_within_band():
    print("Test 4: refit inside the band passes")
    band = {"spearman": {"min": 0.66, "max": 0.71, "mean": 0.68, "std": 0.02, "n": 3},
            "accuracy": {"min": 0.67, "max": 0.69, "mean": 0.68, "std": 0.01, "n": 3}}
    res = gate.evaluate_gate({"spearman": 0.69, "accuracy": 0.68}, band)
    return check("passes when both within band", res["passed"], str(res["passed"]))


def test_fail_outside_band():
    print("Test 5: refit outside the band fails (and reports which metric)")
    band = {"spearman": {"min": 0.66, "max": 0.71, "mean": 0.68, "std": 0.005, "n": 3},
            "accuracy": {"min": 0.67, "max": 0.69, "mean": 0.68, "std": 0.005, "n": 3}}
    # spearman 0.55 is far below min-tol (tol=max(2*0.005,0.02)=0.02 -> lo=0.64)
    res = gate.evaluate_gate({"spearman": 0.55, "accuracy": 0.68}, band)
    ok = True
    ok &= check("overall fails", not res["passed"])
    ok &= check("spearman flagged failing", res["metrics"]["spearman"]["passed"] is False)
    ok &= check("accuracy still passes", res["metrics"]["accuracy"]["passed"] is True)
    return ok


def test_tol_floor():
    print("Test 6: tol floor (0.02) applies when seeds agree tightly")
    band = {"spearman": {"min": 0.70, "max": 0.70, "mean": 0.70, "std": 0.0, "n": 3},
            "accuracy": {"min": 0.68, "max": 0.68, "mean": 0.68, "std": 0.0, "n": 3}}
    # with std=0, tol=floor=0.02 -> [0.68,0.72] for spearman
    ok = True
    ok &= check("0.715 passes via floor",
                gate.evaluate_gate({"spearman": 0.715, "accuracy": 0.68}, band)["passed"])
    ok &= check("0.73 fails (outside floor)",
                not gate.evaluate_gate({"spearman": 0.73, "accuracy": 0.68}, band)["passed"])
    return ok


def test_missing_cell(df):
    print("Test 7: missing cell -> None (hard error upstream)")
    band = gate.band_for(df, "NOPE", "ZZZ", 5000, "random")
    return check("returns None for unknown cell", band is None)


def main():
    results = []
    df = None
    try:
        ok, df = test_load_real_metrics()
        results.append(ok)
    except Exception:
        traceback.print_exc(); results.append(False)
    print()
    tests_needing_df = [test_band_from_real, test_informed_alias, test_missing_cell]
    standalone = [test_pass_within_band, test_fail_outside_band, test_tol_floor]
    for t in [test_band_from_real, test_informed_alias]:
        try:
            results.append(t(df))
        except Exception:
            traceback.print_exc(); results.append(False)
        print()
    for t in standalone:
        try:
            results.append(t())
        except Exception:
            traceback.print_exc(); results.append(False)
        print()
    try:
        results.append(test_missing_cell(df))
    except Exception:
        traceback.print_exc(); results.append(False)
    print()

    n = sum(results)
    print("=" * 60)
    print(f"RESULT: {n}/{len(results)} test groups passed")
    print("=" * 60)
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
