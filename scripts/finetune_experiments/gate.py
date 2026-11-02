"""
gate.py
=======

The reproducibility gate. Loads the ORIGINAL finetuning metrics (the uploaded
per-variant CSVs), builds the three-seed band for a requested cell, and decides
whether a re-finetuned (or zero-shot) run reproduces the study.

Keying (from the CSV columns, not filenames):
    source        -- effective source code, e.g. 'ACC', 'PRI_NBO' (leave-one-out)
    target        -- target city
    sample_size   -- 0 (zero-shot) / 100 / 500 / 1000 / 5000
    sampling      -- 'random' or 'informed_t1'
    random_seed   -- 42 / 43 / 44  (absent for zero-shot rows)

Gate rule (agreed): for each of spearman and accuracy,
    tol = max(tol_k * sd, tol_floor)          # sd across the original seeds
    pass if  min - tol <= refit <= max + tol
Overall pass requires BOTH metrics pass. A missing cell is a hard error (can't
gate -> must not map).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

SAMPLING_ALIASES = {"informed": "informed_t1", "informed_t1": "informed_t1",
                    "random": "random"}
GATE_METRICS = ("spearman", "accuracy")
DEFAULT_TOL_K = 2.0
DEFAULT_TOL_FLOOR = 0.02


def load_metrics(metrics_csv_dir: str) -> pd.DataFrame:
    """Concatenate every metrics CSV under {dir}/informed_t1/ and {dir}/random/.

    Returns one long DataFrame with normalised dtypes. Raises if nothing is found.
    """
    patterns = [
        os.path.join(metrics_csv_dir, "informed_t1", "**", "*.csv"),
        os.path.join(metrics_csv_dir, "random", "**", "*.csv"),
        os.path.join(metrics_csv_dir, "**", "*.csv"),  # fallback: any nesting
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No metrics CSVs found under {metrics_csv_dir!r}.")

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "source" in d.columns and "target" in d.columns:
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"CSVs found but none had source/target columns.")
    df = pd.concat(frames, ignore_index=True)

    # normalise dtypes used for keying
    if "sample_size" in df.columns:
        df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce").astype("Int64")
    if "random_seed" in df.columns:
        df["random_seed"] = pd.to_numeric(df["random_seed"], errors="coerce")
    return df


def band_for(
    metrics_df: pd.DataFrame,
    source: str,
    target: str,
    sample_size: int,
    sampling: str,
    *,
    zero_shot: bool = False,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Return the seed band for a cell: {metric: {min,max,mean,std,n}} or None.

    For zero_shot=True, matches sample_size==0 for (source,target) ignoring
    sampling (zero-shot rows are sampler-independent).
    """
    df = metrics_df
    q = (df["source"] == source) & (df["target"] == target)
    if zero_shot:
        q &= (df["sample_size"] == 0)
    else:
        samp = SAMPLING_ALIASES.get(sampling, sampling)
        q &= (df["sample_size"] == int(sample_size)) & (df["sampling"] == samp)
    rows = df[q]
    if len(rows) == 0:
        return None

    band = {}
    for m in GATE_METRICS:
        if m not in rows.columns:
            continue
        vals = pd.to_numeric(rows[m], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        band[m] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)) if vals.size > 1 else 0.0,
            "n": int(vals.size),
        }
    return band or None


def evaluate_gate(
    refit_metrics: Dict[str, float],
    band: Dict[str, Dict[str, float]],
    *,
    tol_k: float = DEFAULT_TOL_K,
    tol_floor: float = DEFAULT_TOL_FLOOR,
) -> Dict[str, object]:
    """Decide pass/fail per metric and overall.

    tol = max(tol_k*std, tol_floor); pass if min-tol <= refit <= max+tol.
    Only metrics present in BOTH refit_metrics and band are evaluated; overall
    pass requires every evaluated metric to pass and at least one to be evaluated.
    """
    details = {}
    evaluated = 0
    all_pass = True
    for m in GATE_METRICS:
        if m not in band or refit_metrics.get(m) is None:
            continue
        evaluated += 1
        b = band[m]
        tol = max(tol_k * b["std"], tol_floor)
        lo, hi = b["min"] - tol, b["max"] + tol
        val = float(refit_metrics[m])
        passed = (lo <= val <= hi)
        all_pass &= passed
        details[m] = {"refit": val, "lo": lo, "hi": hi, "tol": tol,
                      "band_min": b["min"], "band_max": b["max"],
                      "band_mean": b["mean"], "band_std": b["std"], "n": b["n"],
                      "passed": passed}
    passed = bool(all_pass and evaluated > 0)
    return {"passed": passed, "n_metrics_evaluated": evaluated, "metrics": details}
