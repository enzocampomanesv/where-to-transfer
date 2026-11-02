"""
dataset_summary.py  (review round 1: R1-M5, R2-5, R1-m3)
=========================================================

Per-city description of the comparison data, computed from the vote databases
exactly as the study split them (finetune_data.reconstruct_split). Run on CRIB
from scripts/finetune_experiments/ (it imports finetune_data from there):

  cd scripts/finetune_experiments
  PYTHONNOUSERSITE=1 python ../review_stats/dataset_summary.py --out-dir <dir>

Writes
  dataset_summary.csv      one row per city: comparisons, locations, participants,
                           train/test split and how locations are shared across it
  finetune_locations.csv   unique locations in the random finetune sample per
                           city x sample size x seed (R1-m3)
  vote_columns.txt         column names of each votes table (to check the
                           participant-id column guess below)

No model, TensorFlow or imagery is needed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune_experiments"))
from finetune_data import load_pairs, reconstruct_split, random_sample  # noqa: E402

CITIES = ["ACC", "TEM", "NBO", "KIS", "BEI", "CHI", "MUM"]
SIZES = [100, 500, 1000, 5000]
SEEDS = [42, 43, 44, 45, 46]
# first matching column is treated as the participant identifier
PARTICIPANT_COLS = ["voter_id", "user_id", "participant_id", "participant", "user", "voter",
                    "session_id", "session", "device_id"]


def _ids(df):
    dec = lambda s: s.decode("utf-8") if isinstance(s, bytes) else str(s)
    return df["left"].map(dec), df["right"].map(dec)


def locations(df) -> set:
    l, r = _ids(df)
    return set(l) | set(r)


def per_location_counts(df) -> pd.Series:
    l, r = _ids(df)
    return pd.concat([l, r]).value_counts()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rows, ft_rows, cols_log = [], [], []
    for city in CITIES:
        data = load_pairs(city, args.voter, args.root_path)
        cols_log.append(f"{city}: {list(data.columns)}")
        train, test = reconstruct_split(city, args.voter, args.root_path)
        locs_all, locs_tr, locs_te = locations(data), locations(train), locations(test)
        per_loc = per_location_counts(data)
        per_loc_te = per_location_counts(test)

        pcol = next((c for c in PARTICIPANT_COLS if c in data.columns), None)
        if pcol:
            per_part = data[pcol].value_counts()
            part = {"participant_column": pcol, "participants": int(per_part.size),
                    "comparisons_per_participant_median": float(per_part.median()),
                    "comparisons_per_participant_iqr": f"{per_part.quantile(.25):.0f}-{per_part.quantile(.75):.0f}"}
        else:
            part = {"participant_column": None, "participants": None,
                    "comparisons_per_participant_median": None,
                    "comparisons_per_participant_iqr": None}

        l, r = _ids(data)
        pair_key = pd.Series([tuple(sorted(p)) for p in zip(l, r)])
        rows.append({
            "city": city, "comparisons": len(data),
            "unique_locations": len(locs_all),
            "comparisons_per_location_median": float(per_loc.median()),
            "distinct_location_pairs": int(pair_key.nunique()),
            "repeated_pair_share": float(1 - pair_key.nunique() / len(pair_key)),
            "left_wins_share": float(pd.to_numeric(data["y"]).mean()),
            **part,
            "train_pool_comparisons": len(train), "test_comparisons": len(test),
            "train_locations": len(locs_tr), "test_locations": len(locs_te),
            "test_locations_also_in_train": len(locs_te & locs_tr),
            "test_locations_only_in_test": len(locs_te - locs_tr),
            "test_comparisons_per_location_median": float(per_loc_te.median()),
        })

        for n in SIZES:
            if n >= len(train):
                continue
            for seed in SEEDS:
                s = random_sample(train, n, seed)
                ls = locations(s)
                ft_rows.append({"city": city, "sample_size": n, "seed": seed,
                                "unique_locations": len(ls),
                                "share_of_test_locations_covered": len(ls & locs_te) / len(locs_te)})
        print(f"{city}: {len(data)} comparisons, {len(locs_all)} locations, participants={part['participants']}")

    summary = pd.DataFrame(rows)
    ft = pd.DataFrame(ft_rows)
    summary.to_csv(os.path.join(args.out_dir, "dataset_summary.csv"), index=False)
    ft.to_csv(os.path.join(args.out_dir, "finetune_locations.csv"), index=False)
    with open(os.path.join(args.out_dir, "vote_columns.txt"), "w") as f:
        f.write("\n".join(cols_log))
    pd.set_option("display.width", 250)
    print("\n", summary.T.to_string())
    print("\nUnique locations in the random finetune sample (mean over seeds):")
    print(ft.groupby(["city", "sample_size"])["unique_locations"].mean().unstack().round(0).to_string())
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/review_stats"))
    main(ap.parse_args())
