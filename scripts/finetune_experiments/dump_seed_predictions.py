"""
dump_seed_predictions.py  (review round 1: decision 2, seed-selected aggregates)
================================================================================

Zero-shot predictions of every source model (single cities and multi-city
aggregates) on each target's seed batches, so that the seed-accuracy rule can
choose among aggregates exactly as it chooses among single cities. No
finetuning.

A seed batch is fp.seed_batch(train_pool, n, seed), i.e. the random strategy's
sample at that n and seed; it depends only on the target's pool and the seed, so
every source is scored on the same comparisons. Single cities are included as a
check: their seed accuracy at n = 100 and 500 must equal the rerun's logged
lab_source_accuracy.

Run on CRIB from scripts/finetune_experiments/:

  PYTHONNOUSERSITE=1 python dump_seed_predictions.py

Writes {out-dir}/seed_predictions.csv.gz with one row per source, target and
pool position in the union of the seed batches:
  source, target, pos, y, src_prob, in_batches (e.g. "100:42;500:42;500:45")
"""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import pandas as pd

import finetune_prospective as fp
from finetune_data import reconstruct_split
from run_sampling_experiment import _SRC_TRAIN

CITIES = ["ACC", "TEM", "NBO", "KIS", "BEI", "CHI", "MUM"]
SOURCES = CITIES + ["PRI", "SEC", "AFR", "PRI_ACC", "PRI_NBO", "PRI_BEI", "SEC_TEM", "SEC_KIS",
                    "SEC_CHI", "AFR_ACC", "AFR_NBO", "AFR_BEI", "AFR_TEM", "AFR_KIS", "AFR_CHI"]


def seed_rows(train_pool, sizes, seeds):
    """Pool positions in any seed batch, with the batches each belongs to."""
    member = {}
    for n in sizes:
        for s in seeds:
            for pos in fp.seed_batch(train_pool, n, s):
                member.setdefault(int(pos), []).append(f"{n}:{s}")
    pos = np.array(sorted(member))
    return pos, [";".join(member[p]) for p in pos]


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Zero-shot predictions of all sources on the seed batches")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--sizes", default="100,500")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/seed_predictions"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    import tensorflow as tf
    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands
    from finetune_core import create_dataset, evaluate_model

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "seed_predictions.csv.gz")

    # seed batches per target (same for every source)
    batches = {}
    for tgt in CITIES:
        train_pool, _ = reconstruct_split(tgt, args.voter, args.root_path)
        train_pool = train_pool.reset_index(drop=True)
        pos, member = seed_rows(train_pool, sizes, seeds)
        batches[tgt] = (train_pool.iloc[pos].reset_index(drop=True), pos, member)
        print(f"{tgt}: {len(pos)} pool comparisons in the seed batches")

    parts = []
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        model_path = (f"{args.root_path}/trained_models/voter/"
                      f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        for tgt in [c for c in CITIES if c not in _SRC_TRAIN.get(src, {src})]:
            df, pos, member = batches[tgt]
            _, probs = evaluate_model(model, create_dataset(df, args.root_path, sub_bands, preprocess_fn))
            parts.append(pd.DataFrame({"source": src, "target": tgt, "pos": pos,
                                       "y": np.asarray(df["y"], dtype=int).ravel(),
                                       "src_prob": np.asarray(probs).ravel(), "in_batches": member}))
            print(f"  {src} -> {tgt}: done")
        del model
        gc.collect()
        pd.concat(parts, ignore_index=True).to_csv(out_csv, index=False)   # save as we go
    print(f"\nResults -> {out_csv}")


if __name__ == "__main__":
    main()
