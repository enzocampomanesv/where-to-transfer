"""
train_from_scratch.py  (review round 1: R3-4, transfer vs direct local training)
===============================================================================

Is transfer worth it? For every target city, the source-model architecture is
trained from scratch (fresh random weights, no source city) on

  * the same n = 500 / 1,000 / 5,000 random target comparisons the transfer runs
    were finetuned on (random_sample(train_pool, n, seed), seeds 42-44), and
  * the whole training pool (the full-local reference),

and evaluated on the same held-out 30% test split with the same metrics as
run_sampling_experiment.py. Compare with the rerun's random-sampling rows:
source model + finetuning on n vs scratch on n, and both vs the full reference.

Recipe = the city models' own (where_to_transfer/ACC_train_bradleyTerry_ensemble.py):
Adam 1e-3 (the step schedule there never fires, so constant), Huber loss, batch
64, up to 1,000 epochs, early stopping on validation accuracy with patience 50,
6.25% of the n comparisons held out for validation (so the label budget is n).
A single model is trained; see ensemble_check.json for whether the saved city
"ensembles" contain distinct members.

The architecture is cloned from a saved city model (fresh weights), so it is
exactly the source models' architecture.

Run on CRIB from scripts/finetune_experiments/:

  PYTHONNOUSERSITE=1 python train_from_scratch.py

Writes {out-dir}/from_scratch.csv (resumable) and {out-dir}/ensemble_check.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd

from finetune_data import random_sample, reconstruct_split
from run_sampling_experiment import _path

CITIES = ["ACC", "TEM", "NBO", "KIS", "BEI", "CHI", "MUM"]


def ensemble_check(model):
    """How many distinct member models the saved city ensemble holds.

    The training script appends the same model object for every fold, so a single
    member layer (called five times) is expected; five layers with distinct
    weights would mean a genuine ensemble."""
    import tensorflow as tf
    members = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    w = [np.concatenate([v.ravel() for v in m.get_weights()]) for m in members]
    distinct = []
    for i, wi in enumerate(w):
        if not any(np.array_equal(wi, w[j]) for j in distinct):
            distinct.append(i)
    return {"n_member_layers": len(members), "n_unique_objects": len({id(m) for m in members}),
            "n_distinct_weight_sets": len(distinct)}


def image_arrays(df, root_path, sub_bands, preprocess_fn, cache):
    """Left/right image stacks for df; each location is read once."""
    for p in set(map(_path, df["left"])) | set(map(_path, df["right"])):
        if p not in cache:
            cache[p] = preprocess_fn(os.path.join(root_path, p), sub_bands)
    left = np.stack([cache[_path(p)] for p in df["left"]])
    right = np.stack([cache[_path(p)] for p in df["right"]])
    return [left, right], np.asarray(df["y"], dtype=np.float32).ravel()


def train_one(template, train_df, test_df, seed, args, sub_bands, preprocess_fn, cache, bt_recovery_fn):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = tf.keras.models.clone_model(template)          # same architecture, fresh weights
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber_loss",
                  metrics=["binary_accuracy"])
    fit_df, val_df = train_test_split(train_df, test_size=0.0625, random_state=42)
    x_fit, y_fit = image_arrays(fit_df, args.root_path, sub_bands, preprocess_fn, cache)
    x_val, y_val = image_arrays(val_df, args.root_path, sub_bands, preprocess_fn, cache)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", mode="max", patience=50,
                                            restore_best_weights=True)
    hist = model.fit(x_fit, y_fit, validation_data=(x_val, y_val), epochs=args.max_epochs,
                     batch_size=64, callbacks=[stop], verbose=0)

    x_test, y_test = image_arrays(test_df, args.root_path, sub_bands, preprocess_fn, cache)
    probs = model.predict(x_test, batch_size=256, verbose=0).ravel()
    bt = bt_recovery_fn(test_df, probs)
    out = {"accuracy": float(((probs > 0.5).astype(int) == y_test.astype(int)).mean()),
           "spearman": bt.get("spearman"), "kendall": bt.get("kendall"),
           "bt_n_locations": bt.get("n_locations"),
           "epochs_trained": len(hist.history["loss"]),
           "n_fit": len(fit_df), "n_val": len(val_df)}
    del model
    gc.collect()
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="From-scratch baselines at the transfer budgets")
    ap.add_argument("--targets", default=",".join(CITIES))
    ap.add_argument("--sizes", default="500,1000,5000,full")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--template-source", default="ACC", help="city model whose architecture is cloned")
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/from_scratch"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    import tensorflow as tf
    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands
    from BT_utils.evalBT import evaluate_model_bt_recovery

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "from_scratch.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    # architecture template + ensemble check on every city model
    checks = {}
    for src in CITIES:
        path = (f"{args.root_path}/trained_models/voter/"
                f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
        ens = tf.keras.models.load_model(path)
        checks[src] = ensemble_check(ens)
        if src == args.template_source:
            template = ens.layers[2]          # one member: the single Bradley-Terry model (kept loaded)
    print("ensemble check:", checks)
    with open(os.path.join(args.out_dir, "ensemble_check.json"), "w") as f:
        json.dump(checks, f, indent=2)

    for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
        train_pool, test_data = reconstruct_split(tgt, args.voter, args.root_path)
        cache = {}
        for size in [s.strip() for s in args.sizes.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                if len(done) and ((done["target"] == tgt) & (done["sample_size"].astype(str) == size)
                                  & (done["random_seed"] == seed)).any():
                    print(f"  skip {tgt} n={size} seed={seed} (done)")
                    continue
                train_df = train_pool if size == "full" else random_sample(train_pool, int(size), seed)
                t0 = time.time()
                m = train_one(template, train_df, test_data, seed, args, sub_bands, preprocess_fn,
                              cache, evaluate_model_bt_recovery)
                row = {"target": tgt, "sample_size": size, "random_seed": seed, **m,
                       "duration_seconds": round(time.time() - t0, 1)}
                print(f"  {tgt} n={size:>5} seed={seed}  sp={m['spearman']:.3f}  acc={m['accuracy']:.3f}  "
                      f"epochs={m['epochs_trained']}  ({row['duration_seconds']}s)")
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")


if __name__ == "__main__":
    main()
