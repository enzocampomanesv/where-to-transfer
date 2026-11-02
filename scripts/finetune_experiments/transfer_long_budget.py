"""
transfer_long_budget.py  (review round 1: R3-4, transfer with the from-scratch budget)
======================================================================================

At n = 5,000 the from-scratch baseline (train_from_scratch.py) matched transfer, but
the two arms had different training budgets: finetuning stopped after <= 50 epochs
on training loss at LR <= 5e-4, while scratch trained up to 1,000 epochs at 1e-3
with validation early stopping (median 112 epochs). This script reruns transfer on
exactly the scratch arm's comparisons and validation split, with the scratch arm's
budget (batch 64, <= 1,000 epochs, early stopping on validation accuracy with
patience 50, best weights restored), in two recipes:

  layerwise  the study's finetuning (BN frozen, decayed block LRs, BCE; finetune_core).
             The primary arm: the paper's transfer method.
  match      full finetuning: the scratch optimiser verbatim (Adam 1e-3 on all layers, Huber loss),
             started from the source weights, so only the initial weights differ
             from scratch. Shows whether a less conservative transfer does better.

For every target, all six other single-city source models are run, so the
seed-chosen, average and best-in-hindsight arms can be rebuilt as in
scratch_vs_transfer.py.

Run on CRIB from scripts/finetune_experiments/:

  PYTHONNOUSERSITE=1 python transfer_long_budget.py --recipe layerwise
  PYTHONNOUSERSITE=1 python transfer_long_budget.py --recipe match

(run_transfer_long.sh runs the two in this order). Writes
{out-dir}/transfer_long_{recipe}.csv (resumable).
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

from finetune_data import random_sample, reconstruct_split
from train_from_scratch import CITIES, image_arrays


def train_one(model_path, recipe, train_df, test_df, seed, args, sub_bands, preprocess_fn, cache,
              bt_recovery_fn):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from finetune_core import BASE_LR, setup_layerwise_training

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = tf.keras.models.load_model(model_path)          # source weights
    if recipe == "layerwise":
        setup_layerwise_training(model, BASE_LR)             # compiles with metrics=["accuracy"]
        monitor = "val_accuracy"
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber_loss",
                      metrics=["binary_accuracy"])
        monitor = "val_binary_accuracy"
    # same validation split as train_from_scratch.py
    fit_df, val_df = train_test_split(train_df, test_size=0.0625, random_state=42)
    x_fit, y_fit = image_arrays(fit_df, args.root_path, sub_bands, preprocess_fn, cache)
    x_val, y_val = image_arrays(val_df, args.root_path, sub_bands, preprocess_fn, cache)
    x_test, y_test = image_arrays(test_df, args.root_path, sub_bands, preprocess_fn, cache)

    val0 = float(((model.predict(x_val, batch_size=256, verbose=0).ravel() > 0.5) == y_val.astype(bool)).mean())
    stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=args.patience,
                                            restore_best_weights=True)
    hist = model.fit(x_fit, y_fit, validation_data=(x_val, y_val), epochs=args.max_epochs,
                     batch_size=64, callbacks=[stop], verbose=0)

    probs = model.predict(x_test, batch_size=256, verbose=0).ravel()
    bt = bt_recovery_fn(test_df, probs)
    val_hist = hist.history[monitor]
    out = {"accuracy": float(((probs > 0.5).astype(int) == y_test.astype(int)).mean()),
           "spearman": bt.get("spearman"), "kendall": bt.get("kendall"),
           "bt_n_locations": bt.get("n_locations"),
           "epochs_trained": len(hist.history["loss"]),
           "best_epoch": int(np.argmax(val_hist)) + 1,
           "val_acc_start": val0, "val_acc_best": float(np.max(val_hist)),
           "n_fit": len(fit_df), "n_val": len(val_df)}
    del model
    gc.collect()
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Transfer with the from-scratch training budget")
    ap.add_argument("--targets", default=",".join(CITIES))
    ap.add_argument("--sources", default="", help="default: the six other single cities")
    ap.add_argument("--recipe", required=True, choices=["layerwise", "match"])
    ap.add_argument("--sizes", default="5000")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/from_scratch"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands
    from BT_utils.evalBT import evaluate_model_bt_recovery

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    os.makedirs(args.out_dir, exist_ok=True)
    recipe = args.recipe
    out_csv = os.path.join(args.out_dir, f"transfer_long_{recipe}.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
        sources = ([s.strip() for s in args.sources.split(",") if s.strip()] or CITIES)
        sources = [s for s in sources if s != tgt]
        train_pool, test_data = reconstruct_split(tgt, args.voter, args.root_path)
        cache = {}
        for size in [int(s) for s in args.sizes.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                train_df = random_sample(train_pool, size, seed)
                for src in sources:
                    key = ((done["target"] == tgt) & (done["source"] == src)
                           & (done["sample_size"] == size) & (done["random_seed"] == seed)) if len(done) else None
                    if key is not None and key.any():
                        print(f"  skip {src}->{tgt} n={size} seed={seed} (done)")
                        continue
                    model_path = (f"{args.root_path}/trained_models/voter/"
                                  f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
                    t0 = time.time()
                    m = train_one(model_path, recipe, train_df, test_data, seed, args, sub_bands,
                                  preprocess_fn, cache, evaluate_model_bt_recovery)
                    row = {"source": src, "target": tgt, "recipe": recipe, "sample_size": size,
                           "random_seed": seed, **m, "duration_seconds": round(time.time() - t0, 1)}
                    print(f"  {src}->{tgt} {recipe} n={size} seed={seed}  sp={m['spearman']:.3f}  "
                          f"acc={m['accuracy']:.3f}  val {m['val_acc_start']:.3f}->{m['val_acc_best']:.3f}  "
                          f"epochs={m['epochs_trained']} (best {m['best_epoch']})  ({row['duration_seconds']}s)")
                    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                    done.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")


if __name__ == "__main__":
    main()
