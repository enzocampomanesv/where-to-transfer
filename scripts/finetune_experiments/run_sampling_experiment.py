"""
run_sampling_experiment.py  (v3 -- review round 1, R1-M1 / R3-m3)
=================================================================

One source->target pair per invocation. Compares four finetune sampling
strategies on the study's fixed split, over several seeds:

  random           train_pool.sample(n, seed)  (independent per n, as in the study)
  oracle_t1        study t1 sampler, SF scores fitted on train pool only (no test)
  sat_prospective  sequential rounds, labels used only once collected
  uncertainty      sequential rounds, current-model margin closest to 0.5

Sequential strategies start from the same round-0 batch as random at the first
sample size, then grow the labelled set cumulatively to each later size. Every
checkpoint is finetuned fresh from the source weights on the cumulative set, so
results are comparable across strategies at each n.

The test set is used ONLY for evaluation. Nothing in the selection path sees it.

Example:
  python run_sampling_experiment.py --source ACC --target TEM \
      --samplings random,oracle_t1,sat_prospective,uncertainty \
      --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from finetune_data import reconstruct_split, random_sample
import finetune_prospective as fp

SAMPLE_SIZES = [100, 500, 1000, 5000]
SEEDS = [42, 43, 44, 45, 46]
SAMPLINGS = ["random", "oracle_t1", "sat_prospective", "uncertainty"]
SEQUENTIAL = {"sat_prospective", "uncertainty"}

# Individual cities each source model was trained on / each target contains
# (from ACC_transferLearn_BT.py) -- used to refuse overlapping pairs.
_SRC_TRAIN = {
    "ACC": {"ACC"}, "NBO": {"NBO"}, "BEI": {"BEI"}, "TEM": {"TEM"},
    "KIS": {"KIS"}, "CHI": {"CHI"}, "MUM": {"MUM"},
    "PRI": {"ACC", "NBO", "BEI"}, "SEC": {"TEM", "KIS", "CHI"},
    "AFR": {"ACC", "NBO", "BEI", "TEM", "KIS", "CHI"},
    "PRI_ACC": {"NBO", "BEI"}, "PRI_NBO": {"ACC", "BEI"}, "PRI_BEI": {"ACC", "NBO"},
    "SEC_TEM": {"KIS", "CHI"}, "SEC_KIS": {"TEM", "CHI"}, "SEC_CHI": {"TEM", "KIS"},
    "AFR_ACC": {"NBO", "BEI", "TEM", "KIS", "CHI"}, "AFR_NBO": {"ACC", "BEI", "TEM", "KIS", "CHI"},
    "AFR_BEI": {"ACC", "NBO", "TEM", "KIS", "CHI"}, "AFR_TEM": {"ACC", "NBO", "BEI", "KIS", "CHI"},
    "AFR_KIS": {"ACC", "NBO", "BEI", "TEM", "CHI"}, "AFR_CHI": {"ACC", "NBO", "BEI", "TEM", "KIS"},
}


def _path(p):
    return p.decode("utf-8") if isinstance(p, bytes) else str(p)


# ===========================================================================
# Label-free pool features (source model + source iso-forests)
# ===========================================================================
def extract_pool_features(model, train_pool, iso, root_path, sub_bands, preprocess_fn,
                          create_dataset, evaluate_model, batch_size=64) -> Dict:
    """Everything the selection path may use, computed once per pair.

    Per location: single-image PCA features + Image_IF class.
    Per pair: source prediction / probability, Pair_IF class on the raw
    feature difference (as in categorize_with_iso), pair-diff PCA, tier.
    No labels are read here.
    """
    import tensorflow as tf
    from finetune_informed import _resolve_feature_extractor

    encoder = _resolve_feature_extractor(model)
    left = [_path(p) for p in train_pool["left"]]
    right = [_path(p) for p in train_pool["right"]]
    locs = sorted(set(left) | set(right))
    loc_index = {p: i for i, p in enumerate(locs)}

    feats = []
    for i in range(0, len(locs), batch_size):
        imgs = np.array([preprocess_fn(os.path.join(root_path, p), sub_bands)
                         for p in locs[i:i + batch_size]])
        f = encoder.predict_on_batch(imgs)
        if tf.is_tensor(f):
            f = f.numpy()
        feats.append(f.reshape(f.shape[0], -1))
    raw = np.concatenate(feats)

    loc_pca = iso["pca_single"].transform(iso["scaler_single"].transform(raw))
    loc_img_cls = iso["iso_single"].predict(loc_pca)

    left_loc = np.array([loc_index[p] for p in left])
    right_loc = np.array([loc_index[p] for p in right])
    diff_pca = iso["pca_diff"].transform(iso["scaler_diff"].transform(raw[left_loc] - raw[right_loc]))
    diff_cls = iso["iso_pair_diff"].predict(diff_pca)

    pool_ds = create_dataset(train_pool, root_path, sub_bands, preprocess_fn)
    source_pred, source_prob = evaluate_model(model, pool_ds)

    return {
        "loc_index": loc_index, "loc_pca": loc_pca, "loc_img_cls": loc_img_cls,
        "left_loc": left_loc, "right_loc": right_loc, "diff_pca": diff_pca,
        "tiers": fp.label_free_tiers(loc_img_cls[left_loc], loc_img_cls[right_loc], diff_cls),
        "source_pred": np.asarray(source_pred).astype(int),
        "source_prob": np.asarray(source_prob),
    }


# ===========================================================================
# Finetune + evaluate (study settings, via finetune_core)
# ===========================================================================
def finetune_and_evaluate(model_path, finetune_df, test_data, root_path, sub_bands,
                          preprocess_fn, bt_recovery_fn, keep_model=False):
    import tensorflow as tf
    from finetune_core import (create_dataset, evaluate_model, setup_layerwise_training,
                               FINE_TUNE_EPOCHS, PATIENCE, BASE_LR)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = tf.keras.models.load_model(model_path)
    setup_layerwise_training(model, BASE_LR)
    train_ds = create_dataset(finetune_df, root_path, sub_bands, preprocess_fn)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=PATIENCE,
                                                  restore_best_weights=True)
    model.fit(train_ds, epochs=FINE_TUNE_EPOCHS, callbacks=[early_stop], verbose=0)

    test_ds = create_dataset(test_data, root_path, sub_bands, preprocess_fn)
    preds, probs = evaluate_model(model, test_ds)
    y = np.array(test_data["y"], dtype=int).flatten()
    bt = bt_recovery_fn(test_data, probs)
    metrics = {"accuracy": float((y == np.asarray(preds, int).flatten()).mean()),
               "spearman": bt.get("spearman"), "kendall": bt.get("kendall"),
               "bt_n_locations": bt.get("n_locations")}
    if keep_model:
        return metrics, model
    del model
    gc.collect()
    return metrics, None


# ===========================================================================
# Pair driver
# ===========================================================================
def run_pair(args, preprocess_fn, sub_bands, bt_recovery_fn, bt_estimator):
    import tensorflow as tf
    from finetune_core import create_dataset, evaluate_model
    from finetune_informed import load_iso_models, informed_sample_unified

    src, tgt = args.source, args.target
    if _SRC_TRAIN.get(src, {src}) & {tgt}:
        raise SystemExit(f"{src} -> {tgt}: target is in the source's training cities.")

    model_path = (f"{args.root_path}/trained_models/voter/"
                  f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{src}_{tgt}_sampling_v3.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    train_pool, test_data = reconstruct_split(tgt, args.voter, args.root_path)
    # information-set guard: the split is by row, so the two index sets must be disjoint
    assert not set(train_pool.index) & set(test_data.index), "train pool overlaps test set"
    # positional index from here on; '_pos' survives informed_sample_unified's iloc
    train_pool = train_pool.reset_index(drop=True)
    train_pool["_pos"] = np.arange(len(train_pool))
    iso = load_iso_models(src, args.iso_model_dir, voter=args.voter)
    y_pool = train_pool["y"].to_numpy()

    # zero-shot + label-free pool features (one source-model load)
    tf.keras.backend.clear_session()
    src_model = tf.keras.models.load_model(model_path)
    zs_preds, zs_probs = evaluate_model(src_model, create_dataset(
        test_data, args.root_path, sub_bands, preprocess_fn))
    y_test = np.array(test_data["y"], dtype=int).flatten()
    zero = {"zero_accuracy": float((y_test == np.asarray(zs_preds, int)).mean()),
            "zero_spearman": bt_recovery_fn(test_data, zs_probs).get("spearman")}
    pool = extract_pool_features(src_model, train_pool, iso, args.root_path, sub_bands,
                                 preprocess_fn, create_dataset, evaluate_model)
    del src_model
    tf.keras.backend.clear_session()
    gc.collect()

    true_cats = fp.oracle_categories(pool["tiers"], y_pool, pool["source_pred"])

    # corrected oracle: SF Bradley-Terry on the train pool only
    oracle_sf = fp.sf_flags_from_labelled(
        train_pool, pool["loc_index"], pool["loc_pca"], pool["loc_img_cls"],
        iso.get("mlp_score_regressor"), iso.get("scaler_residual"),
        iso.get("iso_score_feature"), bt_estimator)
    oracle_sf_pairs = [oracle_sf.get(int(l)) is True or oracle_sf.get(int(r)) is True
                       for l, r in zip(pool["left_loc"], pool["right_loc"])]
    # effective SF tier as the study's samplers define it (Structural dominates)
    oracle_sf_eff = np.array(oracle_sf_pairs) & (pool["tiers"] != fp.TIER_STRUCTURAL)

    base = {"source": src, "target": tgt, "strategy": "layerwise",
            "pool_size": len(train_pool), "test_size": len(test_data), **zero,
            "pool_sf_pct_oracle": float(oracle_sf_eff.mean()),
            "pool_cultural_pct": float((true_cats == "Cultural Disagreement").mean()),
            "pool_structural_pct": float((pool["tiers"] == fp.TIER_STRUCTURAL).mean()),
            "pool_relational_pct": float((pool["tiers"] == fp.TIER_RELATIONAL).mean())}
    sizes = sorted(s for s in args.sizes if s < len(train_pool))

    for sampling in args.samplings:
        for seed in args.seeds:
            if len(done) and ((done["sampling"] == sampling) & (done["random_seed"] == seed)).sum() >= len(sizes):
                print(f"  skip {sampling} seed={seed} (done)")
                continue
            rows: List[Dict] = []
            rng = np.random.RandomState(seed)
            labelled = np.array([], dtype=int)
            cur_model = None

            for n in sizes:
                t0 = time.time()
                diag = {}
                if sampling == "random":
                    ft_df = random_sample(train_pool, n, seed)
                    idx = ft_df["_pos"].to_numpy()
                elif sampling == "oracle_t1":
                    ft_df, _ = informed_sample_unified(
                        train_pool, list(true_cats), oracle_sf_pairs, n, seed, temperature=1.0)
                    idx = ft_df["_pos"].to_numpy()
                else:
                    if len(labelled) == 0:
                        labelled = fp.seed_batch(train_pool, n, seed)
                    else:
                        n_new = n - len(labelled)
                        if sampling == "sat_prospective":
                            new, diag = fp.sat_prospective_increment(
                                pool=pool, train_pool=train_pool, labelled_idx=labelled,
                                n_new=n_new, rng=rng, iso=iso, bt_estimator=bt_estimator,
                                k=args.k)
                        else:
                            available = np.setdiff1d(np.arange(len(train_pool)), labelled)
                            p_left = _predict_pool(cur_model, train_pool.iloc[available],
                                                   args.root_path, sub_bands, preprocess_fn,
                                                   create_dataset, evaluate_model)
                            p_full = np.full(len(train_pool), np.nan)
                            p_full[available] = p_left
                            new = fp.uncertainty_pick(p_full, available, n_new, rng)
                        labelled = np.concatenate([labelled, new])
                    idx = labelled
                    ft_df = train_pool.iloc[idx].reset_index(drop=True)

                assert len(ft_df) == n and len(set(idx)) == n, f"expected {n} distinct pairs"

                if cur_model is not None:
                    del cur_model
                    cur_model = None
                metrics, cur_model = finetune_and_evaluate(
                    model_path, ft_df, test_data, args.root_path, sub_bands, preprocess_fn,
                    bt_recovery_fn, keep_model=(sampling == "uncertainty"))

                row = {**base, "sample_size": n, "sampling": sampling, "random_seed": seed,
                       **metrics,
                       "improvement": metrics["accuracy"] - zero["zero_accuracy"],
                       "sp_improvement": (metrics["spearman"] - zero["zero_spearman"])
                       if metrics["spearman"] is not None and zero["zero_spearman"] is not None else None,
                       **fp.selection_breakdown(true_cats, idx),
                       "sel_sf_oracle": int(oracle_sf_eff[idx].sum()),
                       **fp.labelled_diagnostics(pool["tiers"], y_pool, pool["source_pred"],
                                                 idx, pool["left_loc"], pool["right_loc"]),
                       **diag,
                       "duration_seconds": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"  {sampling:16s} seed={seed} n={n:5d}  sp={metrics['spearman']:.3f}  "
                      f"({row['duration_seconds']}s)")

            if cur_model is not None:
                del cur_model
            gc.collect()
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")


def _predict_pool(model, df, root_path, sub_bands, preprocess_fn, create_dataset, evaluate_model):
    _, probs = evaluate_model(model, create_dataset(df, root_path, sub_bands, preprocess_fn))
    return np.asarray(probs)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Leakage-free sampling comparison for one pair")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--samplings", default=",".join(SAMPLINGS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--sizes", default=",".join(map(str, SAMPLE_SIZES)))
    ap.add_argument("--k", type=int, default=fp.DEFAULT_K)
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--iso-model-dir", default=os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"))
    ap.add_argument("--out-dir",
                    default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning/sampling_v3"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()
    args.samplings = [s.strip() for s in args.samplings.split(",") if s.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.sizes = [int(s) for s in args.sizes.split(",")]
    unknown = set(args.samplings) - set(SAMPLINGS)
    if unknown:
        raise SystemExit(f"unknown sampling(s): {sorted(unknown)}")

    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands
    from BT_utils.evalBT import evaluate_model_bt_recovery, estimate_bt_scores_from_predictions

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    run_pair(args, preprocess_fn, sub_bands, evaluate_model_bt_recovery,
             estimate_bt_scores_from_predictions)


if __name__ == "__main__":
    main()
