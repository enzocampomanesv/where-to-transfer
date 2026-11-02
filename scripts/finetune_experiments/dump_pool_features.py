"""
dump_pool_features.py  (review round 1: R1-M7 sensitivity analysis, Tier 2 Stage A)
===================================================================================

Replaying the sat_prospective sampler offline needs what extract_pool_features
computes on CRIB and the score-feature (SF) models, none of which
dump_iso_scores.py stores:

  * single-image PCA features of every train-pool location (kNN for SF rates)
  * pair-difference PCA features of every train-pool pair (kNN for PD rates)
  * the SF pipeline: score MLP, residual scaler and Score_IF, exported as plain
    arrays so the replay does not depend on unpickling sklearn objects locally

Run on CRIB from scripts/finetune_experiments/ (same environment as the rerun):

  PYTHONNOUSERSITE=1 python dump_pool_features.py --sources ACC,TEM,NBO,KIS,BEI,CHI,MUM

Writes, per source S, to {out-dir}/{S}/
  sf_model.npz      MLP weights (coef_i, intercept_i), residual scaler mean/scale,
                    Score_IF trees (concatenated node arrays + per-tree offsets),
                    max_samples, offset
  checks.json       max |difference| between the exported arrays and the sklearn
                    objects (MLP predict, Score_IF score_samples); both should be ~0
  {T}.npz           per target T: loc_path, loc_pca, loc_img_cls (train-pool locations,
                    sorted as in extract_pool_features), diff_pca, diff_cls (train pool
                    in reconstruct_split order)
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np

from dump_iso_scores import CITIES, encode_locations
from finetune_data import reconstruct_split
from run_sampling_experiment import _SRC_TRAIN, _path


def export_sf_model(iso):
    """Plain-array copy of the SF pipeline."""
    mlp, scaler, forest = iso["mlp_score_regressor"], iso["scaler_residual"], iso["iso_score_feature"]
    out = {"n_layers": np.array(len(mlp.coefs_)),
           "activation": np.array(mlp.activation), "out_activation": np.array(mlp.out_activation_),
           "res_mean": np.asarray(scaler.mean_, float), "res_scale": np.asarray(scaler.scale_, float),
           "max_samples": np.array(forest.max_samples_), "offset": np.array(forest.offset_)}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        out[f"coef_{i}"], out[f"intercept_{i}"] = w, b
    trees = [e.tree_ for e in forest.estimators_]
    out["tree_start"] = np.cumsum([0] + [t.node_count for t in trees])[:-1]
    for name in ("children_left", "children_right", "feature", "threshold", "n_node_samples"):
        out[name] = np.concatenate([getattr(t, name) for t in trees])
    # 1-D forest: every tree uses feature 0 of |residual|
    out["estimators_features"] = np.array([f[0] for f in forest.estimators_features_])
    return out


def _avg_path_length(n):
    n = np.asarray(n, float)
    out = np.where(n <= 1, 0.0, np.where(n == 2, 1.0, 0.0))
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out


def sf_score_from_arrays(m, x):
    """Score_IF score_samples from the exported arrays (sklearn's computation)."""
    x = np.asarray(x, np.float32).astype(float).ravel()   # sklearn casts X to float32
    depth = np.zeros(len(x))
    starts = list(m["tree_start"]) + [len(m["threshold"])]
    for s, e in zip(starts[:-1], starts[1:]):
        cl, cr, thr, ns = (m[k][s:e] for k in ("children_left", "children_right", "threshold", "n_node_samples"))
        node, d = np.zeros(len(x), int), np.zeros(len(x))
        active = cl[node] != -1
        while active.any():
            go_left = x[active] <= thr[node[active]]
            node[active] = np.where(go_left, cl[node[active]], cr[node[active]])
            d[active] += 1
            active = cl[node] != -1
        depth += d + _avg_path_length(ns[node])
    n_trees = len(m["tree_start"])
    return -(2.0 ** (-depth / (n_trees * _avg_path_length([int(m["max_samples"])])[0])))


def mlp_from_arrays(m, x):
    h = np.asarray(x, float)
    for i in range(int(m["n_layers"])):
        h = h @ m[f"coef_{i}"] + m[f"intercept_{i}"]
        if i < int(m["n_layers"]) - 1:
            h = np.maximum(h, 0.0)   # relu (checked against predict below)
    return h.ravel()


def run_source(src, args, sub_bands, preprocess_fn):
    import tensorflow as tf
    from finetune_informed import load_iso_models, _resolve_feature_extractor

    out = os.path.join(args.out_dir, src)
    if os.path.exists(os.path.join(out, "checks.json")):
        print(f"[{src}] done, skipping")
        return
    os.makedirs(out, exist_ok=True)
    iso = load_iso_models(src, args.iso_model_dir, voter=args.voter)
    sf = export_sf_model(iso)
    np.savez_compressed(os.path.join(out, "sf_model.npz"), **sf)

    model_path = (f"{args.root_path}/trained_models/voter/"
                  f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path)
    encoder = _resolve_feature_extractor(model)

    checks = {"activation": str(sf["activation"]), "out_activation": str(sf["out_activation"])}
    mlp_err, if_err = 0.0, 0.0
    for tgt in [c for c in CITIES if c not in _SRC_TRAIN.get(src, {src})]:
        train_pool, _ = reconstruct_split(tgt, args.voter, args.root_path)
        left = [_path(p) for p in train_pool["left"]]
        right = [_path(p) for p in train_pool["right"]]
        locs = sorted(set(left) | set(right))
        loc_index = {p: i for i, p in enumerate(locs)}
        raw = encode_locations(encoder, locs, args.root_path, sub_bands, preprocess_fn)
        loc_pca = iso["pca_single"].transform(iso["scaler_single"].transform(raw))
        li = np.array([loc_index[p] for p in left])
        ri = np.array([loc_index[p] for p in right])
        diff_pca = iso["pca_diff"].transform(iso["scaler_diff"].transform(raw[li] - raw[ri]))
        np.savez_compressed(os.path.join(out, f"{tgt}.npz"),
                            loc_path=np.array(locs), loc_pca=loc_pca,
                            loc_img_cls=iso["iso_single"].predict(loc_pca),
                            diff_pca=diff_pca, diff_cls=iso["iso_pair_diff"].predict(diff_pca))

        # exported SF arrays must reproduce the sklearn objects
        pred = iso["mlp_score_regressor"].predict(loc_pca)
        mlp_err = max(mlp_err, float(np.abs(mlp_from_arrays(sf, loc_pca) - pred).max()))
        x = np.abs(np.concatenate([np.linspace(0, 8, 4001),
                                   np.random.RandomState(0).standard_normal(4000)])).reshape(-1, 1)
        if_err = max(if_err, float(np.abs(sf_score_from_arrays(sf, x)
                                          - iso["iso_score_feature"].score_samples(x)).max()))
        print(f"[{src}] target {tgt}: {len(locs)} locations, {len(train_pool)} pool pairs")
        del raw
        gc.collect()

    checks.update({"mlp_max_abs_err": mlp_err, "score_if_max_abs_err": if_err})
    print(f"[{src}] checks {checks}")
    with open(os.path.join(out, "checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Export pool PCA features and SF models per source")
    ap.add_argument("--sources", default=",".join(CITIES))
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--iso-model-dir", default=os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"))
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/pool_features"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        run_source(src, args, sub_bands, preprocess_fn)


if __name__ == "__main__":
    main()
