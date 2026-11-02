"""
dump_iso_scores.py  (review round 1: R1-M7 sensitivity analysis)
================================================================

The study only ever used the isolation forests' binary predict() output, so no
anomaly scores were stored. Re-thresholding (contamination / IF cutoff
sensitivity) needs the continuous scores, both on the data each forest was
trained on (to place a new cutoff at a given quantile) and on the target pairs
(to re-assign categories). This script computes them once per source model so
every sensitivity analysis afterwards runs locally without TensorFlow.

Run on CRIB from scripts/finetune_experiments/:

  PYTHONNOUSERSITE=1 python dump_iso_scores.py --sources ACC,TEM,NBO,KIS,BEI,CHI,MUM

Writes, per source S, to {out-dir}/{S}/
  forests.json        offset_, contamination, n_estimators, max_samples_ per forest,
                      plus sf_offset_check (Score_IF offset_ vs the 10% quantile of the
                      rebuilt training scores; they should agree)
  locations.csv.gz    one row per (city, location):
                        single_score      Image_IF score_samples (higher = more normal)
                        single_cls        study's predict() output (check)
                        sf_res_oof, sf_score_oof   source city only: Score_IF training
                                          residuals (5-fold out-of-fold) and scores
                        sf_res_all/_pool  targets: scaled score-feature residual, BT fitted
                                          on all target comparisons (study) / train pool only
                        sf_score_all/_pool  Score_IF score_samples on |residual|
  pairs.csv.gz        one row per comparison:
                        split             source_ref / train / test
                        y, src_prob       label and source-model P(left wins) (targets only)
                        diff_score_lr     Pair_IF score_samples on f(left) - f(right) (study)
                        diff_score_rl     same on f(right) - f(left)  (orientation check)
                        diff_cls          study's predict() output (check)

sklearn identity used downstream: predict() == -1  <=>  score_samples < offset_.
With contamination='auto' offset_ = -0.5; with contamination=c it is the
c-quantile of the training-set scores, which is what split == source_ref gives.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from finetune_data import load_pairs, reconstruct_split
from run_sampling_experiment import _SRC_TRAIN, _path

CITIES = ["ACC", "TEM", "NBO", "KIS", "BEI", "CHI", "MUM"]


def encode_locations(encoder, locs, root_path, sub_bands, preprocess_fn, batch_size=64):
    import tensorflow as tf
    feats = []
    for i in range(0, len(locs), batch_size):
        imgs = np.array([preprocess_fn(os.path.join(root_path, p), sub_bands)
                         for p in locs[i:i + batch_size]])
        f = encoder.predict_on_batch(imgs)
        if tf.is_tensor(f):
            f = f.numpy()
        feats.append(f.reshape(f.shape[0], -1))
    return np.concatenate(feats)


def forest_meta(forest):
    if forest is None:
        return None
    return {"offset": float(forest.offset_), "contamination": str(forest.contamination),
            "n_estimators": int(forest.n_estimators), "max_samples": int(forest.max_samples_)}


def sf_scores(df, loc_index, loc_pca, iso, bt_estimator):
    """Scaled residual and Score_IF score per location, BT fitted on df only."""
    out_res = np.full(len(loc_index), np.nan)
    out_score = np.full(len(loc_index), np.nan)
    if iso.get("iso_score_feature") is None or len(df) == 0:
        return out_res, out_score
    bt = bt_estimator(df, df["y"].values)
    keys = [p for p in bt if p in loc_index]
    rows = np.array([loc_index[p] for p in keys])
    residuals = np.array([bt[p] for p in keys]) - iso["mlp_score_regressor"].predict(loc_pca[rows])
    res_scaled = iso["scaler_residual"].transform(residuals.reshape(-1, 1)).ravel()
    out_res[rows] = res_scaled
    out_score[rows] = iso["iso_score_feature"].score_samples(np.abs(res_scaled).reshape(-1, 1))
    return out_res, out_score


def sf_oof_scores(df, loc_index, loc_pca, iso, bt_estimator):
    """Score_IF training scores, rebuilt as where_transfer_trainIso_BT_v2.py fits them:
    5-fold out-of-fold MLP residuals over the source locations in first-seen order."""
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPRegressor

    out_res = np.full(len(loc_index), np.nan)
    out_score = np.full(len(loc_index), np.nan)
    if iso.get("iso_score_feature") is None:
        return out_res, out_score, None
    bt = bt_estimator(df, df["y"].values)
    order = [_path(p) for p in pd.concat([df["left"], df["right"]]).unique()]
    sf_locs = [p for p in dict.fromkeys(order) if p in bt]
    rows = np.array([loc_index[p] for p in sf_locs])
    x, y = loc_pca[rows], np.array([bt[p] for p in sf_locs])
    oof = np.zeros(len(sf_locs))
    for tr, va in KFold(n_splits=min(5, len(sf_locs)), shuffle=True, random_state=42).split(x):
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.01,
                           max_iter=1000, random_state=42).fit(x[tr], y[tr])
        oof[va] = y[va] - mlp.predict(x[va])
    res_scaled = iso["scaler_residual"].transform(oof.reshape(-1, 1)).ravel()
    out_res[rows] = res_scaled
    out_score[rows] = iso["iso_score_feature"].score_samples(np.abs(res_scaled).reshape(-1, 1))
    # contamination=0.1 puts offset_ at the 10% quantile of the training scores
    check = {"offset": float(iso["iso_score_feature"].offset_),
             "q10_rebuilt_scores": float(np.quantile(out_score[rows], 0.1))}
    return out_res, out_score, check


def city_tables(city, pair_sets, encoder, iso, args, sub_bands, preprocess_fn, bt_estimator):
    """pair_sets: {split: DataFrame}. Returns (locations, pairs, sf_check) for one city.

    For the source reference the SF columns hold the rebuilt out-of-fold training
    residuals (sf_res_oof / sf_score_oof); for targets they hold residuals of the
    final MLP with BT fitted on all target comparisons (_all, study) or the train
    pool only (_pool)."""
    all_pairs = pd.concat(pair_sets.values())
    locs = sorted(set(map(_path, all_pairs["left"])) | set(map(_path, all_pairs["right"])))
    loc_index = {p: i for i, p in enumerate(locs)}
    raw = encode_locations(encoder, locs, args.root_path, sub_bands, preprocess_fn)
    loc_pca = iso["pca_single"].transform(iso["scaler_single"].transform(raw))

    loc_tab = pd.DataFrame({"city": city, "location": locs,
                            "single_score": iso["iso_single"].score_samples(loc_pca),
                            "single_cls": iso["iso_single"].predict(loc_pca)})
    sf_check = None
    if "source_ref" in pair_sets:
        loc_tab["sf_res_oof"], loc_tab["sf_score_oof"], sf_check = sf_oof_scores(
            pair_sets["source_ref"], loc_index, loc_pca, iso, bt_estimator)
    else:
        labelled = {"all": pd.concat([pair_sets["train"], pair_sets["test"]]),
                    "pool": pair_sets["train"]}
        for tag, df in labelled.items():
            loc_tab[f"sf_res_{tag}"], loc_tab[f"sf_score_{tag}"] = sf_scores(
                df, loc_index, loc_pca, iso, bt_estimator)

    pair_tabs = []
    for split, df in pair_sets.items():
        li = np.array([loc_index[_path(p)] for p in df["left"]])
        ri = np.array([loc_index[_path(p)] for p in df["right"]])
        d_lr = iso["pca_diff"].transform(iso["scaler_diff"].transform(raw[li] - raw[ri]))
        d_rl = iso["pca_diff"].transform(iso["scaler_diff"].transform(raw[ri] - raw[li]))
        pair_tabs.append(pd.DataFrame({
            "city": city, "split": split,
            "left": [locs[i] for i in li], "right": [locs[i] for i in ri],
            "y": np.asarray(df["y"], dtype=int).ravel(),
            "diff_score_lr": iso["iso_pair_diff"].score_samples(d_lr),
            "diff_score_rl": iso["iso_pair_diff"].score_samples(d_rl),
            "diff_cls": iso["iso_pair_diff"].predict(d_lr),
        }))
    return loc_tab, pd.concat(pair_tabs, ignore_index=True), sf_check


def run_source(src, args, sub_bands, preprocess_fn, bt_estimator):
    import tensorflow as tf
    from finetune_core import create_dataset, evaluate_model
    from finetune_informed import load_iso_models, _resolve_feature_extractor

    out = os.path.join(args.out_dir, src)
    if os.path.exists(os.path.join(out, "pairs.csv.gz")):
        print(f"[{src}] done, skipping")
        return
    os.makedirs(out, exist_ok=True)
    iso = load_iso_models(src, args.iso_model_dir, voter=args.voter)
    meta = {k: forest_meta(iso.get(k)) for k in ("iso_single", "iso_pair_diff", "iso_score_feature")}

    model_path = (f"{args.root_path}/trained_models/voter/"
                  f"{src}_voter_{args.voter}_ensembleVoter_{args.d_src}_1_bradley_terry.ckpt")
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path)
    encoder = _resolve_feature_extractor(model)

    loc_tabs, pair_tabs = [], []
    # reference: the data the source forests were fitted on -- every comparison in
    # the source's own votes table, as in where_transfer_trainIso_BT_v2.py
    l, p, meta["sf_offset_check"] = city_tables(
        src, {"source_ref": load_pairs(src, args.voter, args.root_path)},
        encoder, iso, args, sub_bands, preprocess_fn, bt_estimator)
    loc_tabs.append(l)
    pair_tabs.append(p)
    print(f"[{src}] reference: {len(p)} pairs, SF offset check {meta['sf_offset_check']}")
    with open(os.path.join(out, "forests.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for tgt in [c for c in CITIES if c not in _SRC_TRAIN.get(src, {src})]:
        train, test = reconstruct_split(tgt, args.voter, args.root_path)
        l, p, _ = city_tables(tgt, {"train": train, "test": test},
                              encoder, iso, args, sub_bands, preprocess_fn, bt_estimator)
        probs = []
        for df in (train, test):
            _, pr = evaluate_model(model, create_dataset(df, args.root_path, sub_bands, preprocess_fn))
            probs.append(np.asarray(pr).ravel())
        p["src_prob"] = np.concatenate(probs)
        loc_tabs.append(l)
        pair_tabs.append(p)
        print(f"[{src}] target {tgt}: {len(p)} pairs")

    pd.concat(loc_tabs, ignore_index=True).to_csv(os.path.join(out, "locations.csv.gz"), index=False)
    pd.concat(pair_tabs, ignore_index=True).to_csv(os.path.join(out, "pairs.csv.gz"), index=False)
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Cache isolation-forest scores per source model")
    ap.add_argument("--sources", default=",".join(CITIES))
    ap.add_argument("--voter", default="communities")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--iso-model-dir", default=os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"))
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/iso_scores"))
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    from calibrate_degree import _load_callable
    from citywide_score import default_get_sub_bands
    from BT_utils.evalBT import estimate_bt_scores_from_predictions

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        run_source(src, args, sub_bands, preprocess_fn, estimate_bt_scores_from_predictions)


if __name__ == "__main__":
    main()
