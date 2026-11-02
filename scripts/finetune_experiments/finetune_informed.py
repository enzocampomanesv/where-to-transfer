"""
finetune_informed.py  (v2 -- informed sampling)
==============================================

Ported verbatim (in logic) from ``ACC_transferLearn_BT.py`` so an informed-t1
finetune reproduces the study. Split by testability:

  PURE (no model / tiles / TF)  -- unit-tested in the sandbox:
    * INFORMED_PRIORITY_ORDER / INFORMED_BASE_WEIGHTS / HARD_FILL_TEMPERATURE
    * informed_sample_unified  (effective tier assignment + temperature draw)

  MODEL/TILE-BOUND -- validated on infra by the gate (informed variant in the
  informed_t1 band):
    * categorize_with_iso        (AoA labels via iso-forests)
    * get_sf_misalignment_flags  (SF flags; optional -> {} if SF models absent)
    * categorize_pool_for_sampling (orchestrator -> (aoa_labels, sf_pair_flags))
    * load_iso_models            (the 6 required + 3 optional joblib artifacts)

Scoping (per study): temperature=1.0, adaptive=False (t1 soft-mix). Hard-fill and
uniform paths are ported for completeness but not used.

Path convention (option 2): identifiers stay RELATIVE (raw DB paths); images are
loaded with root_path prepended via os.path.join, matching the original's
categorize/SF loops. So this is CWD-independent.

Version-sensitive seam GUARDED: the feature extractor is
``model.layers[2].get_layer('feature_extractor')`` in the source. We resolve it
the same way but verify the result, raising loudly if the structure differs on a
given Keras/model rather than silently extracting the wrong tensor.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- tier constants (verbatim) ---------------------------------------------
HARD_FILL_TEMPERATURE = 15.0
INFORMED_PRIORITY_ORDER = [
    "SF_Misaligned",
    "Cultural Disagreement",
    "Relational Ignorance",
    "Consensus",
    "Structural Ignorance",
]
INFORMED_BASE_WEIGHTS = {
    "SF_Misaligned":         7.0,
    "Cultural Disagreement": 1.0,
    "Relational Ignorance":  1.0,
    "Consensus":             0.7,
    "Structural Ignorance":  0.3,
}


# ===========================================================================
# PURE sampler (verbatim logic; fully testable)
# ===========================================================================
def informed_sample_unified(train_pool, aoa_labels, sf_pair_flags,
                            sample_size, seed, temperature):
    """Temperature-controlled informed draw. Returns (sampled_df, breakdown).

    Effective tier per pair: Structural Ignorance dominates (OOD); else an SF flag
    promotes to SF_Misaligned; else the AoA label. For T < HARD_FILL_TEMPERATURE
    (the t1 case, T=1) a softmax over base weights gives a without-replacement draw.
    """
    rng = np.random.RandomState(seed)

    effective = []
    for aoa, sf in zip(aoa_labels, sf_pair_flags):
        if aoa == "Structural Ignorance":
            effective.append("Structural Ignorance")
        elif sf:
            effective.append("SF_Misaligned")
        else:
            effective.append(aoa)

    if temperature >= HARD_FILL_TEMPERATURE:
        tier_indices = {t: [] for t in INFORMED_PRIORITY_ORDER}
        for i, t in enumerate(effective):
            tier_indices[t].append(i)
        for t in INFORMED_PRIORITY_ORDER:
            rng.shuffle(tier_indices[t])
        selected = []
        for t in INFORMED_PRIORITY_ORDER:
            needed = sample_size - len(selected)
            if needed <= 0:
                break
            selected.extend(tier_indices[t][:needed])
    else:
        base_w = np.array([INFORMED_BASE_WEIGHTS.get(t, 1e-6) for t in effective], dtype=float)
        if temperature == 0:
            prob = np.ones(len(effective)) / len(effective)
        else:
            log_w = np.log(np.maximum(base_w, 1e-12)) * temperature
            log_w -= log_w.max()
            prob = np.exp(log_w)
        prob_sum = prob.sum()
        prob = prob / prob_sum if prob_sum > 0 else np.ones(len(effective)) / len(effective)
        actual_size = min(sample_size, len(train_pool))
        selected = rng.choice(len(train_pool), size=actual_size, replace=False, p=prob).tolist()

    breakdown = Counter(effective[i] for i in selected)
    sample_breakdown = {
        "sample_structural":    breakdown.get("Structural Ignorance", 0),
        "sample_sf_misaligned": breakdown.get("SF_Misaligned", 0),
        "sample_cultural":      breakdown.get("Cultural Disagreement", 0),
        "sample_relational":    breakdown.get("Relational Ignorance", 0),
        "sample_consensus":     breakdown.get("Consensus", 0),
    }
    return train_pool.iloc[selected].reset_index(drop=True), sample_breakdown


# ===========================================================================
# iso-forest artifacts
# ===========================================================================
_REQUIRED = ["single_scaler", "single_pca", "single_iso_pca",
             "pair_diff_scaler", "pair_diff_pca", "pair_diff_iso_pca"]
_OPTIONAL = ["score_feature_mlp", "score_feature_residual_scaler", "score_feature_iso"]
_KEYMAP = {
    "single_scaler": "scaler_single", "single_pca": "pca_single",
    "single_iso_pca": "iso_single", "pair_diff_scaler": "scaler_diff",
    "pair_diff_pca": "pca_diff", "pair_diff_iso_pca": "iso_pair_diff",
    "score_feature_mlp": "mlp_score_regressor",
    "score_feature_residual_scaler": "scaler_residual",
    "score_feature_iso": "iso_score_feature",
}


def load_iso_models(source_code: str, iso_model_dir: str, voter: str = "communities") -> Dict:
    """Load the iso-forest pipeline artifacts for a source. Required set must exist;
    optional SF set may be absent (SF tier then skipped)."""
    import joblib
    iso = {}
    base = f"{iso_model_dir}/models/{source_code}_{voter}"
    for name in _REQUIRED:
        path = f"{base}_{name}.joblib"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required iso artifact missing: {path}")
        iso[_KEYMAP[name]] = joblib.load(path)
    for name in _OPTIONAL:
        path = f"{base}_{name}.joblib"
        iso[_KEYMAP[name]] = joblib.load(path) if os.path.exists(path) else None
    return iso


def _resolve_feature_extractor(model):
    """model.layers[2].get_layer('feature_extractor'), guarded."""
    try:
        nested = model.layers[2]
        enc = nested.get_layer("feature_extractor")
    except Exception as e:
        raise RuntimeError(
            f"Could not resolve feature extractor at model.layers[2]"
            f".get_layer('feature_extractor'): {e}. Model structure differs from "
            f"the study's; informed categorization cannot proceed safely."
        )
    return enc


# ===========================================================================
# MODEL/TILE-bound categorization (gate-validated)
# ===========================================================================
def categorize_with_iso(test_df, preds, iso_models, model, root_path, sub_bands,
                        preprocess_fn: Callable):
    """AoA category per pair via iso-forests (verbatim logic; root_path prepended)."""
    scaler_single, pca_single, iso_single = (iso_models["scaler_single"],
                                             iso_models["pca_single"], iso_models["iso_single"])
    scaler_diff, pca_diff, iso_pair_diff = (iso_models["scaler_diff"],
                                            iso_models["pca_diff"], iso_models["iso_pair_diff"])
    import tensorflow as tf
    single_encoder = _resolve_feature_extractor(model)

    all_l, all_r, all_d = [], [], []
    bs = 64
    for i in range(0, len(test_df), bs):
        batch = test_df.iloc[i:i + bs]
        bl, br = [], []
        for row in batch.itertuples():
            lp = row.left.decode("utf-8") if isinstance(row.left, bytes) else str(row.left)
            rp = row.right.decode("utf-8") if isinstance(row.right, bytes) else str(row.right)
            bl.append(preprocess_fn(os.path.join(root_path, lp), sub_bands))
            br.append(preprocess_fn(os.path.join(root_path, rp), sub_bands))
        bl, br = np.array(bl), np.array(br)
        fl = single_encoder.predict_on_batch(bl)
        fr = single_encoder.predict_on_batch(br)
        if tf.is_tensor(fl):
            fl, fr = fl.numpy(), fr.numpy()
        fl_flat = fl.reshape(fl.shape[0], -1)
        fr_flat = fr.reshape(fr.shape[0], -1)
        fl_pca = pca_single.transform(scaler_single.transform(fl_flat))
        fr_pca = pca_single.transform(scaler_single.transform(fr_flat))
        all_l.extend(iso_single.predict(fl_pca))
        all_r.extend(iso_single.predict(fr_pca))
        pair_diff = fl_flat - fr_flat
        all_d.extend(iso_pair_diff.predict(pca_diff.transform(scaler_diff.transform(pair_diff))))

    test_df = test_df.copy()
    test_df["model_prediction"] = preds
    test_df["iso_left_class"] = all_l
    test_df["iso_right_class"] = all_r
    test_df["iso_pair_diff_class"] = all_d

    def _cat(row):
        correct = (row["y"] == row["model_prediction"])
        if row["iso_pair_diff_class"] == 1:
            return "Consensus" if correct else "Cultural Disagreement"
        if row["iso_left_class"] == 1 and row["iso_right_class"] == 1:
            return "Relational Ignorance"
        return "Structural Ignorance"

    test_df["category"] = test_df.apply(_cat, axis=1)
    return {"category_labels": test_df["category"].tolist()}


def get_sf_misalignment_flags(data, model, root_path, sub_bands, iso_models,
                              preprocess_fn: Callable, bt_estimator: Callable,
                              all_target_data=None):
    """Per-location SF misalignment {path: True/False/None}; {} if SF models absent.

    ``bt_estimator`` is evalBT.estimate_bt_scores_from_predictions (the study's dense
    estimator on human votes) -- keeps the SF residual computation identical.
    """
    if iso_models.get("iso_score_feature") is None:
        return {}
    import tensorflow as tf
    scaler_single, pca_single, iso_single = (iso_models["scaler_single"],
                                             iso_models["pca_single"], iso_models["iso_single"])
    mlp = iso_models["mlp_score_regressor"]
    scaler_residual = iso_models["scaler_residual"]
    iso_sf = iso_models["iso_score_feature"]
    single_encoder = _resolve_feature_extractor(model)

    seen, unique_locs = set(), []
    for _, row in data.iterrows():
        for col in ("left", "right"):
            p = row[col].decode("utf-8") if isinstance(row[col], bytes) else str(row[col])
            if p not in seen:
                seen.add(p); unique_locs.append(p)

    bt_data = all_target_data if all_target_data is not None else data
    true_bt = bt_estimator(bt_data, bt_data["y"].values)

    bs, loc_to_feat = 64, {}
    for i in range(0, len(unique_locs), bs):
        chunk = unique_locs[i:i + bs]
        imgs = np.array([preprocess_fn(os.path.join(root_path, p), sub_bands) for p in chunk])
        feats = single_encoder.predict_on_batch(imgs)
        if tf.is_tensor(feats):
            feats = feats.numpy()
        feats = feats.reshape(feats.shape[0], -1)
        for j, p in enumerate(chunk):
            loc_to_feat[p] = feats[j]

    locs = list(loc_to_feat.keys())
    feat_matrix = np.array([loc_to_feat[p] for p in locs])
    feat_pca = pca_single.transform(scaler_single.transform(feat_matrix))
    img_cls = iso_single.predict(feat_pca)

    flags = {loc: None for loc, c in zip(locs, img_cls) if c == -1}
    inlier_idx = [i for i, c in enumerate(img_cls) if c == 1 and locs[i] in true_bt]
    if inlier_idx:
        sf_locs = [locs[i] for i in inlier_idx]
        sf_pca = feat_pca[np.array(inlier_idx)]
        sf_bt = np.array([true_bt[loc] for loc in sf_locs])
        residuals = sf_bt - mlp.predict(sf_pca)
        res_scaled = scaler_residual.transform(residuals.reshape(-1, 1))
        sf_cls = iso_sf.predict(np.abs(res_scaled))
        for loc, c in zip(sf_locs, sf_cls):
            flags[loc] = bool(c == -1)
    for loc, c in zip(locs, img_cls):
        if c == 1 and loc not in flags:
            flags[loc] = False
    return flags


def categorize_pool_for_sampling(model, train_pool, root_path, sub_bands, iso_models,
                                 preprocess_fn: Callable, evaluate_fn: Callable,
                                 bt_estimator: Callable, all_target_data=None):
    """Orchestrate categorization -> (aoa_labels, sf_pair_flags), model already loaded.

    ``evaluate_fn`` is finetune_core.evaluate_model; ``bt_estimator`` is
    evalBT.estimate_bt_scores_from_predictions. Both injected to avoid import cycles.
    """
    from finetune_core import create_dataset  # local import to avoid cycle
    pool_ds = create_dataset(train_pool, root_path, sub_bands, preprocess_fn)
    pool_preds, _ = evaluate_fn(model, pool_ds)

    aoa_labels = categorize_with_iso(train_pool, pool_preds, iso_models, model,
                                     root_path, sub_bands, preprocess_fn)["category_labels"]
    sf_loc = get_sf_misalignment_flags(train_pool, model, root_path, sub_bands, iso_models,
                                       preprocess_fn, bt_estimator,
                                       all_target_data=all_target_data)
    sf_pair_flags = []
    for _, row in train_pool.iterrows():
        left = row["left"].decode("utf-8") if isinstance(row["left"], bytes) else str(row["left"])
        right = row["right"].decode("utf-8") if isinstance(row["right"], bytes) else str(row["right"])
        sf_pair_flags.append(sf_loc.get(left) is True or sf_loc.get(right) is True)
    return aoa_labels, sf_pair_flags
