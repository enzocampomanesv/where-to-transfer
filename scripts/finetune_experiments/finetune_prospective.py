"""
finetune_prospective.py  (v3 -- leakage-free sampling, review round 1)
=====================================================================

Pure (TF-free) sampling logic for the R1-M1 / R3-m3 rerun. Three informed
variants are compared against random sampling, each using a strictly defined
information set:

  oracle_t1        -- the study's t1 sampler, but SF Bradley-Terry scores are
                      fitted on the TRAIN POOL only (the study used train+test).
                      Still uses every train-pool label, so it is reported as a
                      retrospective upper bound, not a prospective strategy.
  sat_prospective  -- sequential rounds. Round 0 is the random seed batch; after
                      that, only labels already collected are used. Label-free
                      SAT tiers (Structural / Relational / in-domain) come from the
                      source iso-forests; Cultural Disagreement and SF probabilities
                      for unlabelled pairs are kNN estimates from the labelled set.
                      Expected t1 base weights then drive the draw.
  uncertainty      -- sequential rounds, picks pairs whose current-model
                      P(left) is closest to 0.5 (standard active-learning baseline).

Everything here is numpy / sklearn only, so the sampling rules are unit-testable
without a model or tiles. The model-bound feature extraction and finetuning live
in ``run_sampling_experiment.py``.

Information-set invariants (asserted by the runner):
  * the held-out test set is never passed to any function in this module;
  * for sat_prospective, labels enter only through ``labelled_idx``.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from finetune_informed import INFORMED_BASE_WEIGHTS

# label-free tiers
TIER_STRUCTURAL = "Structural Ignorance"
TIER_RELATIONAL = "Relational Ignorance"
TIER_IN_DOMAIN = "In-domain"          # Consensus or Cultural Disagreement, label pending

DEFAULT_K = 15
PRIOR_STRENGTH = 2.0                  # pseudo-count weight of the global rate in kNN smoothing


# ===========================================================================
# Label-free SAT tiers
# ===========================================================================
def label_free_tiers(img_cls_left, img_cls_right, diff_cls) -> np.ndarray:
    """Tier per pair from iso-forest classes alone (1 = inlier, -1 = outlier).

    Mirrors categorize_with_iso: pair-diff inlier -> in-domain (CD vs Consensus
    needs the label); else both images inlier -> Relational; else Structural.
    """
    l, r, d = (np.asarray(a) for a in (img_cls_left, img_cls_right, diff_cls))
    return np.where(d == 1, TIER_IN_DOMAIN,
                    np.where((l == 1) & (r == 1), TIER_RELATIONAL, TIER_STRUCTURAL))


def oracle_categories(tiers, y, source_pred) -> np.ndarray:
    """Full SAT category given labels (used for the oracle and for reporting)."""
    tiers = np.asarray(tiers)
    correct = np.asarray(y).astype(int) == np.asarray(source_pred).astype(int)
    return np.where(tiers == TIER_IN_DOMAIN,
                    np.where(correct, "Consensus", "Cultural Disagreement"),
                    tiers)


# ===========================================================================
# kNN propagation of label-derived rates to unlabelled items
# ===========================================================================
def knn_rate(query_x, ref_x, ref_hits, k=DEFAULT_K, prior=None,
             prior_strength=PRIOR_STRENGTH) -> np.ndarray:
    """Smoothed local hit rate: (hits_in_kNN + s*prior) / (k + s).

    Falls back to the prior everywhere when there are no reference items.
    """
    query_x = np.asarray(query_x, dtype=float)
    ref_hits = np.asarray(ref_hits, dtype=float)
    if prior is None:
        prior = float(ref_hits.mean()) if len(ref_hits) else 0.0
    if len(ref_hits) == 0 or len(query_x) == 0:
        return np.full(len(query_x), prior)
    k_eff = min(k, len(ref_hits))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(np.asarray(ref_x, dtype=float))
    _, idx = nn.kneighbors(query_x)
    hits = ref_hits[idx].sum(axis=1)
    return (hits + prior_strength * prior) / (k_eff + prior_strength)


def estimate_cd_prob(tiers, diff_pca, source_pred, y, labelled_idx, k=DEFAULT_K):
    """P(Cultural Disagreement | in-domain) per pair, from labelled in-domain pairs.

    Labelled pairs get their observed value (0/1); unlabelled in-domain pairs get
    the kNN rate in pair-difference PCA space. Non-in-domain pairs get 0.
    ``y`` is only read at ``labelled_idx``.
    """
    tiers = np.asarray(tiers)
    n = len(tiers)
    labelled_idx = np.asarray(labelled_idx, dtype=int)
    lab_mask = np.zeros(n, bool)
    lab_mask[labelled_idx] = True

    in_dom = tiers == TIER_IN_DOMAIN
    ref = np.where(lab_mask & in_dom)[0]
    wrong = (np.asarray(y)[ref].astype(int) != np.asarray(source_pred)[ref].astype(int))

    p = np.zeros(n)
    query = np.where(in_dom & ~lab_mask)[0]
    p[query] = knn_rate(diff_pca[query], diff_pca[ref], wrong, k=k)
    p[ref] = wrong.astype(float)
    return p


# ===========================================================================
# Score-feature misalignment from labelled comparisons only
# ===========================================================================
def sf_flags_from_labelled(labelled_df, loc_index, loc_pca, loc_img_cls,
                           mlp, scaler_residual, iso_sf, bt_estimator) -> Dict[int, bool]:
    """SF flags for image-inlier locations that appear in the labelled pairs.

    Same computation as finetune_informed.get_sf_misalignment_flags, except the
    Bradley-Terry scores are fitted on ``labelled_df`` alone. Returns
    {location_row: flag}; locations not in the labelled set are absent.
    """
    if iso_sf is None or len(labelled_df) == 0:
        return {}
    bt = bt_estimator(labelled_df, labelled_df["y"].values)
    keys = [p for p in bt if p in loc_index and loc_img_cls[loc_index[p]] == 1]
    if not keys:
        return {}
    rows = np.array([loc_index[p] for p in keys])
    bt_vals = np.array([bt[p] for p in keys])
    residuals = bt_vals - mlp.predict(loc_pca[rows])
    res_scaled = scaler_residual.transform(residuals.reshape(-1, 1))
    cls = iso_sf.predict(np.abs(res_scaled))
    return {int(r): bool(c == -1) for r, c in zip(rows, cls)}


def estimate_sf_pair_prob(sf_flags: Dict[int, bool], loc_pca, loc_img_cls,
                          left_loc, right_loc, k=DEFAULT_K) -> np.ndarray:
    """P(pair is SF-misaligned) = 1 - (1-p_left)(1-p_right).

    Location probabilities: observed flag where available, else kNN rate over
    flagged locations in single-image PCA space. Image-outliers get 0 (SF is
    undefined for unfamiliar images, as in the study).
    """
    n_loc = len(loc_img_cls)
    p_loc = np.zeros(n_loc)
    inlier = np.asarray(loc_img_cls) == 1
    if sf_flags:
        ref = np.array(list(sf_flags.keys()))
        hits = np.array([sf_flags[r] for r in ref], dtype=float)
        query = np.where(inlier)[0]
        p_loc[query] = knn_rate(loc_pca[query], loc_pca[ref], hits, k=k)
        p_loc[ref] = hits
    p_loc[~inlier] = 0.0
    pl, pr = p_loc[np.asarray(left_loc)], p_loc[np.asarray(right_loc)]
    return 1.0 - (1.0 - pl) * (1.0 - pr)


# ===========================================================================
# Weights and draws
# ===========================================================================
def expected_base_weight(tiers, p_sf, p_cd, weights=INFORMED_BASE_WEIGHTS) -> np.ndarray:
    """Expected t1 base weight under estimated category probabilities.

    Same effective-tier rule as informed_sample_unified: Structural dominates;
    otherwise SF overrides; otherwise the AoA label.
    """
    tiers = np.asarray(tiers)
    w_sf, w_cd = weights["SF_Misaligned"], weights["Cultural Disagreement"]
    w_ri, w_co = weights["Relational Ignorance"], weights["Consensus"]
    w_si = weights["Structural Ignorance"]
    in_dom = w_sf * p_sf + (1 - p_sf) * (w_cd * p_cd + w_co * (1 - p_cd))
    rel = w_sf * p_sf + (1 - p_sf) * w_ri
    return np.where(tiers == TIER_STRUCTURAL, w_si,
                    np.where(tiers == TIER_RELATIONAL, rel, in_dom))


def weighted_draw(weights, available_idx, n, rng, temperature=1.0) -> np.ndarray:
    """Without-replacement draw of n items from available_idx, P ∝ w**T."""
    available_idx = np.asarray(available_idx, dtype=int)
    n = min(n, len(available_idx))
    w = np.maximum(np.asarray(weights, float)[available_idx], 1e-12)
    log_w = np.log(w) * temperature
    log_w -= log_w.max()
    prob = np.exp(log_w)
    prob /= prob.sum()
    return rng.choice(available_idx, size=n, replace=False, p=prob)


def uncertainty_pick(p_left, available_idx, n, rng) -> np.ndarray:
    """n items with P(left) closest to 0.5; random tie-break."""
    available_idx = np.asarray(available_idx, dtype=int)
    n = min(n, len(available_idx))
    margin = np.abs(np.asarray(p_left, float)[available_idx] - 0.5)
    order = np.lexsort((rng.random(len(available_idx)), margin))
    return available_idx[order[:n]]


def seed_batch(train_pool: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Positional indices of the round-0 batch.

    Identical to the random strategy's sample at the same n and seed
    (train_pool.sample(n, random_state=seed)), so all strategies coincide at the
    first sample size by construction.
    """
    chosen = train_pool.sample(n=n, random_state=seed).index
    return train_pool.index.get_indexer(chosen)


# ===========================================================================
# One sequential increment for sat_prospective
# ===========================================================================
def sat_prospective_increment(*, pool: Dict, train_pool: pd.DataFrame, labelled_idx,
                              n_new: int, rng, iso: Dict, bt_estimator: Callable,
                              k: int = DEFAULT_K, temperature: float = 1.0):
    """Pick ``n_new`` further pairs using only labels at ``labelled_idx``.

    ``pool`` holds label-free arrays from the runner: tiers, diff_pca,
    source_pred, left_loc, right_loc, loc_pca, loc_img_cls, loc_index.
    Returns (new_idx, diagnostics).
    """
    labelled_idx = np.asarray(labelled_idx, dtype=int)
    n = len(train_pool)
    y = train_pool["y"].to_numpy()

    p_cd = estimate_cd_prob(pool["tiers"], pool["diff_pca"], pool["source_pred"],
                            y, labelled_idx, k=k)
    sf_flags = sf_flags_from_labelled(
        train_pool.iloc[labelled_idx], pool["loc_index"], pool["loc_pca"],
        pool["loc_img_cls"], iso.get("mlp_score_regressor"),
        iso.get("scaler_residual"), iso.get("iso_score_feature"), bt_estimator)
    p_sf = estimate_sf_pair_prob(sf_flags, pool["loc_pca"], pool["loc_img_cls"],
                                 pool["left_loc"], pool["right_loc"], k=k)

    w = expected_base_weight(pool["tiers"], p_sf, p_cd)
    available = np.setdiff1d(np.arange(n), labelled_idx)
    new_idx = weighted_draw(w, available, n_new, rng, temperature=temperature)

    in_dom_lab = labelled_idx[pool["tiers"][labelled_idx] == TIER_IN_DOMAIN]
    diag = {
        "est_cd_rate_labelled": float(p_cd[in_dom_lab].mean()) if len(in_dom_lab) else None,
        "est_sf_locs_flagged": int(sum(sf_flags.values())),
        "est_sf_locs_evaluated": int(len(sf_flags)),
        "est_mean_weight_selected": float(w[new_idx].mean()) if len(new_idx) else None,
    }
    return new_idx, diag


# ===========================================================================
# Reporting helpers (labels used for description only, never for selection)
# ===========================================================================
def labelled_diagnostics(tiers, y, source_pred, labelled_idx, left_loc, right_loc) -> Dict:
    """What a practitioner could compute from the labels collected so far."""
    labelled_idx = np.asarray(labelled_idx, dtype=int)
    t = np.asarray(tiers)[labelled_idx]
    cats = oracle_categories(t, np.asarray(y)[labelled_idx], np.asarray(source_pred)[labelled_idx])
    c = Counter(cats)
    m = max(len(labelled_idx), 1)
    acc = float((np.asarray(y)[labelled_idx].astype(int)
                 == np.asarray(source_pred)[labelled_idx].astype(int)).mean()) if len(labelled_idx) else None
    locs = set(np.asarray(left_loc)[labelled_idx]) | set(np.asarray(right_loc)[labelled_idx])
    return {
        "lab_source_accuracy": acc,
        "lab_consensus": c.get("Consensus", 0) / m,
        "lab_cultural_disagreement": c.get("Cultural Disagreement", 0) / m,
        "lab_relational_ignorance": c.get(TIER_RELATIONAL, 0) / m,
        "lab_structural_ignorance": c.get(TIER_STRUCTURAL, 0) / m,
        "n_unique_locations": len(locs),
    }


def selection_breakdown(categories: Sequence[str], idx) -> Dict[str, int]:
    """Counts of TRUE categories among selected pairs (post hoc, for reporting)."""
    c = Counter(np.asarray(categories)[np.asarray(idx, dtype=int)])
    return {
        "sel_consensus": c.get("Consensus", 0),
        "sel_cultural": c.get("Cultural Disagreement", 0),
        "sel_relational": c.get(TIER_RELATIONAL, 0),
        "sel_structural": c.get(TIER_STRUCTURAL, 0),
    }
