"""
Unit tests for finetune_prospective (no TF / tiles needed).

Run:  python -m pytest test_finetune_prospective.py -q
  or  python test_finetune_prospective.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import finetune_prospective as fp
from finetune_data import random_sample
from finetune_informed import INFORMED_BASE_WEIGHTS, informed_sample_unified

try:
    from BT_utils.evalBT import estimate_bt_scores_from_predictions as bt_est
except ImportError:  # local layout
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from evalBT import estimate_bt_scores_from_predictions as bt_est


# ---------------------------------------------------------------------------
# synthetic pool
# ---------------------------------------------------------------------------
def _toy_pool(n_loc=120, n_pairs=900, seed=0):
    rng = np.random.RandomState(seed)
    locs = [f"img/{i}.tif" for i in range(n_loc)]
    loc_pca = rng.normal(size=(n_loc, 4))
    loc_img_cls = np.where(rng.random(n_loc) < 0.9, 1, -1)
    left = rng.randint(0, n_loc, n_pairs)
    right = (left + 1 + rng.randint(0, n_loc - 1, n_pairs)) % n_loc
    diff_pca = loc_pca[left] - loc_pca[right]
    diff_cls = np.where(rng.random(n_pairs) < 0.8, 1, -1)
    y = rng.randint(0, 2, n_pairs)
    train_pool = pd.DataFrame({"left": [locs[i] for i in left],
                               "right": [locs[i] for i in right], "y": y})
    train_pool["_pos"] = np.arange(n_pairs)
    pool = {
        "loc_index": {p: i for i, p in enumerate(locs)}, "loc_pca": loc_pca,
        "loc_img_cls": loc_img_cls, "left_loc": left, "right_loc": right,
        "diff_pca": diff_pca,
        "tiers": fp.label_free_tiers(loc_img_cls[left], loc_img_cls[right], diff_cls),
        "source_pred": rng.randint(0, 2, n_pairs),
    }
    # stand-ins for the source's SF artifacts
    mlp = LinearRegression().fit(loc_pca, rng.normal(size=n_loc))
    res = rng.normal(size=(n_loc, 1))
    iso = {"mlp_score_regressor": mlp,
           "scaler_residual": StandardScaler().fit(res),
           "iso_score_feature": IsolationForest(contamination=0.1, random_state=0).fit(np.abs(res))}
    return train_pool, pool, iso


# ---------------------------------------------------------------------------
def test_tiers_match_categorize_with_iso_rule():
    l = np.array([1, 1, -1, 1, -1])
    r = np.array([1, 1, 1, -1, -1])
    d = np.array([1, -1, -1, -1, 1])
    t = fp.label_free_tiers(l, r, d)
    assert list(t) == [fp.TIER_IN_DOMAIN, fp.TIER_RELATIONAL, fp.TIER_STRUCTURAL,
                       fp.TIER_STRUCTURAL, fp.TIER_IN_DOMAIN]
    cats = fp.oracle_categories(t, [1, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    assert list(cats) == ["Consensus", fp.TIER_RELATIONAL, fp.TIER_STRUCTURAL,
                          fp.TIER_STRUCTURAL, "Cultural Disagreement"]


def test_seed_batch_equals_random_sample():
    train_pool, _, _ = _toy_pool()
    idx = fp.seed_batch(train_pool, 100, 43)
    rs = random_sample(train_pool, 100, 43)
    assert list(train_pool.iloc[idx]["_pos"]) == list(rs["_pos"])


def test_cd_estimate_ignores_unlabelled_labels():
    train_pool, pool, _ = _toy_pool()
    lab = fp.seed_batch(train_pool, 100, 42)
    y = train_pool["y"].to_numpy()
    y_scrambled = y.copy()
    mask = np.ones(len(y), bool); mask[lab] = False
    y_scrambled[mask] = 1 - y_scrambled[mask]
    a = fp.estimate_cd_prob(pool["tiers"], pool["diff_pca"], pool["source_pred"], y, lab)
    b = fp.estimate_cd_prob(pool["tiers"], pool["diff_pca"], pool["source_pred"], y_scrambled, lab)
    assert np.allclose(a, b)


def test_prospective_selection_ignores_unlabelled_labels():
    """Core R1-M1 property: selection depends only on labels already collected."""
    train_pool, pool, iso = _toy_pool()
    lab = fp.seed_batch(train_pool, 100, 42)
    tp2 = train_pool.copy()
    mask = np.ones(len(tp2), bool); mask[lab] = False
    tp2.loc[mask, "y"] = 1 - tp2.loc[mask, "y"]
    sel = []
    for tp in (train_pool, tp2):
        new, _ = fp.sat_prospective_increment(
            pool=pool, train_pool=tp, labelled_idx=lab, n_new=200,
            rng=np.random.RandomState(7), iso=iso, bt_estimator=bt_est)
        sel.append(np.sort(new))
    assert np.array_equal(sel[0], sel[1])


def test_increment_excludes_labelled_and_is_distinct():
    train_pool, pool, iso = _toy_pool()
    lab = fp.seed_batch(train_pool, 100, 42)
    new, diag = fp.sat_prospective_increment(
        pool=pool, train_pool=train_pool, labelled_idx=lab, n_new=400,
        rng=np.random.RandomState(1), iso=iso, bt_estimator=bt_est)
    assert len(new) == 400 and len(set(new)) == 400
    assert not set(new) & set(lab)
    assert diag["est_sf_locs_evaluated"] > 0


def test_expected_weight_reduces_to_base_weights():
    W = INFORMED_BASE_WEIGHTS
    t = np.array([fp.TIER_IN_DOMAIN, fp.TIER_IN_DOMAIN, fp.TIER_IN_DOMAIN,
                  fp.TIER_RELATIONAL, fp.TIER_STRUCTURAL])
    w = fp.expected_base_weight(t, p_sf=np.array([1, 0, 0, 0, 1.0]),
                                p_cd=np.array([0, 1, 0, 0, 0.0]))
    assert np.allclose(w, [W["SF_Misaligned"], W["Cultural Disagreement"], W["Consensus"],
                           W["Relational Ignorance"], W["Structural Ignorance"]])


def test_uncertainty_pick_closest_to_half():
    p = np.array([0.9, 0.51, 0.1, 0.48, 0.7, 0.5])
    got = fp.uncertainty_pick(p, np.arange(6), 3, np.random.RandomState(0))
    assert set(got) == {5, 1, 3}


def test_informed_sample_unified_keeps_pos_column():
    train_pool, pool, _ = _toy_pool()
    cats = fp.oracle_categories(pool["tiers"], train_pool["y"], pool["source_pred"])
    df, _ = informed_sample_unified(train_pool, list(cats), [False] * len(train_pool),
                                    200, 42, temperature=1.0)
    assert "_pos" in df and df["_pos"].is_unique and len(df) == 200


if __name__ == "__main__":
    import inspect, sys
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in tests:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(tests)} passed")
