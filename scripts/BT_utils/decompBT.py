"""
Bradley-Terry Score-Space Decompositions
=========================================
Three location-level, output-space analyses that complement the pair-level,
input-space AoA categorisation in ACC_aoaBT_utils.py.

Dependency hierarchy (no circular imports):
    ACC_evalBT_utils  ←  ACC_aoaBT_utils  ←  ACC_btDecomp_utils

Score convention
----------------
Both "true" and "estimated" BT scores are fitted from the SAME test-set pair
comparisons using estimate_bt_scores_from_predictions():
  - true scores:      fitted from ground-truth labels   (test_data['y'])
  - estimated scores: fitted from model probabilities   (test_probs)

This mirrors exactly what evaluate_model_bt_recovery() and
add_bt_metrics_to_categories() already do, keeping all scores on the same
scale and over the same locations.  The pre-computed DB scores are NOT used
here because they are derived from a different (full) set of comparisons and
are not directly comparable to scores estimated from the test-set subset.

Location keys
-------------
Both score dicts are keyed by the full decoded path string — the same
convention used throughout ACC_evalBT_utils.py (e.g.
"ACC/S1_200m/ACC_S1_200m_0042.tif").  No separate location-ID extraction
is required.

Decompositions
--------------
1. Score Residual Analysis  (per-location)
   residual_i = estimated_score_i − true_score_i
   Captures systematic mis-ranking regardless of AoA category.  A location
   can be an input-space inlier (familiar image, not flagged by AoA) yet
   still carry a large residual — ordinal displacement invisible to AoA.

2. Rank Stratum Analysis  (per-quintile)
   Divides locations into n_strata quantiles of true BT score and computes
   per-stratum Spearman and bias.  Middle strata (ambiguous cases) are
   expected to show the lowest Spearman and the least improvement from
   fine-tuning, aligning with where Relational Ignorance concentrates.

3. Score Gap × Error  (per-pair)
   Correlates the true score gap |θᵢ − θⱼ| with the per-pair BT probability
   error (reusing compute_pairwise_bt_errors from ACC_aoaBT_utils).
   Expected negative: clearer pairs should be easier.  A weak or positive
   correlation signals systematic ordinal disagreement beyond noise.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evalBT import estimate_bt_scores_from_predictions
from .aoaBT import compute_pairwise_bt_errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bt_score_decompositions(test_data, test_probs, n_strata=5):
    """
    Run all three BT score-space decompositions and return a flat dict of
    summary statistics ready to spread directly into a results row.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test-set pairs with columns 'left', 'right', 'y'.
    test_probs : array-like
        Model-predicted P(left wins) for each row of test_data.
    n_strata : int
        Number of quantile bins for the rank stratum analysis (default 5).

    Returns
    -------
    dict with 15 keys (all None on failure):

    Residual analysis (4 keys):
        bt_residual_mean          Signed mean residual; positive = model
                                  systematically over-ranks locations.
        bt_residual_mae           Mean absolute residual.
        bt_residual_std           Std of residuals (consistency).
        bt_residual_position_corr Spearman(true_score, |residual|).
                                  Positive → errors concentrate at the top
                                  of the preference distribution (affluent).
                                  Negative → errors at the bottom (deprived).

    Rank stratum analysis (2 × n_strata keys):
        bt_q{1..n}_spearman       Within-stratum Spearman correlation.
                                  q1 = most deprived, q5 = least deprived.
        bt_q{1..n}_bias           Mean (estimated − true) within stratum.

    Score gap analysis (1 key):
        bt_gap_error_corr         Spearman(|θᵢ − θⱼ|, BT_prob_error).
                                  Expected negative (clear pairs are easy).
                                  Near-zero or positive → ordinal disagreement.
    """
    null_keys = (
        ['bt_residual_mean', 'bt_residual_mae', 'bt_residual_std',
         'bt_residual_position_corr'] +
        [f'bt_q{q}_{m}' for q in range(1, n_strata + 1)
         for m in ('spearman', 'bias')] +
        ['bt_gap_error_corr']
    )
    null_result = {k: None for k in null_keys}

    try:
        # ── Fit both score dicts from the test set ────────────────────────
        # Exactly mirrors evaluate_model_bt_recovery and add_bt_metrics_to_categories.
        true_scores = estimate_bt_scores_from_predictions(
            test_data, test_data['y'].values
        )
        est_scores = estimate_bt_scores_from_predictions(
            test_data, np.asarray(test_probs, dtype=float)
        )

        # ── Align to locations present in both dicts ──────────────────────
        common_locs = sorted(set(true_scores.keys()) & set(est_scores.keys()))
        if len(common_locs) < 10:
            print(f"      [BT decomp] Only {len(common_locs)} matched locations — skipping")
            return null_result

        true_arr = np.array([true_scores[loc] for loc in common_locs])
        est_arr  = np.array([est_scores[loc]  for loc in common_locs])

        # Mean-centre both arrays so residuals and bias share a common scale
        true_arr = true_arr - true_arr.mean()
        est_arr  = est_arr  - est_arr.mean()

        results = {}

        # ── 1. Score Residual Analysis ────────────────────────────────────
        results.update(_residual_analysis(true_arr, est_arr))

        # ── 2. Rank Stratum Analysis ──────────────────────────────────────
        results.update(_stratum_analysis(true_arr, est_arr, n_strata))

        # ── 3. Score Gap × Error ──────────────────────────────────────────
        results.update(_gap_error_analysis(test_data, test_probs,
                                           true_scores, est_scores))

        return results

    except Exception as e:
        print(f"      [BT decomp error: {e}]")
        import traceback
        traceback.print_exc()
        return null_result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _residual_analysis(true_arr, est_arr):
    """
    Decomposition 1: per-location score residuals.

    Parameters
    ----------
    true_arr, est_arr : np.ndarray
        Mean-centred arrays of true and estimated BT scores for the same
        set of locations (in the same order).

    Returns
    -------
    dict with keys: bt_residual_mean, bt_residual_mae, bt_residual_std,
                    bt_residual_position_corr
    """
    residuals     = est_arr - true_arr
    abs_residuals = np.abs(residuals)

    pos_corr, _ = spearmanr(true_arr, abs_residuals)

    return {
        'bt_residual_mean':          float(residuals.mean()),
        'bt_residual_mae':           float(abs_residuals.mean()),
        'bt_residual_std':           float(residuals.std()),
        'bt_residual_position_corr': float(pos_corr) if not np.isnan(pos_corr) else None,
    }


def _stratum_analysis(true_arr, est_arr, n_strata):
    """
    Decomposition 2: per-quantile-stratum Spearman and bias.

    Parameters
    ----------
    true_arr, est_arr : np.ndarray
        Mean-centred arrays (same locations, same order).
    n_strata : int
        Number of quantile bins.

    Returns
    -------
    dict with keys: bt_q{1..n}_spearman, bt_q{1..n}_bias
        Stratum 1 = most deprived (lowest true BT score),
        Stratum n = least deprived.
    """
    try:
        stratum_labels = pd.qcut(true_arr, q=n_strata,
                                 labels=False, duplicates='drop')
    except ValueError:
        # Degenerate distribution — assign everything to one stratum
        stratum_labels = np.zeros(len(true_arr), dtype=int)

    results = {}
    for q in range(n_strata):
        mask    = stratum_labels == q
        key_sp  = f'bt_q{q + 1}_spearman'
        key_bias = f'bt_q{q + 1}_bias'

        if mask.sum() < 5:
            results[key_sp]   = None
            results[key_bias] = None
            continue

        sp, _ = spearmanr(true_arr[mask], est_arr[mask])
        results[key_sp]   = float(sp) if not np.isnan(sp) else None
        results[key_bias] = float((est_arr[mask] - true_arr[mask]).mean())

    return results


def _gap_error_analysis(test_data, test_probs, true_scores, est_scores):
    """
    Decomposition 3: true score gap vs BT probability error.

    Reuses compute_pairwise_bt_errors() from ACC_aoaBT_utils so the error
    definition is identical to what add_bt_metrics_to_categories() uses.
    The gap is derived from the same true_scores dict.

    Parameters
    ----------
    test_data : pd.DataFrame
    test_probs : array-like
        Passed through to compute_pairwise_bt_errors as estimated scores
        are already captured in est_scores.
    true_scores, est_scores : dict
        Full-path-keyed score dicts from estimate_bt_scores_from_predictions.

    Returns
    -------
    dict with key: bt_gap_error_corr
    """
    # Per-pair BT probability errors using the fitted score dicts
    pairwise_errors = compute_pairwise_bt_errors(
        test_data, true_scores, est_scores
    )

    gaps   = []
    errors = []

    for i, (_, row) in enumerate(test_data.iterrows()):
        if np.isnan(pairwise_errors[i]):
            continue

        left  = row['left']
        right = row['right']
        if isinstance(left,  bytes): left  = left.decode('utf-8')
        if isinstance(right, bytes): right = right.decode('utf-8')
        left, right = str(left), str(right)

        if left not in true_scores or right not in true_scores:
            continue

        gaps.append(abs(true_scores[left] - true_scores[right]))
        errors.append(pairwise_errors[i])

    if len(gaps) < 10:
        return {'bt_gap_error_corr': None}

    gap_corr, _ = spearmanr(gaps, errors)
    return {
        'bt_gap_error_corr': float(gap_corr) if not np.isnan(gap_corr) else None
    }