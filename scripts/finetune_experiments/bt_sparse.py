"""
bt_sparse.py
============

Scalable Bradley-Terry score estimation for city-wide livability mapping.

WHY THIS MODULE EXISTS
----------------------
The original `estimate_bt_scores_from_predictions` in ``BT_utils/evalBT.py`` builds
a *dense* design matrix of shape ``(M_pairs x N_locations)``. That is fine for the
clustering subset (a few thousand pairs over a few hundred nodes) but is impossible
at city scale:

    ACC  (N=5,523,  M~138k)  -> dense X ~   6.1 GB
    NBO  (N=14,260, M~357k)  -> dense X ~  40.7 GB
    MUM  (N=26,563, M~664k)  -> dense X ~ 141.1 GB

Every row of that matrix has *exactly two* non-zeros (+1 for the left location,
-1 for the right). So the natural representation is sparse: the same problems become
3.3 / 8.6 / 15.9 MB respectively.

WHAT THIS MODULE IS (AND IS NOT)
--------------------------------
This is a **drop-in, numerically-identical** replacement for the original estimator,
NOT a modeling change. It preserves, exactly:

  * orientation        : ``predictions`` = P(left preferred); left=+1, right=-1,
                         so a *positive* score = higher livability.
  * vote binarisation  : ``y = 1 if prediction > 0.5 else 0`` (hard vote, as in the
                         original -- probability magnitude is intentionally discarded).
  * estimator          : sklearn LogisticRegression, ``fit_intercept=False``,
                         L2 penalty at ``C=1.0``, ``solver='lbfgs'``, ``max_iter=1000``.
                         This is what the published quantitative BT scores used, so
                         the maps must use it too, or the figures and the numbers in
                         the manuscript can be driven apart by a reviewer.
  * node ordering      : ``sorted(set(left) | set(right))`` (deterministic).
  * normalisation      : scores mean-centred to 0 (BT identifiability up to a
                         constant; L2 additionally pins the scale).

The ONLY differences from the original are (a) a sparse CSR design matrix and
(b) vectorised construction (no ``DataFrame.iterrows``). ``test_bt_parity.py``
asserts coefficient agreement with the original to < 1e-9.

Because column ordering is a pure permutation, it does not change any per-node
score; nodes are keyed by their path string on the way out regardless.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


# Estimator constants -- these MUST match evalBT.estimate_bt_scores_from_predictions
# so that city-wide maps stay consistent with the published subset BT scores.
_BT_C = 1.0
_BT_SOLVER = "lbfgs"
_BT_MAX_ITER = 1000
_VOTE_THRESHOLD = 0.5


def _as_str(x) -> str:
    """Decode bytes -> str, else str(). Mirrors the original's per-cell handling."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _decode_column(values: np.ndarray) -> np.ndarray:
    """Vectorised bytes/str normalisation of a left/right column."""
    return np.array([_as_str(v) for v in values], dtype=object)


def estimate_bt_scores_from_predictions(
    pairs_df: pd.DataFrame,
    predictions: Sequence[float],
    *,
    C: float = _BT_C,
    max_iter: int = _BT_MAX_ITER,
    solver: str = _BT_SOLVER,
    vote_threshold: float = _VOTE_THRESHOLD,
) -> Dict[str, float]:
    """Estimate per-location Bradley-Terry scores from pairwise model predictions.

    Drop-in replacement for ``BT_utils.evalBT.estimate_bt_scores_from_predictions``
    that uses a sparse design matrix so it scales to full-city pair sets.

    Parameters
    ----------
    pairs_df : DataFrame
        Must contain ``left`` and ``right`` columns identifying the two locations
        in each comparison. Values may be bytes or str (image *paths*, in the
        city-wide use case). A ``y`` column, if present, is ignored -- votes come
        from ``predictions``.
    predictions : array-like of float, length == len(pairs_df)
        P(left preferred) for each pair, positionally aligned with ``pairs_df``.
        Values are hard-thresholded at ``vote_threshold`` (default 0.5), exactly
        as in the original estimator.
    C, max_iter, solver, vote_threshold :
        Exposed for experimentation only. **Leave at defaults for parity with the
        published BT scores.** Changing ``C`` changes the L2 shrinkage and makes
        maps non-comparable to the manuscript's numbers.

    Returns
    -------
    dict : {location_str : bt_score_float}
        Mean-centred BT scores. Positive = higher livability.

    Raises
    ------
    ValueError
        If lengths mismatch, the frame is empty, or the binarised votes contain
        only a single class (degenerate -- LogisticRegression needs two classes;
        this usually signals a broken model that always predicts one side).
    """
    n_pairs = len(pairs_df)
    predictions = np.asarray(predictions, dtype=float).ravel()

    if n_pairs == 0:
        raise ValueError("pairs_df is empty; nothing to estimate.")
    if predictions.shape[0] != n_pairs:
        raise ValueError(
            f"predictions length ({predictions.shape[0]}) does not match "
            f"number of pairs ({n_pairs})."
        )

    # --- normalise left/right to strings, positionally (matches original) -------
    left = _decode_column(pairs_df["left"].values)
    right = _decode_column(pairs_df["right"].values)

    # --- deterministic node ordering (identical to original) --------------------
    nodes = sorted(set(left.tolist()) | set(right.tolist()))
    n_locs = len(nodes)
    loc_to_idx = {loc: i for i, loc in enumerate(nodes)}

    left_idx = np.fromiter((loc_to_idx[x] for x in left), dtype=np.int64, count=n_pairs)
    right_idx = np.fromiter((loc_to_idx[x] for x in right), dtype=np.int64, count=n_pairs)

    # --- binarise predictions into hard votes (identical to original) -----------
    y = (predictions > vote_threshold).astype(np.int64)
    n_classes = np.unique(y).size
    if n_classes < 2:
        raise ValueError(
            "Binarised votes contain a single class "
            f"(all {'left' if y[0] == 1 else 'right'} wins); "
            "LogisticRegression requires two classes. This typically means the "
            "model collapsed to always predicting one side over this pair set."
        )

    # --- sparse design matrix: 2 non-zeros per row (+1 left, -1 right) ----------
    rows = np.repeat(np.arange(n_pairs, dtype=np.int64), 2)
    cols = np.empty(2 * n_pairs, dtype=np.int64)
    cols[0::2] = left_idx
    cols[1::2] = right_idx
    vals = np.empty(2 * n_pairs, dtype=np.float64)
    vals[0::2] = 1.0
    vals[1::2] = -1.0
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pairs, n_locs))

    # --- fit BT (== L2 logistic regression on the pairwise design) --------------
    model = LogisticRegression(
        fit_intercept=False,
        C=C,
        max_iter=max_iter,
        solver=solver,
    )
    model.fit(X, y)

    coef = model.coef_[0].astype(np.float64)
    coef -= coef.mean()  # mean-centre for BT identifiability

    return {loc: float(coef[i]) for loc, i in loc_to_idx.items()}


def bt_scores_to_frame(
    bt_scores: Dict[str, float],
    image_id_fn=None,
) -> pd.DataFrame:
    """Convert a {path: score} dict into a tidy DataFrame.

    Parameters
    ----------
    bt_scores : dict
        Output of :func:`estimate_bt_scores_from_predictions`.
    image_id_fn : callable, optional
        Maps a path string to an image/cell id. If given, an ``image_id`` column
        is added (used downstream to join to PARTIMAP_I geometry). If omitted,
        only ``path`` and ``bt_score`` are returned.

    Returns
    -------
    DataFrame with columns [path, (image_id), bt_score], sorted by path.
    """
    rows = []
    for path, score in bt_scores.items():
        rec = {"path": path, "bt_score": score}
        if image_id_fn is not None:
            rec["image_id"] = image_id_fn(path)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    cols = ["path"] + (["image_id"] if image_id_fn is not None else []) + ["bt_score"]
    return df[cols]
