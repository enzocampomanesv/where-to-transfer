"""
citywide_score.py
=================

Run a (two-tower pairwise) model over a frozen city-wide pair set and produce
per-patch Bradley-Terry livability scores, routed through the sparse BT estimator
in ``bt_sparse``.

THE ONE IDEA: preprocess each unique patch ONCE.
------------------------------------------------
A naive pass (as in ``create_dataset``) re-reads and re-preprocesses each ``.tif``
once per pair it appears in. At mean degree ~50 that is ~50 reads per tile:

    MUM: 26,563 patches x ~50  ~=  6.9M image reads / decodes  (naive)
    MUM: 26,563 unique reads               (this module)

So we build an in-RAM cache of preprocessed patch arrays (for MUM ~85 MB at
20x20x2 float32), then assemble each model batch by *indexing* into that cache.
Because ``load_and_preprocess_image`` is deterministic (GDAL read + fixed crop +
transpose, no randomness), the cache is a **pure optimisation** -- it cannot change
any prediction. ``test_citywide_score.py`` asserts cached vs naive probabilities
are bitwise identical.

PARITY IS BY CONSTRUCTION
-------------------------
The preprocessing function and sub-band lookup are *injected*. On your machine you
pass the project's own ``load_and_preprocess_image`` and ``get_sub_bands``, so each
per-image array is identical to what training/``create_dataset`` produced -- this
module never reimplements preprocessing. The prob-extraction logic mirrors
``evaluate_model`` exactly (flattened sigmoid when ``shape[1]==1``, else softmax
``[:,1]``), so ``probs`` = P(left preferred), matching the orientation the BT
estimator expects.

The single parity checkpoint to confirm against ``create_dataset`` is the batch
dtype used to stack arrays for the model (default float32); see ``score_city``.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from bt_sparse import estimate_bt_scores_from_predictions, bt_scores_to_frame


# ---------------------------------------------------------------------------
# Defaults that mirror the project (pure-python; safe to import anywhere)
# ---------------------------------------------------------------------------
def default_get_sub_bands(d_src: str):
    """Verbatim copy of the project's get_sub_bands (no external deps)."""
    n_width = 10
    n_height = 10
    if d_src == "RGBN":
        sub_bands = [0, 1, 2, 6]
    elif d_src == "RGB":
        sub_bands = [0, 1, 2]
    elif d_src == "RGN":
        sub_bands = [0, 1, 6]
    elif d_src == "all":
        sub_bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif d_src == "S1":
        sub_bands = [0, 1]
    elif d_src == "wv3":
        sub_bands = [0, 1, 2, 3]
        n_width = 50
        n_height = 50
    elif d_src == "S1_200m":
        sub_bands = [0, 1]
        n_width = 20
        n_height = 20
    else:
        raise ValueError(f"Unknown d_src: {d_src!r}")
    n_channels = len(sub_bands)
    return sub_bands, n_width, n_height, n_channels


def default_extract_image_id(path: str):
    """Path -> integer patch id (e.g. ACC_S1_200m_33230.tif -> 33230)."""
    base = os.path.basename(str(path))
    m = re.search(r"(\d+)\.", base)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not extract image id from path: {path!r}")


def _to_numpy(pred):
    """EagerTensor / tensor / array -> numpy, without a hard TF import."""
    if hasattr(pred, "numpy"):
        return pred.numpy()
    return np.asarray(pred)


def _extract_p_left(pred) -> np.ndarray:
    """Mirror evaluate_model's prob extraction: P(left preferred)."""
    pred = _to_numpy(pred)
    if pred.ndim == 1:
        return pred.astype(np.float64)          # already a flat prob vector
    if pred.shape[1] == 1:
        return pred.reshape(-1).astype(np.float64)   # sigmoid head
    return pred[:, 1].astype(np.float64)             # softmax head -> P(class 1)=P(left)


# ---------------------------------------------------------------------------
# Patch cache
# ---------------------------------------------------------------------------
def build_patch_cache(
    unique_paths,
    sub_bands,
    preprocess_fn: Callable,
    norm_data: bool,
    expected_shape: Optional[Tuple[int, int, int]],
    dtype=np.float32,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Preprocess each unique patch once into a contiguous array.

    Returns (A, path_to_row) where A has shape (U, H, W, C) and path_to_row maps
    each path to its row in A. Validates that every patch has ``expected_shape``
    (if given), raising a clear error naming the first offender -- a mismatch
    almost always means a corrupt/short tile or a wrong d_src.
    """
    arrays = []
    path_to_row: Dict[str, int] = {}
    for i, p in enumerate(unique_paths):
        arr = preprocess_fn(p, sub_bands, norm_data)
        arr = np.asarray(arr, dtype=dtype)
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(
                f"Patch {p!r} has shape {arr.shape}, expected {expected_shape}. "
                f"Check d_src / tile integrity."
            )
        arrays.append(arr)
        path_to_row[str(p)] = i
    A = np.stack(arrays, axis=0)
    return A, path_to_row


def score_pairs_with_cache(model, left_idx, right_idx, A, batch_size: int = 256,
                           progress: bool = False) -> np.ndarray:
    """Run the model over pairs given a prebuilt patch cache, returning P(left).

    ``A`` is the (U, H, W, C) cache; ``left_idx``/``right_idx`` index its rows.
    Shared by ``score_city`` and by callers (e.g. the degree-calibration script)
    that build one cache and reuse it across many pair sets / graphs. Contains no
    caching policy of its own -- it is pure gather+predict, so it cannot change any
    result relative to the naive path.
    """
    n_pairs = int(left_idx.shape[0])
    probs = np.empty(n_pairs, dtype=np.float64)
    n_batches = (n_pairs + batch_size - 1) // batch_size
    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_pairs)
        lb = A[left_idx[s:e]]
        rb = A[right_idx[s:e]]
        pred = model.predict_on_batch([lb, rb])
        probs[s:e] = _extract_p_left(pred)
        if progress and (b % 200 == 0 or b == n_batches - 1):
            print(f"  batch {b + 1}/{n_batches}", end="\r")
    if progress:
        print()
    return probs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def score_city(
    model,
    pairs_df: pd.DataFrame,
    *,
    d_src: str = "S1_200m",
    preprocess_fn: Callable,                      # REQUIRED: project's load_and_preprocess_image
    get_sub_bands_fn: Callable = default_get_sub_bands,
    image_id_fn: Callable = default_extract_image_id,
    norm_data: bool = False,
    batch_size: int = 256,
    dtype=np.float32,                             # parity checkpoint vs create_dataset
    validate_shape: bool = True,
    progress: bool = True,
) -> Dict[str, object]:
    """Score a full city: cached inference over ``pairs_df`` -> per-patch BT scores.

    Parameters
    ----------
    model : object with ``predict_on_batch([left_batch, right_batch])``
        The loaded (two-tower) ensembleVoter. Called exactly as ``evaluate_model``
        calls it; prob extraction mirrors ``evaluate_model`` (P(left preferred)).
    pairs_df : DataFrame
        Frozen pair set from ``pairgraph`` with ``left``/``right`` = image paths.
    preprocess_fn : callable(path, sub_bands, norm_data) -> HxWxC array
        Pass the project's ``load_and_preprocess_image`` for exact training parity.
    d_src, norm_data, dtype :
        Must match how ``create_dataset`` built training/eval batches. ``dtype`` is
        the one thing not fully determined by the functions you shared -- default
        float32; change only if ``create_dataset`` stacked a different dtype.

    Returns
    -------
    dict with:
        'bt_scores'  : {path: bt_score}
        'frame'      : DataFrame [path, image_id, bt_score]
        'probs'      : np.ndarray P(left) aligned to pairs_df rows
        'n_patches'  : unique patches scored
        'n_pairs'    : comparisons evaluated
    """
    if preprocess_fn is None:
        raise ValueError(
            "preprocess_fn is required -- pass the project's load_and_preprocess_image "
            "so preprocessing matches training exactly."
        )
    if not {"left", "right"}.issubset(pairs_df.columns):
        raise ValueError("pairs_df must have 'left' and 'right' columns (image paths).")

    sub_bands, n_w, n_h, n_c = get_sub_bands_fn(d_src)
    expected_shape = (n_h, n_w, n_c) if validate_shape else None

    left = pairs_df["left"].astype(str).to_numpy()
    right = pairs_df["right"].astype(str).to_numpy()
    n_pairs = left.shape[0]

    # --- unique patches, preprocessed once ------------------------------------
    unique_paths = pd.unique(np.concatenate([left, right]))
    if progress:
        print(f"[score_city] caching {len(unique_paths)} unique patches "
              f"for {n_pairs} pairs ...")
    A, path_to_row = build_patch_cache(
        unique_paths, sub_bands, preprocess_fn, norm_data, expected_shape, dtype
    )

    left_idx = np.fromiter((path_to_row[p] for p in left), dtype=np.int64, count=n_pairs)
    right_idx = np.fromiter((path_to_row[p] for p in right), dtype=np.int64, count=n_pairs)

    # --- batched inference (gather from cache; never re-read tiles) ------------
    probs = score_pairs_with_cache(model, left_idx, right_idx, A,
                                   batch_size=batch_size, progress=progress)

    # --- BT scores via the sparse estimator (Module 1) ------------------------
    bt_scores = estimate_bt_scores_from_predictions(pairs_df, probs)
    frame = bt_scores_to_frame(bt_scores, image_id_fn)

    return {
        "bt_scores": bt_scores,
        "frame": frame,
        "probs": probs,
        "n_patches": int(len(unique_paths)),
        "n_pairs": int(n_pairs),
    }


# ---------------------------------------------------------------------------
# Reference (naive) scorer -- used only to PROVE the cache changes nothing.
# Re-preprocesses every pair independently. Do not use in production.
# ---------------------------------------------------------------------------
def score_city_naive(
    model,
    pairs_df: pd.DataFrame,
    *,
    d_src: str = "S1_200m",
    preprocess_fn: Callable,
    get_sub_bands_fn: Callable = default_get_sub_bands,
    norm_data: bool = False,
    batch_size: int = 256,
    dtype=np.float32,
) -> np.ndarray:
    """Return P(left) per pair, re-preprocessing each pair's tiles every time."""
    sub_bands, _, _, _ = get_sub_bands_fn(d_src)
    left = pairs_df["left"].astype(str).to_numpy()
    right = pairs_df["right"].astype(str).to_numpy()
    n_pairs = left.shape[0]
    probs = np.empty(n_pairs, dtype=np.float64)
    n_batches = (n_pairs + batch_size - 1) // batch_size
    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_pairs)
        lb = np.stack([np.asarray(preprocess_fn(p, sub_bands, norm_data), dtype=dtype)
                       for p in left[s:e]], axis=0)
        rb = np.stack([np.asarray(preprocess_fn(p, sub_bands, norm_data), dtype=dtype)
                       for p in right[s:e]], axis=0)
        pred = model.predict_on_batch([lb, rb])
        probs[s:e] = _extract_p_left(pred)
    return probs
