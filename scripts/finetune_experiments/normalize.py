"""
normalize.py
============

Put every city map on a **common, comparable scale by normalising against the
reference (local-model) map's score distribution**, so that maps from different
sources -- and difference maps between them -- can share a colourbar and be read
directly.

PRIMARY: reference-ECDF (percentile-against-reference)
------------------------------------------------------
For a patch with BT score ``s``, its normalised value is the fraction of *reference*
patches scoring at or below ``s`` (tie-aware midrank):

    norm(s) = ( #{ref <= s}  +  #{ref < s} ) / (2 * N_ref)
            = ( #{ref < s}  +  0.5 * #{ref == s} ) / N_ref      in [0, 1]

Consequences that make this the right primary for livability *geography*:
  * the reference map, normalised against itself, is ~uniform on [0, 1];
  * any source map lands on the SAME reference-percentile axis, so a shared
    colourbar is meaningful;
  * the transform is strictly monotonic in ``s`` (ties aside), so it preserves the
    map's own ranking exactly -- Spearman vs raw BT is 1.0;
  * it is invariant to BT's arbitrary (L2-penalised) scale and robust to outliers;
  * a difference map ``source_norm - reference_norm`` lives in [-1, 1] and reads as
    "this source ranks the patch N percentile-points above/below where the local
    model ranks it."

SECONDARY: reference z-score
----------------------------
``(s - mean_ref) / std_ref`` for every map. Preserves linear spacing and shows
distributional *shift* (a source whose whole surface sits above the reference is
positive everywhere), but inherits BT's scale and is outlier-sensitive. Provided
for the distributional-shift view; ECDF is the headline.

All functions align maps by ``image_id`` (the patch key shared across every map of
a city) and never assume row order.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# Array-level transforms
# ===========================================================================
def reference_ecdf(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """Tie-aware midrank ECDF of ``values`` against ``reference_values`` -> [0, 1].

    Values below/above the entire reference range saturate to 0.0 / 1.0.
    """
    values = np.asarray(values, dtype=float)
    ref_sorted = np.sort(np.asarray(reference_values, dtype=float))
    n_ref = ref_sorted.size
    if n_ref == 0:
        raise ValueError("reference_values is empty.")
    lo = np.searchsorted(ref_sorted, values, side="left")    # #{ref < s}
    hi = np.searchsorted(ref_sorted, values, side="right")   # #{ref <= s}
    return (lo + hi) / (2.0 * n_ref)


def reference_zscore(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """Standardise ``values`` using the reference mean/std."""
    values = np.asarray(values, dtype=float)
    ref = np.asarray(reference_values, dtype=float)
    mu = ref.mean()
    sd = ref.std()
    if sd == 0:
        raise ValueError("reference std is zero; cannot z-score.")
    return (values - mu) / sd


# ===========================================================================
# Frame-level API (align by image_id)
# ===========================================================================
def normalize_map(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    value_col: str = "bt_score",
    key: str = "image_id",
    method: str = "ecdf",
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return ``frame`` with an added column normalised against ``reference_frame``.

    Parameters
    ----------
    frame, reference_frame : DataFrame
        Each has columns [key, value_col]. ``reference_frame`` supplies the basis
        distribution (all its patches). For the reference map itself, pass it as
        both arguments -> a ~uniform [0,1] band.
    method : {'ecdf', 'zscore'}
        'ecdf' (primary) or 'zscore' (secondary).
    out_col : str, optional
        Name for the new column. Defaults to ``{value_col}_{method}``.

    Returns
    -------
    A copy of ``frame`` with the normalised column added.
    """
    if method not in {"ecdf", "zscore"}:
        raise ValueError(f"method must be 'ecdf' or 'zscore', got {method!r}.")
    for name, f in [("frame", frame), ("reference_frame", reference_frame)]:
        missing = {key, value_col} - set(f.columns)
        if missing:
            raise ValueError(f"{name} missing columns {missing}.")

    out_col = out_col or f"{value_col}_{method}"
    ref_vals = reference_frame[value_col].to_numpy()
    fn = reference_ecdf if method == "ecdf" else reference_zscore

    out = frame.copy()
    out[out_col] = fn(out[value_col].to_numpy(), ref_vals)
    return out


def difference_map(
    source_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    value_col: str,
    key: str = "image_id",
    out_col: str = "difference",
    require_full_overlap: bool = False,
) -> pd.DataFrame:
    """Per-patch ``source[value_col] - reference[value_col]``, aligned by ``key``.

    Both frames should already carry the SAME normalised column (e.g. 'bt_score_ecdf')
    so the difference is on a comparable scale. Returns [key, source, reference, diff]
    for patches present in both (inner join). With ``require_full_overlap=True``,
    raises if either frame has patches the other lacks.
    """
    for name, f in [("source_frame", source_frame), ("reference_frame", reference_frame)]:
        missing = {key, value_col} - set(f.columns)
        if missing:
            raise ValueError(f"{name} missing columns {missing}.")

    s = source_frame[[key, value_col]].rename(columns={value_col: "source"})
    r = reference_frame[[key, value_col]].rename(columns={value_col: "reference"})
    how = "inner"
    merged = s.merge(r, on=key, how=how)

    if require_full_overlap:
        n_expected = len(set(source_frame[key]) | set(reference_frame[key]))
        if len(merged) != n_expected:
            raise ValueError(
                f"Incomplete overlap: {len(merged)} shared vs {n_expected} union patches."
            )

    merged[out_col] = merged["source"] - merged["reference"]
    return merged
