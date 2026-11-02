"""
dyadic_inference.py  (review round 1, R1-M3)
============================================

Inference for directed source->target city pairs, where every city appears
many times as both source and target, so the pairs are not independent.

  qap_spearman       exact QAP permutation test for a rank correlation
  mrqap_dsp          multiple-regression QAP (Dekker et al. 2007, double
                     semi-partialling), exact over all city relabellings
  city_bootstrap     resample cities (not pairs) -> percentile intervals
  leave_one_city_out refit with every pair involving one city removed
  crossed_effects    mixed model with crossed source and target random intercepts

QAP permutes city identities jointly on rows and columns of the source x target
matrix, which keeps the dependence structure of the pairs intact. With 7 cities
there are 7! = 5,040 relabellings, so the tests are exact rather than sampled.
QAP needs a complete square design (every off-diagonal pair present).

All functions take a long DataFrame with 'source' and 'target' columns.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ===========================================================================
# Helpers
# ===========================================================================
def _cities(df: pd.DataFrame) -> List[str]:
    return sorted(set(df["source"]) | set(df["target"]))


def check_square(df: pd.DataFrame) -> None:
    """QAP requires every ordered pair of distinct cities exactly once."""
    c = _cities(df)
    want = {(s, t) for s in c for t in c if s != t}
    have = list(zip(df["source"], df["target"]))
    if set(have) != want or len(have) != len(want):
        raise ValueError(f"QAP needs all {len(want)} ordered pairs once; got {len(have)} "
                         f"rows, {len(want - set(have))} missing.")


def _relabelled(df: pd.DataFrame, col: str, perm: Dict[str, str]) -> np.ndarray:
    """col value of pair (perm[s], perm[t]) for every row (s, t)."""
    lookup = dict(zip(zip(df["source"], df["target"]), df[col]))
    return np.array([lookup[(perm[s], perm[t])] for s, t in zip(df["source"], df["target"])])


def all_relabellings(cities: Sequence[str]):
    for p in permutations(cities):
        yield dict(zip(cities, p))


def ols(X: np.ndarray, y: np.ndarray):
    """OLS with intercept already in X. Returns (beta, t, r2)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, beta / se, r2


def _design(df, xs):
    return np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])


# ===========================================================================
# QAP tests
# ===========================================================================
def qap_spearman(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    """Spearman rho between two dyadic variables with an exact QAP p-value."""
    check_square(df)
    xv = df[x].to_numpy(float)
    rho = spearmanr(xv, df[y]).correlation
    null = np.array([spearmanr(xv, _relabelled(df, y, p)).correlation
                     for p in all_relabellings(_cities(df))])
    return {"rho": rho, "p_qap": float(np.mean(np.abs(null) >= abs(rho) - 1e-12)),
            "p_naive": spearmanr(xv, df[y]).pvalue, "n_perm": len(null)}


def mrqap_dsp(df: pd.DataFrame, y: str, xs: Sequence[str]) -> pd.DataFrame:
    """OLS coefficients with exact MRQAP (double semi-partialling) p-values.

    For each predictor k, X_k is residualised on the other predictors; the
    residual matrix is relabelled under every city permutation and the t of the
    permuted residual is compared with the observed t (equal to X_k's t in the
    full model, by Frisch-Waugh-Lovell).
    """
    check_square(df)
    yv = df[y].to_numpy(float)
    beta, t_obs, r2 = ols(_design(df, xs), yv)
    perms = list(all_relabellings(_cities(df)))
    rows = []
    for k, xk in enumerate(xs):
        others = [x for x in xs if x != xk]
        Z = _design(df, others)
        g, *_ = np.linalg.lstsq(Z, df[xk].to_numpy(float), rcond=None)
        tmp = df[["source", "target"]].copy()
        tmp["_e"] = df[xk].to_numpy(float) - Z @ g
        t_null = np.empty(len(perms))
        for i, p in enumerate(perms):
            e_p = _relabelled(tmp, "_e", p)
            _, t, _ = ols(np.column_stack([Z, e_p]), yv)
            t_null[i] = t[-1]
        rows.append({"term": xk, "coef": beta[k + 1], "t": t_obs[k + 1],
                     # relative tolerance: the identity relabelling reproduces t_obs only up
                     # to rounding (residualised vs. original design), and must count
                     "p_qap": float(np.mean(np.abs(t_null) >= abs(t_obs[k + 1]) * (1 - 1e-9)))})
    out = pd.DataFrame(rows)
    out.attrs["r2"] = r2
    return out


# ===========================================================================
# Resampling and sensitivity
# ===========================================================================
def city_bootstrap(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], Dict[str, float]],
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Percentile intervals from resampling cities with replacement.

    A drawn city contributes all its pairs with every other drawn city; a city
    drawn twice contributes its pairs twice. Pairs of a city with itself do not
    exist and are skipped. Draws with fewer than 3 distinct cities are redrawn.
    """
    rng = np.random.RandomState(seed)
    cities = _cities(df)
    by_pair = {k: g for k, g in df.groupby(["source", "target"])}
    draws = []
    while len(draws) < n_boot:
        pick = list(rng.choice(cities, size=len(cities), replace=True))
        if len(set(pick)) < 3:
            continue
        parts = [by_pair[(s, t)] for s in pick for t in pick if s != t and (s, t) in by_pair]
        try:
            draws.append(stat_fn(pd.concat(parts, ignore_index=True)))
        except (np.linalg.LinAlgError, ValueError):
            continue
    d = pd.DataFrame(draws)
    return pd.DataFrame({"est": stat_fn(df), "ci_low": d.quantile(0.025),
                         "ci_high": d.quantile(0.975)})


def leave_one_city_out(df: pd.DataFrame,
                       stat_fn: Callable[[pd.DataFrame], Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for c in _cities(df):
        sub = df[(df["source"] != c) & (df["target"] != c)]
        rows.append({"dropped": c, "n_pairs": len(sub), **stat_fn(sub)})
    return pd.DataFrame(rows)


def crossed_effects(df: pd.DataFrame, formula: str):
    """Mixed model with crossed random intercepts for source and target city."""
    import statsmodels.formula.api as smf
    d = df.assign(_all=1)
    m = smf.mixedlm(formula, d, groups="_all", re_formula="0",
                    vc_formula={"source": "0 + C(source)", "target": "0 + C(target)"})
    return m.fit(reml=True)
