"""
finetune_data.py
================

TF-free data preparation for re-finetuning, ported verbatim (in logic) from
``ACC_transferLearn_BT.py`` so the reconstructed train/test split and the random
sample are IDENTICAL to the study runs. This module has no TensorFlow dependency,
so its determinism is unit-testable without a model or tiles.

Key fidelity points, matched to the source script:
  * The split is fixed at random_state=42 for BOTH the shuffle and the
    train_test_split (source lines 2026-2028) -- it does NOT depend on the
    per-variant finetune seed. The variant seed only drives the sample within
    the train pool.
  * load_pairs reads ``{root}/votes/db/{city}/{city}_voterAI.db`` table
    ``{city}_{voter}_votes`` (source lines 787-792).
  * The random sample is ``train_pool.sample(n=sample_size, random_state=seed)
    .reset_index(drop=True)`` (source random branch).
"""

from __future__ import annotations

import sqlite3
from typing import Callable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42          # fixed in the source for shuffle + split (independent of variant seed)
TEST_SIZE = 0.3


# --- ported verbatim from the transfer script -------------------------------
def retrieve_db(db_path: str, sql: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(sql, con=con)
    con.close()
    return df


def load_pairs(city: str, voter: str, root_path: str) -> pd.DataFrame:
    db_path = f"{root_path}/votes/db/{city}/{city}_voterAI.db"
    tbl_name = f"{city}_{voter}_votes"
    sql = f"SELECT * FROM {tbl_name}"
    return retrieve_db(db_path, sql)


# --- deterministic split reconstruction -------------------------------------
def reconstruct_split(
    target_city: str,
    voter: str,
    root_path: str,
    *,
    split_seed: int = SPLIT_SEED,
    test_size: float = TEST_SIZE,
    _loader: Callable = load_pairs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce (train_pool, test_data) exactly as the study did.

    target_data = load_pairs(...); shuffle at split_seed; train_test_split at
    split_seed. Row ORDER of the returned frames matches the source, so a
    subsequent ``random_sample`` reproduces the study's finetune subset.

    ``_loader`` is injectable for testing without a database.
    """
    target_data = _loader(target_city, voter, root_path)
    target_data = target_data.sample(frac=1, random_state=split_seed).reset_index(drop=True)
    train_pool, test_data = train_test_split(
        target_data, test_size=test_size, random_state=split_seed
    )
    # Intentionally NOT reset here: keep the exact row order train_test_split
    # produced, so train_pool.sample(random_state=seed) matches the source.
    return train_pool, test_data


def all_target_data(train_pool: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """concat(train_pool, test_data) as the source builds it (line 2085)."""
    return pd.concat([train_pool, test_data]).reset_index(drop=True)


# --- random sampling (the v1 path) ------------------------------------------
def random_sample(train_pool: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """train_pool.sample(n=sample_size, random_state=seed).reset_index(drop=True).

    Verbatim from the source's random branch. Deterministic given the train_pool
    row order (fixed by reconstruct_split) and ``seed``.
    """
    if sample_size > len(train_pool):
        raise ValueError(
            f"sample_size={sample_size} exceeds train_pool size {len(train_pool)}."
        )
    return train_pool.sample(n=sample_size, random_state=seed).reset_index(drop=True)
