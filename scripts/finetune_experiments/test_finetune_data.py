"""
test_finetune_data.py
=====================

Verifies the TF-free data prep reproduces the study's split and sample exactly.
No database or TensorFlow needed -- load_pairs is injected with a synthetic loader.

Run:  python test_finetune_data.py   (exit 0 = all passed)
"""

import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import finetune_data as fd

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def _synthetic_target_data(n=2000, seed=0):
    """Mimic load_pairs output: left/right paths + y label."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "left":  [f"votes/2022/TEM/S1_200m/TEM_S1_200m_{i}.tif" for i in range(n)],
        "right": [f"votes/2022/TEM/S1_200m/TEM_S1_200m_{i+10000}.tif" for i in range(n)],
        "y": rng.randint(0, 2, size=n),
    })


def _loader_factory(df):
    def _loader(city, voter, root_path):
        return df.copy()
    return _loader


def test_split_matches_direct():
    print("Test 1: reconstruct_split matches a direct train_test_split at seed 42")
    df = _synthetic_target_data()
    loader = _loader_factory(df)

    train_pool, test_data = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)

    # replicate the source's exact three lines directly
    td = df.sample(frac=1, random_state=42).reset_index(drop=True)
    exp_train, exp_test = train_test_split(td, test_size=0.3, random_state=42)

    ok = True
    ok &= check("train_pool identical (index + rows)",
                train_pool.equals(exp_train), f"{len(train_pool)} rows")
    ok &= check("test_data identical", test_data.equals(exp_test), f"{len(test_data)} rows")
    ok &= check("70/30 split sizes", len(train_pool) == 1400 and len(test_data) == 600,
                f"{len(train_pool)}/{len(test_data)}")
    return ok


def test_split_deterministic():
    print("Test 2: split is deterministic across calls and independent of variant seed")
    df = _synthetic_target_data()
    loader = _loader_factory(df)
    t1, s1 = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)
    t2, s2 = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)
    return check("two reconstructions identical", t1.equals(t2) and s1.equals(s2))


def test_random_sample_determinism():
    print("Test 3: random_sample reproduces the study subset for a given seed")
    df = _synthetic_target_data()
    loader = _loader_factory(df)
    train_pool, _ = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)

    ok = True
    # same seed -> identical sample (row order + content)
    a = fd.random_sample(train_pool, 500, seed=42)
    b = fd.random_sample(train_pool, 500, seed=42)
    ok &= check("same seed -> identical sample", a.equals(b))

    # matches the source expression exactly
    exp = train_pool.sample(n=500, random_state=42).reset_index(drop=True)
    ok &= check("matches train_pool.sample(random_state=42)", a.equals(exp))

    # different seed -> different sample
    c = fd.random_sample(train_pool, 500, seed=43)
    ok &= check("different seed -> different sample", not a.equals(c))

    # nested sample sizes draw from the same pool deterministically
    s100 = fd.random_sample(train_pool, 100, seed=42)
    ok &= check("100-sample reproducible", s100.equals(
        train_pool.sample(n=100, random_state=42).reset_index(drop=True)))
    return ok


def test_sample_size_guard():
    print("Test 4: oversize sample raises")
    df = _synthetic_target_data(n=300)
    loader = _loader_factory(df)
    train_pool, _ = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)  # 210 rows
    try:
        fd.random_sample(train_pool, 5000, seed=42)
        return check("oversize sample raises", False)
    except ValueError:
        return check("oversize sample raises", True)


def test_all_target_data():
    print("Test 5: all_target_data = concat(train_pool, test_data)")
    df = _synthetic_target_data(n=1000)
    loader = _loader_factory(df)
    train_pool, test_data = fd.reconstruct_split("TEM", "communities", "/x", _loader=loader)
    atd = fd.all_target_data(train_pool, test_data)
    return check("concatenated length == total", len(atd) == 1000, f"{len(atd)}")


def main():
    tests = [test_split_matches_direct, test_split_deterministic,
             test_random_sample_determinism, test_sample_size_guard, test_all_target_data]
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception:
            print(f"  [{FAIL}] {t.__name__} raised:")
            traceback.print_exc()
            results.append(False)
        print()
    n = sum(results)
    print("=" * 60)
    print(f"RESULT: {n}/{len(results)} test groups passed")
    print("=" * 60)
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
