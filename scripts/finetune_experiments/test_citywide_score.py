"""
test_citywide_score.py
=======================

Proves the cached scorer's plumbing with a mock model + mock preprocessing
(no TensorFlow, no GDAL). Covers:
  1. cache == naive: cached probabilities are bitwise identical to re-reading
     every pair (the cache is a pure optimisation).
  2. orientation + BT recovery: BT scores rank-correlate with the latent strength
     baked into the mock imagery (P(left) convention preserved end-to-end).
  3. sigmoid head (shape (n,1)) and softmax head (shape (n,2)) both yield P(left).
  4. batch boundaries: ragged final batch, batch_size=1, batch_size > n_pairs.
  5. shape validation catches a malformed patch.

Run:  python test_citywide_score.py    (exit 0 = all passed)
"""

import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import citywide_score as cs


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


# ---- mock imagery: each patch id -> a 20x20x2 array encoding a latent strength
def make_strengths(n_nodes, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n_nodes)


def path_for(i):
    return f"votes/2022/ACC/S1_200m/ACC_S1_200m_{30000 + i}.tif"


def id_of(path):
    return int(path.split("_")[-1].split(".")[0]) - 30000


class MockPreprocess:
    """Deterministic 'image' whose mean encodes the node's latent strength.

    Signature matches load_and_preprocess_image(fp, sub_bands, norm_data=False).
    """
    def __init__(self, strengths):
        self.strengths = strengths
        self.calls = 0

    def __call__(self, fp, sub_bands, norm_data=False):
        self.calls += 1
        s = self.strengths[id_of(fp)]
        # constant-filled 20x20x2 patch equal to the strength (+ tiny id texture)
        arr = np.full((20, 20, 2), s, dtype=np.float64)
        return arr


class MockModel:
    """Two-tower stand-in. 'Strength' of an image = mean over pixels.
    Returns P(left) = sigmoid(mean_left - mean_right).

    head='sigmoid' -> shape (n,1); head='softmax' -> shape (n,2).
    """
    def __init__(self, head="sigmoid"):
        self.head = head

    def predict_on_batch(self, inputs):
        lb, rb = inputs
        sl = lb.reshape(lb.shape[0], -1).mean(axis=1)
        sr = rb.reshape(rb.shape[0], -1).mean(axis=1)
        p_left = 1.0 / (1.0 + np.exp(-(sl - sr)))
        if self.head == "sigmoid":
            return p_left.reshape(-1, 1)
        else:
            return np.stack([1.0 - p_left, p_left], axis=1)


def make_pairs(n_nodes, n_pairs, seed=1):
    rng = np.random.RandomState(seed)
    L = rng.randint(0, n_nodes, size=n_pairs)
    R = rng.randint(0, n_nodes, size=n_pairs)
    keep = L != R
    L, R = L[keep], R[keep]
    return pd.DataFrame({
        "left": [path_for(i) for i in L],
        "right": [path_for(i) for i in R],
        "y": 0,
    })


# ---------------------------------------------------------------------------
def test_cache_equals_naive():
    print("Test 1: cached probs are bitwise identical to naive re-read")
    strengths = make_strengths(200, seed=0)
    pairs = make_pairs(200, 5000, seed=2)
    pre = MockPreprocess(strengths)
    model = MockModel("sigmoid")

    out = cs.score_city(model, pairs, preprocess_fn=pre, progress=False)
    cached_probs = out["probs"]

    pre_naive = MockPreprocess(strengths)
    naive_probs = cs.score_city_naive(model, pairs, preprocess_fn=pre_naive)

    ok = True
    ok &= check("probs bitwise identical",
                np.array_equal(cached_probs, naive_probs),
                f"max diff {np.max(np.abs(cached_probs - naive_probs)):.2e}")
    # the whole point: cache preprocesses each unique patch once
    n_unique = len(pd.unique(np.concatenate([pairs['left'].values, pairs['right'].values])))
    ok &= check("cache preprocessed each unique patch once",
                pre.calls == n_unique, f"{pre.calls} calls for {n_unique} patches")
    ok &= check("naive preprocessed far more (~2 per pair)",
                pre_naive.calls == 2 * len(pairs), f"{pre_naive.calls} calls")
    print(f"       cache: {pre.calls} preprocess calls vs naive: {pre_naive.calls} "
          f"({pre_naive.calls / pre.calls:.1f}x fewer)")
    return ok


def test_orientation_recovery():
    print("Test 2: BT scores recover latent strength (orientation preserved)")
    strengths = make_strengths(300, seed=5)
    pairs = make_pairs(300, 12000, seed=6)
    out = cs.score_city(MockModel("sigmoid"), pairs,
                        preprocess_fn=MockPreprocess(strengths), progress=False)
    frame = out["frame"]
    est = frame.set_index("image_id")["bt_score"]
    true = pd.Series({30000 + i: strengths[i] for i in range(300)})
    common = est.index.intersection(true.index)
    rho, _ = spearmanr(est.loc[common].values, true.loc[common].values)
    return check("BT rank-correlates with latent strength (rho > 0.95)",
                 rho > 0.95, f"rho={rho:.3f}")


def test_sigmoid_vs_softmax():
    print("Test 3: sigmoid head and softmax head give identical P(left)")
    strengths = make_strengths(150, seed=7)
    pairs = make_pairs(150, 4000, seed=8)
    p_sig = cs.score_city(MockModel("sigmoid"), pairs,
                          preprocess_fn=MockPreprocess(strengths), progress=False)["probs"]
    p_soft = cs.score_city(MockModel("softmax"), pairs,
                           preprocess_fn=MockPreprocess(strengths), progress=False)["probs"]
    return check("sigmoid vs softmax probs identical",
                 np.allclose(p_sig, p_soft, atol=1e-12),
                 f"max diff {np.max(np.abs(p_sig - p_soft)):.2e}")


def test_batch_boundaries():
    print("Test 4: batch-boundary correctness")
    strengths = make_strengths(80, seed=9)
    pairs = make_pairs(80, 337, seed=10)   # prime-ish count -> ragged batches
    ref = cs.score_city(MockModel("sigmoid"), pairs,
                        preprocess_fn=MockPreprocess(strengths),
                        batch_size=256, progress=False)["probs"]
    ok = True
    for bs in [1, 7, 64, 10_000]:
        p = cs.score_city(MockModel("sigmoid"), pairs,
                          preprocess_fn=MockPreprocess(strengths),
                          batch_size=bs, progress=False)["probs"]
        ok &= check(f"batch_size={bs}: length + values match",
                    len(p) == len(pairs) and np.array_equal(p, ref))
    return ok


def test_shape_validation():
    print("Test 5: malformed patch shape is caught")
    strengths = make_strengths(30, seed=11)
    pairs = make_pairs(30, 200, seed=12)

    def bad_pre(fp, sub_bands, norm_data=False):
        if cs.default_extract_image_id(fp) == 30005:      # one broken tile
            return np.zeros((20, 19, 2))                   # wrong width
        return np.full((20, 20, 2), strengths[id_of(fp)])

    try:
        cs.score_city(MockModel("sigmoid"), pairs, preprocess_fn=bad_pre, progress=False)
        return check("malformed patch raises", False)
    except ValueError as e:
        return check("malformed patch raises", "expected" in str(e).lower())


def main():
    tests = [
        test_cache_equals_naive,
        test_orientation_recovery,
        test_sigmoid_vs_softmax,
        test_batch_boundaries,
        test_shape_validation,
    ]
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
