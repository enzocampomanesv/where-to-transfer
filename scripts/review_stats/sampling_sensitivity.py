"""
sampling_sensitivity.py  (review round 1: R1-M7, Tier 2 Stage A)
================================================================

Selection-only replay of the informed samplers under alternative settings, with
no finetuning. A setting that leaves the selection's make-up unchanged cannot
change the finetuning result, so this decides which settings (if any) need a
Stage B finetuning check.

Replayed samplers, using the rerun's own code (finetune_prospective /
finetune_informed) on cached inputs:

  oracle_t1        from the score cache (iso_scores.zip) alone
  sat_prospective  also needs pool_features.zip (dump_pool_features.py on CRIB):
                   PCA features for the kNN estimates and the SF model as arrays

Settings, varied one at a time around the study's baseline:

  image / pair     Image_IF / Pair_IF cutoff at contamination 0.02 0.05 0.10 0.20
                   (baseline 'auto' = -0.5), placed on the source's own data
  sf               Score_IF contamination 0.05 0.20 (baseline 0.10), placed on the
                   rebuilt out-of-fold training scores shifted by the per-source gap
                   so that 0.10 reproduces the saved offset_ exactly
  weights          equal (= random sampling), or one category x3 relative to the
                   study's 7 : 1 : 1 : 0.7 : 0.3
  temperature      0.5, 2 (baseline 1)
  k                kNN neighbours 5, 30 (baseline 15; sat_prospective only)

Per selection we record its make-up under the baseline categories (so settings
are compared on the same yardstick) and its overlap with the baseline selection.
Because both samplers draw at random with weights, overlap is low even for a
trivial change; the total-variation distance between the draw probabilities
(setting vs baseline, setting vs uniform) is the deterministic measure.

Baseline replays are checked against the rerun's recorded selection counts.

  python sampling_sensitivity.py --scores-zip ../../results/iso_scores.zip \
      --features-zip ../../results/pool_features.zip --runs-zip ../../results/sampling_v3.zip \
      --out-dir ../../results/review_stats

Writes sampling_sensitivity_runs.csv, sampling_sensitivity_summary.csv,
sampling_sensitivity_check.csv.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "finetune_experiments"), os.path.join(HERE, "..")]

import finetune_prospective as fp                                   # noqa: E402
from finetune_informed import INFORMED_BASE_WEIGHTS, informed_sample_unified  # noqa: E402
from dump_pool_features import mlp_from_arrays, sf_score_from_arrays  # noqa: E402
from BT_utils.evalBT import estimate_bt_scores_from_predictions               # noqa: E402

CITIES = ["ACC", "TEM", "NBO", "KIS", "BEI", "CHI", "MUM"]
SEEDS = [42, 43, 44, 45, 46]
SEED_N, SIZES = 100, [500, 1000]
AUTO = -0.5
CATS = ["Consensus", "Cultural Disagreement", "Relational Ignorance", "Structural Ignorance"]
SEL_COLS = ["sel_consensus", "sel_cultural", "sel_relational", "sel_structural"]

BASE = {"image": "auto", "pair": "auto", "sf": 0.10, "weights": "study", "temperature": 1.0, "k": fp.DEFAULT_K}
SETTINGS = ([("baseline", {})]
            + [(f"image={c}", {"image": c}) for c in (0.02, 0.05, 0.10, 0.20)]
            + [(f"pair={c}", {"pair": c}) for c in (0.02, 0.05, 0.10, 0.20)]
            + [(f"sf={c}", {"sf": c}) for c in (0.05, 0.20)]
            + [("weights=equal", {"weights": "equal"})]
            + [(f"weights={w}x3", {"weights": w}) for w in INFORMED_BASE_WEIGHTS]
            + [(f"T={t}", {"temperature": t}) for t in (0.5, 2.0)]
            + [(f"k={k}", {"k": k}) for k in (5, 30)])


# ===========================================================================
# Inputs
# ===========================================================================
def _csv(z, path):
    return pd.read_csv(io.BytesIO(z.read(path)), compression="gzip")


def _npz(z, path):
    return dict(np.load(io.BytesIO(z.read(path)), allow_pickle=False))


class _Mlp:
    def __init__(self, m):
        self.m = m

    def predict(self, x):
        return mlp_from_arrays(self.m, x)


class _Scaler:
    def __init__(self, m):
        self.mean, self.scale = m["res_mean"], m["res_scale"]

    def transform(self, x):
        return (np.asarray(x, float) - self.mean) / self.scale


class _ScoreIF:
    """Score_IF with a movable cutoff: predict() == -1 iff score < cutoff."""
    def __init__(self, m, cut):
        self.m, self.cut = m, cut

    def predict(self, x):
        return np.where(sf_score_from_arrays(self.m, x) < self.cut, -1, 1)


def load_source(args, src):
    z = zipfile.ZipFile(args.scores_zip)
    pairs, locs = _csv(z, f"iso_scores/{src}/pairs.csv.gz"), _csv(z, f"iso_scores/{src}/locations.csv.gz")
    forests = json.loads(z.read(f"iso_scores/{src}/forests.json"))
    ref_loc = locs[locs["city"] == src]
    ref = {"image": ref_loc["single_score"].to_numpy(),
           "pair": pairs.loc[pairs["split"] == "source_ref", "diff_score_lr"].to_numpy(),
           "sf": ref_loc["sf_score_oof"].dropna().to_numpy()}
    return pairs, locs, forests, ref


def cutoffs(setting, ref, forests):
    def q(name):
        c = setting[name]
        return AUTO if c == "auto" else float(np.quantile(ref[name], c))
    sf_off = forests["iso_score_feature"]["offset"]
    sf_gap = sf_off - float(np.quantile(ref["sf"], 0.10))
    sf_cut = sf_off if setting["sf"] == 0.10 else float(np.quantile(ref["sf"], setting["sf"])) + sf_gap
    return q("image"), q("pair"), sf_cut


@contextmanager
def base_weights(scheme):
    """Patch the study's weight dict in place (both samplers read the same object)."""
    saved = dict(INFORMED_BASE_WEIGHTS)
    if scheme == "equal":
        INFORMED_BASE_WEIGHTS.update({k: 1.0 for k in saved})
    elif scheme != "study":
        INFORMED_BASE_WEIGHTS[scheme] = 3.0 * saved[scheme]
    try:
        yield
    finally:
        INFORMED_BASE_WEIGHTS.update(saved)


# ===========================================================================
# One source -> target pair
# ===========================================================================
class PairContext:
    def __init__(self, pairs, locs, tgt, feats):
        tp = pairs[(pairs["city"] == tgt) & (pairs["split"] == "train")].reset_index(drop=True)
        self.train_pool = tp[["left", "right", "y"]].assign(_pos=np.arange(len(tp)))
        self.y = tp["y"].to_numpy().astype(int)
        self.source_pred = (tp["src_prob"] > 0.5).astype(int).to_numpy()
        self.diff_score = tp["diff_score_lr"].to_numpy()
        loc_paths = sorted(set(tp["left"]) | set(tp["right"]))
        self.loc_index = {p: i for i, p in enumerate(loc_paths)}
        lt = locs[locs["city"] == tgt].set_index("location").reindex(loc_paths)
        self.single_score = lt["single_score"].to_numpy()
        self.sf_score_pool = lt["sf_score_pool"].to_numpy()
        self.left_loc = tp["left"].map(self.loc_index).to_numpy()
        self.right_loc = tp["right"].map(self.loc_index).to_numpy()
        self.feats = feats
        if feats is not None:
            assert list(feats["loc_path"]) == loc_paths, "feature locations differ from cache"
            assert len(feats["diff_pca"]) == len(tp), "feature pairs differ from cache"

    def tiers(self, img_cut, pair_cut):
        img_cls = np.where(self.single_score >= img_cut, 1, -1)
        diff_cls = np.where(self.diff_score >= pair_cut, 1, -1)
        return img_cls, fp.label_free_tiers(img_cls[self.left_loc], img_cls[self.right_loc], diff_cls)

    def oracle_sf_pairs(self, img_cls, sf_cut):
        flag = (img_cls == 1) & (self.sf_score_pool < sf_cut)      # NaN score -> not flagged
        return flag[self.left_loc] | flag[self.right_loc]


def effective_tiers(cats, sf_pairs):
    return np.where(cats == "Structural Ignorance", "Structural Ignorance",
                    np.where(sf_pairs, "SF_Misaligned", cats))


def draw_prob(w, temperature):
    log_w = np.log(np.maximum(w, 1e-12)) * temperature
    p = np.exp(log_w - log_w.max())
    return p / p.sum()


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def prospective_weights(ctx, pool, labelled, iso, k):
    """The weights sat_prospective_increment draws with (recomputed for the TV measure)."""
    p_cd = fp.estimate_cd_prob(pool["tiers"], pool["diff_pca"], pool["source_pred"], ctx.y, labelled, k=k)
    sf_flags = fp.sf_flags_from_labelled(
        ctx.train_pool.iloc[labelled], pool["loc_index"], pool["loc_pca"], pool["loc_img_cls"],
        iso["mlp_score_regressor"], iso["scaler_residual"], iso["iso_score_feature"],
        estimate_bt_scores_from_predictions)
    p_sf = fp.estimate_sf_pair_prob(sf_flags, pool["loc_pca"], pool["loc_img_cls"],
                                    pool["left_loc"], pool["right_loc"], k=k)
    return fp.expected_base_weight(pool["tiers"], p_sf, p_cd)


def run_pair(job):
    src, tgt, args = job
    pairs, locs, forests, ref = load_source(args, src)
    feats = sf_model = None
    if args.features_zip:
        fz = zipfile.ZipFile(args.features_zip)
        feats = _npz(fz, f"pool_features/{src}/{tgt}.npz")
        sf_model = _npz(fz, f"pool_features/{src}/sf_model.npz")
    ctx = PairContext(pairs, locs, tgt, feats)
    n_pool = len(ctx.train_pool)

    # yardstick: baseline categories and baseline oracle SF flags
    b_img, b_pair, b_sf = cutoffs(BASE, ref, forests)
    b_img_cls, b_tiers = ctx.tiers(b_img, b_pair)
    b_cats = fp.oracle_categories(b_tiers, ctx.y, ctx.source_pred)
    b_eff = effective_tiers(b_cats, ctx.oracle_sf_pairs(b_img_cls, b_sf))

    strategies = ["oracle_t1"] + (["sat_prospective"] if feats is not None else [])
    rows, base_sel, base_prob = [], {}, {}
    for name, change in SETTINGS:
        setting = {**BASE, **change}
        img_cut, pair_cut, sf_cut = cutoffs(setting, ref, forests)
        img_cls, tiers = ctx.tiers(img_cut, pair_cut)
        cats = fp.oracle_categories(tiers, ctx.y, ctx.source_pred)
        sf_pairs = ctx.oracle_sf_pairs(img_cls, sf_cut)
        iso = pool = None
        if feats is not None:
            iso = {"mlp_score_regressor": _Mlp(sf_model), "scaler_residual": _Scaler(sf_model),
                   "iso_score_feature": _ScoreIF(sf_model, sf_cut)}
            pool = {"tiers": tiers, "diff_pca": feats["diff_pca"], "source_pred": ctx.source_pred,
                    "left_loc": ctx.left_loc, "right_loc": ctx.right_loc, "loc_pca": feats["loc_pca"],
                    "loc_img_cls": img_cls, "loc_index": ctx.loc_index}

        with base_weights(setting["weights"]):
            for strat in strategies:
                if strat == "oracle_t1" and name.startswith("k="):
                    continue      # the oracle has no kNN step
                for seed in SEEDS:
                    # draw probabilities of the first informed round, for the TV measure
                    if strat == "oracle_t1":
                        w = np.array([INFORMED_BASE_WEIGHTS[t] for t in effective_tiers(cats, sf_pairs)])
                        prob = draw_prob(w, setting["temperature"])
                    else:
                        lab0 = fp.seed_batch(ctx.train_pool, SEED_N, seed)
                        avail = np.setdiff1d(np.arange(n_pool), lab0)
                        w = prospective_weights(ctx, pool, lab0, iso, setting["k"])
                        prob = draw_prob(w[avail], setting["temperature"])
                    if name == "baseline":
                        base_prob[(strat, seed)] = prob
                    tv_base = tv(prob, base_prob[(strat, seed)])
                    tv_unif = tv(prob, np.full(len(prob), 1.0 / len(prob)))

                    rng = np.random.RandomState(seed)
                    labelled = np.array([], dtype=int)
                    for n in [SEED_N] + SIZES:
                        diag = {}
                        if strat == "oracle_t1":
                            df, _ = informed_sample_unified(ctx.train_pool, list(cats), list(sf_pairs),
                                                            n, seed, temperature=setting["temperature"])
                            idx = df["_pos"].to_numpy()
                        elif len(labelled) == 0:
                            labelled = idx = fp.seed_batch(ctx.train_pool, n, seed)
                        else:
                            new, diag = fp.sat_prospective_increment(
                                pool=pool, train_pool=ctx.train_pool, labelled_idx=labelled,
                                n_new=n - len(labelled), rng=rng, iso=iso,
                                bt_estimator=estimate_bt_scores_from_predictions,
                                k=setting["k"], temperature=setting["temperature"])
                            labelled = idx = np.concatenate([labelled, new])
                        sel = set(idx.tolist())
                        if name == "baseline":
                            base_sel[(strat, seed, n)] = sel
                        bsel = base_sel[(strat, seed, n)]
                        eff = pd.Series(b_eff[idx]).value_counts(normalize=True)
                        rows.append({
                            "source": src, "target": tgt, "setting": name, "sampling": strat,
                            "random_seed": seed, "sample_size": n, "pool_size": n_pool,
                            **fp.selection_breakdown(b_cats, idx),
                            "sel_sf_oracle": int((b_eff[idx] == "SF_Misaligned").sum()),
                            "own_pd": int((cats[idx] == "Cultural Disagreement").sum()),
                            "own_sf": int(sf_pairs[idx].sum()),
                            "pd_share": float((b_cats[idx] == "Cultural Disagreement").mean()),
                            "eff_sf_share": float(eff.get("SF_Misaligned", 0.0)),
                            "jaccard_vs_baseline": len(sel & bsel) / len(sel | bsel),
                            "tv_vs_baseline": tv_base, "tv_vs_uniform": tv_unif,
                            **{k: v for k, v in diag.items() if k.startswith("est_")},
                        })
    out = pd.DataFrame(rows)
    pool_pd = float((b_cats == "Cultural Disagreement").mean())
    return out.assign(pool_pd_share=pool_pd)


# ===========================================================================
# Checks and summary
# ===========================================================================
def check_against_rerun(runs, rerun_zip):
    z = zipfile.ZipFile(rerun_zip)
    rr = pd.concat([pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith("_sampling_v3.csv")])
    key = ["source", "target", "sampling", "random_seed", "sample_size"]
    rr = rr.drop_duplicates(key, keep="last")
    est = ["est_sf_locs_flagged", "est_sf_locs_evaluated"]
    base = runs[runs["setting"] == "baseline"]
    m = base.merge(rr[key + SEL_COLS + est], on=key, suffixes=("", "_rerun"))
    cols = SEL_COLS + [c for c in est if c in base.columns]
    m["match"] = np.all([(m[c].fillna(-1) == m[f"{c}_rerun"].fillna(-1)) for c in cols], axis=0)
    return m


def summarise(runs):
    g = runs.groupby(["sampling", "setting", "sample_size"])
    s = g.agg(pd_share=("pd_share", "mean"), pool_pd_share=("pool_pd_share", "mean"),
              eff_sf_share=("eff_sf_share", "mean"),
              jaccard=("jaccard_vs_baseline", "mean"),
              tv_vs_baseline=("tv_vs_baseline", "mean"), tv_vs_uniform=("tv_vs_uniform", "mean"),
              n=("pd_share", "size")).reset_index()
    order = {name: i for i, (name, _) in enumerate(SETTINGS)}
    return s.sort_values(["sampling", "sample_size", "setting"], key=lambda c: c.map(order) if c.name == "setting" else c)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    from run_sampling_experiment import _SRC_TRAIN
    jobs = [(s, t, args) for s in CITIES for t in CITIES if t not in _SRC_TRAIN.get(s, {s})]
    if args.pairs:
        keep = set(args.pairs.split(","))
        jobs = [j for j in jobs if f"{j[0]}-{j[1]}" in keep]
    with Pool(args.workers) as p:
        runs = pd.concat(p.map(run_pair, jobs), ignore_index=True)
    runs.to_csv(os.path.join(args.out_dir, "sampling_sensitivity_runs.csv"), index=False)

    chk = check_against_rerun(runs, args.runs_zip)
    chk.to_csv(os.path.join(args.out_dir, "sampling_sensitivity_check.csv"), index=False)
    print("Baseline replay vs rerun selection counts:")
    print(chk.groupby("sampling")["match"].agg(["mean", "size"]).to_string())

    summ = summarise(runs)
    summ.to_csv(os.path.join(args.out_dir, "sampling_sensitivity_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n" + summ.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-zip", default="../../results/iso_scores.zip")
    ap.add_argument("--features-zip", default=None, help="pool_features.zip; without it only oracle_t1 is replayed")
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--pairs", default=None, help="comma list like ACC-TEM,KIS-BEI (default: all 42)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    main(ap.parse_args())
