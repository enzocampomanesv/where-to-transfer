"""
iso_sensitivity.py  (review round 1: R1-M7, Tier 1)
===================================================

Sensitivity of the SAT categories, and of the zero-shot ALR result built on
them, to the isolation-forest cutoffs, plus the pair-orientation check.

Input is the score cache written on CRIB by dump_iso_scores.py. Each forest
flags a point when its anomaly score is below a cutoff:

  'auto' (the study)   cutoff = -0.5 for Image_IF and Pair_IF
  contamination c      cutoff = c-quantile of the forest's scores on its own
                       training data (all source images / all source pairs)

Category rule (categorize_with_iso): Pair_IF inlier -> Consensus if the source
predicts the pair correctly, else Preference Disagreement (PD; "cultural" in the
code); Pair_IF outlier -> Relational Ignorance if both images are Image_IF
inliers, else Structural Ignorance. Shares are computed on each target's test
split, as in the paper.

  python iso_sensitivity.py --scores-zip ../../results/iso_scores.zip \
      --results-zip ../../results/random__layerwise_lr1e-04.zip --out-dir ../../results/review_stats

Writes iso_sensitivity_alr.csv, iso_sensitivity_shares.csv, iso_orientation.csv.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

import dyadic_inference as di
from alr_inference import CITIES, SHARES, TERMS, add_alr, alr_coefs, load_zero_shot

AUTO_CUTOFF = -0.5
GRID = ["auto", 0.02, 0.05, 0.10, 0.20]


def _read(z, src, name):
    return pd.read_csv(io.BytesIO(z.read(f"iso_scores/{src}/{name}")), compression="gzip")


def load_source(z, src):
    """Test pairs of every target with the scores the category rule needs,
    and the source's own reference scores for placing quantile cutoffs."""
    pairs, locs = _read(z, src, "pairs.csv.gz"), _read(z, src, "locations.csv.gz")
    single = locs.set_index(["city", "location"])["single_score"]
    test = pairs[pairs["split"] == "test"].copy()
    test["single_l"] = single.reindex(pd.MultiIndex.from_arrays([test["city"], test["left"]])).to_numpy()
    test["single_r"] = single.reindex(pd.MultiIndex.from_arrays([test["city"], test["right"]])).to_numpy()
    test["correct"] = test["y"].astype(int) == (test["src_prob"] > 0.5).astype(int)
    ref = {"single": locs.loc[locs["city"] == src, "single_score"].to_numpy(),
           "diff": pairs.loc[pairs["split"] == "source_ref", "diff_score_lr"].to_numpy()}
    return test.rename(columns={"city": "target"}).assign(source=src), ref


def cutoff(setting, ref_scores):
    return AUTO_CUTOFF if setting == "auto" else float(np.quantile(ref_scores, setting))


def categorise(test, cut_single, cut_diff, diff_col="diff_score_lr"):
    pair_in = test[diff_col] >= cut_diff
    imgs_in = (test["single_l"] >= cut_single) & (test["single_r"] >= cut_single)
    return np.select([pair_in & test["correct"], pair_in & ~test["correct"], imgs_in],
                     ["consensus", "cultural_disagreement", "relational_ignorance"],
                     "structural_ignorance")


def shares(test, cats):
    t = test[["source", "target"]].assign(cat=cats, wrong=~test["correct"])
    s = pd.crosstab([t["source"], t["target"]], t["cat"], normalize="index")
    s = s.reindex(columns=SHARES, fill_value=0.0)
    # share of the source's zero-shot errors that fall in PD
    err = t[t["wrong"]].groupby(["source", "target"])["cat"].apply(
        lambda c: (c == "cultural_disagreement").mean()).rename("pd_share_of_errors")
    return s.join(err).reset_index()


def alr_summary(sh, zs, n_boot):
    df = add_alr(sh.merge(zs[["source", "target", "spearman"]], on=["source", "target"]))
    mr = di.mrqap_dsp(df, "spearman", list(TERMS)).set_index("term")
    boot = di.city_bootstrap(df, alr_coefs, n_boot=n_boot, seed=0)
    return {"coef_pd": mr.loc["alr_cd", "coef"], "p_qap_pd": mr.loc["alr_cd", "p_qap"],
            "ci_low_pd": boot.loc["alr_cd", "ci_low"], "ci_high_pd": boot.loc["alr_cd", "ci_high"],
            "coef_ri": mr.loc["alr_ri", "coef"], "coef_si": mr.loc["alr_si", "coef"],
            "r2": mr.attrs["r2"]}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    z = zipfile.ZipFile(args.scores_zip)
    loaded = {s: load_source(z, s) for s in CITIES}
    test = pd.concat([t for t, _ in loaded.values()], ignore_index=True)
    zs = load_zero_shot(args.results_zip)

    # what 'auto' means as a contamination: share of the training data it flags
    auto_rate = pd.DataFrame({s: {f: float((r[f] < AUTO_CUTOFF).mean()) for f in r}
                              for s, (_, r) in loaded.items()}).T
    print("Share of each source's training data flagged at the study cutoff (-0.5):")
    print(auto_rate.round(3).to_string())

    def cuts(img_setting, pair_setting):
        cs = {s: (cutoff(img_setting, r["single"]), cutoff(pair_setting, r["diff"]))
              for s, (_, r) in loaded.items()}
        return test["source"].map(lambda s: cs[s][0]), test["source"].map(lambda s: cs[s][1])

    # 1. the baseline must reproduce the study's zero-shot shares
    base = shares(test, categorise(test, *cuts("auto", "auto")))
    chk = base.merge(zs, on=["source", "target"], suffixes=("", "_study"))
    max_dev = max((chk[c] - chk[f"{c}_study"]).abs().max() for c in SHARES)
    print(f"\nBaseline vs study zero-shot shares: max |difference| = {max_dev:.2e} over {len(chk)} pairs")

    # 2. one-at-a-time sweep: image forest, pair forest, both together
    rows, share_rows = [], []
    for varied in ["image", "pair", "both"]:
        for c in GRID:
            if c == "auto" and varied != "image":
                continue   # the baseline is the same row for all three
            img = c if varied in ("image", "both") else "auto"
            pr = c if varied in ("pair", "both") else "auto"
            sh = shares(test, categorise(test, *cuts(img, pr)))
            label = "baseline" if c == "auto" else varied
            share_rows.append(sh.assign(varied=label, setting=str(c)))
            rows.append({"varied": label, "setting": str(c),
                         **{f"mean_{k}": sh[k].mean() for k in SHARES + ["pd_share_of_errors"]},
                         **alr_summary(sh, zs, args.n_boot)})
            print(f"  {label:8s} {str(c):5s} PD coef {rows[-1]['coef_pd']:+.3f} "
                  f"p_QAP {rows[-1]['p_qap_pd']:.4f}")
    alr = pd.DataFrame(rows)

    # 3. orientation: Pair_IF on f(left)-f(right) (study) vs f(right)-f(left)
    il, pl = cuts("auto", "auto")
    flag_lr, flag_rl = test["diff_score_lr"] < pl, test["diff_score_rl"] < pl
    orient = (test.assign(flag_lr=flag_lr, flag_rl=flag_rl, flip=flag_lr != flag_rl)
              .groupby(["source", "target"])[["flag_lr", "flag_rl", "flip"]].mean().reset_index())
    print(f"\nPair_IF flag changes when the pair is reversed: {orient['flip'].mean():.3%} of test pairs "
          f"(max {orient['flip'].max():.3%}); flagged share {orient['flag_lr'].mean():.3%} vs "
          f"{orient['flag_rl'].mean():.3%}")
    for name, col in [("reversed", "diff_score_rl")]:
        sh = shares(test, categorise(test, il, pl, diff_col=col))
        r = alr_summary(sh, zs, args.n_boot)
        print(f"  ALR with {name} orientation: PD coef {r['coef_pd']:+.3f}, p_QAP {r['p_qap_pd']:.4f}")
        alr = pd.concat([alr, pd.DataFrame([{"varied": "orientation", "setting": name,
                                             **{f"mean_{k}": sh[k].mean() for k in SHARES + ["pd_share_of_errors"]},
                                             **r}])], ignore_index=True)

    alr.to_csv(os.path.join(args.out_dir, "iso_sensitivity_alr.csv"), index=False)
    pd.concat(share_rows).to_csv(os.path.join(args.out_dir, "iso_sensitivity_shares.csv"), index=False)
    orient.to_csv(os.path.join(args.out_dir, "iso_orientation.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n" + alr.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-zip", default="../../results/iso_scores.zip")
    ap.add_argument("--results-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    ap.add_argument("--n-boot", type=int, default=1000)
    main(ap.parse_args())
