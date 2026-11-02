"""
source_selection.py  (review round 1: decisions 2 and 3; R2-3, R2-4, R1-m7)
==========================================================================

Can the source city be chosen before finetuning, from the seed sample alone, and
does a chosen single city do better than the multi-city aggregates?

Setting. For a target and seed, every candidate source sees the same seed batch
(the random sample of 100 target comparisons; it depends only on the target pool
and the seed). From it a practitioner can compute, per source, the source's
accuracy on the seed and its PD rate among in-domain seed pairs. Label-free
quantities (Structural share from the forests, coverage) need no seed at all.

Selection rules, applied per target x seed among the single-city sources:
  seed_accuracy    highest accuracy on the seed batch               (seed labels)
  seed_pd          lowest PD rate among in-domain seed pairs         (seed labels)
  seed500_accuracy as seed_accuracy, with a 500-comparison seed        (seed labels)
  fitted           linear model of final Spearman on seed accuracy and pool
                   Structural share, fitted on the other 6 targets   (seed labels)
  coverage         highest overlap ratio                             (label-free)
  low_structural   lowest Structural Ignorance share of the pool     (label-free)
  random           expected value of picking a source at random
  zs_spearman      highest zero-shot Spearman on the full test set   (retrospective)
  best             the best source in hindsight                      (retrospective)

Outcome: final Spearman of the chosen source after finetuning on n random
target comparisons (n = 0 is zero-shot), averaged over the rerun's five seeds so
that training noise does not enter the yardstick. Regret = best source's value
- chosen source's value, per target; summarised over the 7 targets and 5 seeds.

Decision 2: per target, the source picked by seed_accuracy vs the average tier
and continental aggregate (study random runs, 3 seeds), within target, with the
exact Wilcoxon test over the 7 targets.

  python source_selection.py --runs-zip ../../results/sampling_v3.zip \
      --study-zip ../../results/random__layerwise_lr1e-04.zip \
      --validation-zip ../../results/tier1/validation.zip --out-dir ../../results/review_stats

Writes source_selection_regret.csv, source_selection_choices.csv,
source_selection_vs_aggregates.csv.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from aggregation_inference import CITIES, source_type
from sampling_inference import load_runs

SIZES = [0, 100, 500, 1000, 5000]
SEEDS = [42, 43, 44, 45, 46]
RULES = ["seed_accuracy", "seed_pd", "seed500_accuracy", "fitted", "coverage",
         "low_structural", "random", "zs_spearman", "best"]


def load_singles(runs_zip, validation_zip, scores_zip):
    runs, _ = load_runs(runs_zip)
    r = runs[runs["sampling"] == "random"]
    # yardstick: seed-mean final Spearman per source, target, n (n = 0: zero-shot)
    final = r.groupby(["source", "target", "sample_size"])["spearman"].mean().unstack()
    final[0] = r.groupby(["source", "target"])["zero_spearman"].first()
    # what the seed batch tells a practitioner, per source, target, seed
    seed = r[r["sample_size"] == 100].set_index(["source", "target", "random_seed"])
    seed500 = r[r["sample_size"] == 500].set_index(["source", "target", "random_seed"])
    info = pd.DataFrame({
        "seed_acc": seed["lab_source_accuracy"],
        "seed_pd": seed["lab_cultural_disagreement"] / (seed["lab_cultural_disagreement"] + seed["lab_consensus"]),
        "seed500_acc": seed500["lab_source_accuracy"],
    }).reset_index()
    # label-free quantities
    zf = zipfile.ZipFile(validation_zip)
    val = pd.read_csv(io.BytesIO(zf.read("validation/validation_merged_zeroshot.csv"))).rename(
        columns={"source_city": "source", "target_city": "target"})
    info = info.merge(val[["source", "target", "overlap"]], on=["source", "target"])
    # the runs file lacks the pool's Structural share; take it from the score cache
    info = info.merge(pool_structural(scores_zip), on=["source", "target"])
    return final, info


def pool_structural(scores_zip):
    """Structural Ignorance share of each target's training pool (label-free), from the score cache."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune_experiments"))
    import finetune_prospective as fp
    from sampling_sensitivity import BASE, PairContext, cutoffs, load_source
    from run_sampling_experiment import _SRC_TRAIN

    class A:
        pass
    A.scores_zip = scores_zip
    rows = []
    for s in CITIES:
        pairs, locs, forests, ref = load_source(A, s)
        for t in [c for c in CITIES if c not in _SRC_TRAIN[s]]:
            ctx = PairContext(pairs, locs, t, None)
            i, p, _ = cutoffs(BASE, ref, forests)
            _, tiers = ctx.tiers(i, p)
            rows.append({"source": s, "target": t,
                         "pool_structural": float((tiers == fp.TIER_STRUCTURAL).mean())})
    return pd.DataFrame(rows)


def choose(info, final, n):
    """Chosen source per rule, target and seed."""
    out = []
    best_by_t = final[n].groupby(level="target").idxmax()
    for (t, seed), g in info.groupby(["target", "random_seed"]):
        g = g.set_index("source")
        zs = final.xs(t, level="target")[0]
        picks = {
            "seed_accuracy": g["seed_acc"].idxmax(),
            "seed_pd": g["seed_pd"].idxmin(),
            "seed500_accuracy": g["seed500_acc"].idxmax(),
            "fitted": g["fitted"].idxmax(),
            "coverage": g["overlap"].idxmax(),
            "low_structural": g["pool_structural"].idxmin(),
            "zs_spearman": zs.idxmax(),
            "best": best_by_t[t][0],
        }
        for rule, src in picks.items():
            out.append({"target": t, "random_seed": seed, "sample_size": n, "rule": rule, "source": src})
    return pd.DataFrame(out)


def fit_loto(info, final, n):
    """Leave-one-target-out linear prediction of final Spearman at n."""
    d = info.merge(final[n].rename("y").reset_index(), on=["source", "target"])
    pred = pd.Series(index=d.index, dtype=float)
    X = lambda df: np.column_stack([np.ones(len(df)), df["seed_acc"], df["pool_structural"]])
    for t in d["target"].unique():
        tr, te = d["target"] != t, d["target"] == t
        beta, *_ = np.linalg.lstsq(X(d[tr]), d.loc[tr, "y"], rcond=None)
        pred[te] = X(d[te]) @ beta
    return pred.to_numpy()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    final, info = load_singles(args.runs_zip, args.validation_zip, args.scores_zip)

    choices, regret_rows = [], []
    for n in SIZES:
        info["fitted"] = fit_loto(info, final, n)
        ch = choose(info, final, n)
        val = final[n]
        ch["value"] = [val[(s, t)] for s, t in zip(ch["source"], ch["target"])]
        best = val.groupby(level="target").max()
        ch["regret"] = ch["target"].map(best) - ch["value"]
        ranks = val.groupby(level="target").rank(ascending=False)
        ch["rank"] = [ranks[(s, t)] for s, t in zip(ch["source"], ch["target"])]
        choices.append(ch)
        # random choice: expected regret over the target's candidates
        rnd = (best - val.groupby(level="target").mean()).rename("regret").reset_index()
        for rule, g in ch.groupby("rule"):
            regret_rows.append({"sample_size": n, "rule": rule, "mean_regret": g["regret"].mean(),
                                "median_regret": g["regret"].median(),
                                "p_best": (g["rank"] == 1).mean(), "p_top2": (g["rank"] <= 2).mean(),
                                "n_distinct_choices": g.groupby("target")["source"].nunique().mean()})
        regret_rows.append({"sample_size": n, "rule": "random", "mean_regret": rnd["regret"].mean(),
                            "median_regret": rnd["regret"].median(), "p_best": 1 / 6, "p_top2": 2 / 6,
                            "n_distinct_choices": np.nan})
    choices = pd.concat(choices, ignore_index=True)
    regret = pd.DataFrame(regret_rows)
    regret["rule"] = pd.Categorical(regret["rule"], RULES)
    regret = regret.sort_values(["sample_size", "rule"])

    # how well does seed accuracy rank sources within a target?
    rk = []
    for n in SIZES:
        for (t, seed), g in info.groupby(["target", "random_seed"]):
            y = [final[n][(s, t)] for s in g["source"]]
            rk.append({"sample_size": n, "rho_seed_acc": spearmanr(g["seed_acc"], y).correlation,
                       "rho_seed500_acc": spearmanr(g["seed500_acc"], y).correlation,
                       "rho_overlap": spearmanr(g["overlap"], y).correlation})
    rk = pd.DataFrame(rk).groupby("sample_size").mean()

    # decision 2: chosen single city vs aggregates, within target
    zf = zipfile.ZipFile(args.study_zip)
    st = pd.concat([pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith(".csv")])
    st = st[st["target"].isin(CITIES)]
    st["type"] = st["source"].map(source_type)
    agg = st[st["type"] != "single"]
    zs = agg[agg["strategy"] == "zero_shot"].groupby(["source", "target", "type"])["spearman"].first().rename(0)
    ft = agg[agg["strategy"] != "zero_shot"].groupby(["source", "target", "type", "sample_size"])["spearman"].mean().unstack()
    agg_v = pd.concat([zs, ft], axis=1).groupby(["target", "type"]).mean()
    vs = []
    for n in SIZES:
        sel = choices[(choices["rule"] == "seed_accuracy") & (choices["sample_size"] == n)].groupby("target")["value"].mean()
        avg_single = final[n].groupby(level="target").mean()
        best_single = final[n].groupby(level="target").max()
        for typ in ["tier", "continental", "aggregate"]:
            a = (agg_v[n].groupby(level="target").mean() if typ == "aggregate"
                 else agg_v[n].xs(typ, level="type"))
            for name, s in [("selected_single", sel), ("average_single", avg_single), ("best_single", best_single)]:
                d = (s - a).dropna()
                vs.append({"sample_size": n, "single": name, "vs": typ, "mean_diff": d.mean(),
                           "targets_single_better": f"{int((d > 0).sum())}/{len(d)}",
                           "p_wilcoxon_exact": wilcoxon(d, alternative="two-sided", method="exact").pvalue})
    vs = pd.DataFrame(vs)

    # decision 2, fair version: the seed rule chooses among aggregates too
    fair = None
    if args.seed_preds and os.path.exists(args.seed_preds):
        acc = seed_accuracy_table(args.seed_preds)
        agg_src = pd.concat([zs, ft], axis=1).reset_index()
        fair = fair_comparison(acc, final, agg_src)
        fair.to_csv(os.path.join(args.out_dir, "source_selection_fair.csv"), index=False)

    regret.to_csv(os.path.join(args.out_dir, "source_selection_regret.csv"), index=False)
    choices.to_csv(os.path.join(args.out_dir, "source_selection_choices.csv"), index=False)
    vs.to_csv(os.path.join(args.out_dir, "source_selection_vs_aggregates.csv"), index=False)
    pd.set_option("display.width", 250)
    print("Regret (best source's final Spearman - chosen source's), mean over 7 targets x 5 seeds:")
    print(regret.pivot(index="rule", columns="sample_size", values="mean_regret").round(3).to_string())
    print("\nShare of choices that are the best source:")
    print(regret.pivot(index="rule", columns="sample_size", values="p_best").round(2).to_string())
    print("\nWithin-target Spearman between a seed quantity and final Spearman (mean over targets x seeds):")
    print(rk.round(2).to_string())
    print("\nSingle city vs aggregates, within target (positive = single better):")
    print(vs.round(3).to_string(index=False))
    if fair is not None:
        print("\nSeed-selected single city vs seed-selected aggregate, within target:")
        print(fair.round(3).to_string(index=False))


def seed_accuracy_table(path):
    """Accuracy of every source on every seed batch, from dump_seed_predictions.py."""
    d = pd.read_csv(path)
    d["correct"] = d["y"] == (d["src_prob"] > 0.5).astype(int)
    d = d.assign(batch=d["in_batches"].str.split(";")).explode("batch")
    d[["seed_n", "random_seed"]] = d["batch"].str.split(":", expand=True).astype(int)
    return d.groupby(["source", "target", "seed_n", "random_seed"])["correct"].mean().rename("acc").reset_index()


def fair_comparison(acc, final, agg_src):
    """Per target and seed, the seed rule picks one single city and one aggregate;
    their yardstick values (seed-mean final Spearman) are compared within target.
    Also: the rule applied to all candidates together."""
    acc["type"] = acc["source"].map(source_type)
    agg_val = agg_src.set_index(["source", "target"])
    rows = []
    for seed_n in sorted(acc["seed_n"].unique()):
        for n in [m for m in SIZES if m == 0 or m >= seed_n]:
            single_val, agg_val_n = final[n], agg_val[n]
            picks = []
            for (t, seed), g in acc[acc["seed_n"] == seed_n].groupby(["target", "random_seed"]):
                g = g.set_index("source")
                s_pick = g[g["type"] == "single"]["acc"].idxmax()
                a_pick = g[g["type"] != "single"]["acc"].idxmax()
                all_pick = g["acc"].idxmax()
                v = lambda src: single_val[(src, t)] if source_type(src) == "single" else agg_val_n[(src, t)]
                best_all = max(single_val.xs(t, level="target").max(), agg_val_n.xs(t, level="target").max())
                picks.append({"target": t, "single": v(s_pick), "aggregate": v(a_pick),
                              "all_is_aggregate": source_type(all_pick) != "single",
                              "regret_all": best_all - v(all_pick)})
            p = pd.DataFrame(picks).groupby("target").mean()
            d = p["single"] - p["aggregate"]
            rows.append({"seed_n": seed_n, "sample_size": n, "mean_diff": d.mean(),
                         "targets_single_better": f"{int((d > 0).sum())}/{len(d)}",
                         "p_wilcoxon_exact": wilcoxon(d, alternative="two-sided", method="exact").pvalue,
                         "share_all_rule_picks_aggregate": p["all_is_aggregate"].mean(),
                         "regret_all_candidates": p["regret_all"].mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-zip", default="../../results/sampling_v3.zip")
    ap.add_argument("--study-zip", default="../../results/random__layerwise_lr1e-04.zip")
    ap.add_argument("--validation-zip", default="../../results/tier1/validation.zip")
    ap.add_argument("--scores-zip", default="../../results/iso_scores.zip")
    ap.add_argument("--seed-preds", default="../../results/seed_predictions.csv.gz",
                    help="output of dump_seed_predictions.py; skipped if absent")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    main(ap.parse_args())
