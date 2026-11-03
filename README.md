# where-to-transfer

Code and data for the article

> **When places look alike but communities disagree: an empirical guide to transferring urban livability models across cities**

The study trains a pairwise livability model for each of seven cities (Accra, Tema, Nairobi, Kisumu, Beira, Chimoio and Mumbai), using residents' comparisons of satellite image patches. It then asks how well each model transfers to the other six cities, why transfer fails, and how a practitioner should choose a source city and annotate the target.

The repository has two parts:

- **Reproducing the statistics and figures** takes a laptop and the result files in [`results/`](results/), about 50 MB. No GPU is needed.
- **Rerunning the experiments** (model training, the SAT isolation forests, finetuning and the review-round reruns) needs a GPU and the raw data from Zenodo ([doi:10.5281/zenodo.23910257](https://doi.org/10.5281/zenodo.23910257); see [`data/README.md`](data/README.md)).

## Quick start: statistics and figures

```bash
git clone https://github.com/enzocampomanesv/where-to-transfer.git
cd where-to-transfer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
bash run_analysis.sh                                    # all statistics, then all figures
```

This takes about 15 minutes; the permutation and bootstrap tests take most of it. The figures it produces are already in [`figures/`](figures/). `dataset_summary.py` needs the vote databases (Zenodo), so its outputs (`dataset_summary.csv`, `finetune_locations.csv`) are included rather than regenerated.

Every script in `scripts/review_stats/` defaults to reading from `results/` and writes statistics to `results/review_stats/` and figures to `figures/`. Each can also be run on its own (`python alr_inference.py --help`).

## What reproduces what

| Paper content | Script (`scripts/review_stats/`) | Inputs (`results/`) |
|---|---|---|
| SAT decomposition of zero-shot performance: ALR regression with exact MRQAP, city bootstrap and leave-one-city-out | `alr_inference.py` | `random__layerwise_lr1e-04.zip` |
| Sensitivity of the SAT categories and the ALR to the isolation-forest settings | `iso_sensitivity.py` | `iso_scores.zip`, `random__layerwise_lr1e-04.zip` |
| Error rates in familiar-looking vs unfamiliar comparisons (concept vs covariate shift) | `sat_error_rates.py` | `review_stats/iso_sensitivity_shares.csv` |
| Coverage (MMD, overlap) vs SAT shares, zero-shot and finetuning; threshold sensitivity | `coverage_inference.py`, `coverage_thresholds.py` | `tier1/validation.zip`, `tier1/overlap_percentiles.csv`, `random__layerwise_lr1e-04.zip`, `sampling_v3.zip` |
| Fig. 3: error rates on unfamiliar vs covered comparisons; Fig. 4: coverage vs Structural Ignorance and Preference Disagreement | `fig3_sat_coverage.py` (uses `fig1f_error_rates.py`, `fig2_coverage_vs_disagreement.py`) | as above, `review_stats/iso_sensitivity_shares.csv` |
| Fig. 5: source selection from a seed sample, and its calibration | `source_selection.py`, `fig4_source_selection.py` | `sampling_v3.zip`, `iso_scores.zip`, `tier1/validation.zip`, `seed_predictions.csv.gz` |
| Fig. 6: single-city vs multi-city sources | `aggregation_inference.py`, `fig5_source_types.py` (uses `fig3_source_types.py`) | `random__layerwise_lr1e-04.zip`, `sampling_v3.zip`, `seed_predictions.csv.gz` |
| Fig. 7 and Supplementary Fig. S3: transfer vs training from scratch (layer-wise; full finetuning) | `scratch_vs_transfer.py`, `fig6_scratch_vs_transfer.py` | `from_scratch.csv`, `transfer_long_layerwise.csv`, `transfer_long_match.csv`, `sampling_v3.zip` |
| Supplementary Fig. S1: sampling strategies (random, SAT-informed, uncertainty, oracle) | `sampling_inference.py`, `figS1_sampling.py` | `sampling_v3.zip` |
| Supplementary Tables S4 and S5: participants and agreement per city | `participant_summary.py` | the raw workshop database, which holds personal data and is not released; its aggregate output `review_stats/participant_summary.csv` is included |
| Additional sensitivity figure (not in the paper) | `figS_sensitivity.py` | `review_stats/iso_sensitivity_*.csv`, `review_stats/sampling_sensitivity_summary.csv` |

Figs. 1, 2, and 8 (study overview, SAT classes, and workflow) are drawings and have no script.

`dyadic_inference.py` holds the inference for dependent source–target pairs (exact QAP / MRQAP over the 5,040 city relabellings, city bootstrap, crossed random effects); `fig_style.py` holds shared plotting settings.

`sampling_sensitivity.py` replays the SAT sampler under different settings. It needs `pool_features.zip` (171 MB, on Zenodo), so its outputs (`results/review_stats/sampling_sensitivity_*.csv`) are included and `figS_sensitivity.py` reads them directly.

## The full pipeline

| Stage | Folder | Main entry points |
|---|---|---|
| 1. City models: a Bradley–Terry pairwise model per city | [`scripts/city_models/`](scripts/city_models/) | `ACC_train_bradleyTerry_ensemble.py` (with `train_voter_600k.py`, `archs/` and `utils/`); `join_bt_gpkg.py` joins scores to the patch grid |
| 2. SAT: isolation forests on the source model's image and pair features; categorisation of target comparisons into Consensus, Preference Disagreement, Relational and Structural Ignorance | [`scripts/sat/`](scripts/sat/) | `train_isoforest.sh` → `where_transfer_trainIso_BT_v2.py`; the two notebooks show the original build and inference steps |
| 3. Coverage: MMD and k-NN overlap between source and target embeddings | [`scripts/coverage/`](scripts/coverage/) | `t1_metrics.py` → `t1_validation.py`, `t1_rankings.py`; `t1_overlap_percentiles.py` |
| 4. Transfer experiments: zero-shot, finetuning, sampling strategies, training from scratch, longer training budgets | [`scripts/finetune_experiments/`](scripts/finetune_experiments/) | see below |
| 5. Statistics and figures | [`scripts/review_stats/`](scripts/review_stats/) | see above |

Shared by several stages: [`scripts/BT_utils/`](scripts/BT_utils/) estimates Bradley–Terry scores from pairwise predictions and evaluates them (`evalBT.py`), decomposes them (`decompBT.py`) and adds score metrics to the SAT categories (`aoaBT.py`).

Stage 4 in more detail:

- `ACC_transferLearn_BT.py`: the original finetuning experiments (zero-shot and finetuning at n = 100–5,000 for every source–target pair). Its outputs are `random__layerwise_lr1e-04.zip` and the `informed*.zip` files.
- `run_sampling_experiment.py` (all pairs: `run_all_pairs.sh`): the leak-free sampling rerun, with four strategies and five seeds per pair. Output `sampling_v3.zip`. The sampling logic is in `finetune_prospective.py` and `finetune_informed.py`, the training in `finetune_core.py` and the data split in `finetune_data.py`.
- `train_from_scratch.py`: the same architecture trained from random weights on the same target comparisons. Output `from_scratch.csv`.
- `transfer_long_budget.py` (both recipes: `run_transfer_long.sh`): transfer with the from-scratch training budget, using layer-wise or full finetuning. Output `transfer_long_*.csv`.
- `dump_seed_predictions.py`, `dump_iso_scores.py` and `dump_pool_features.py`: export the intermediate files used by the analysis.
- `run_pipeline.py`, `registry.py`, `gate.py`, `pairgraph.py`, `calibrate_degree.py`, `citywide_score.py`, `phase3.py`, `bt_sparse.py`, `normalize.py` and `rasterize.py`: shared infrastructure. They resolve models and paths, check reproducibility against the original runs, and score city-wide comparison graphs.
- `test_*.py`: self-checks for the modules above. Most need the raw data.

### Paths and environment

The experiment scripts read two environment variables:

```bash
export WT_DATA_ROOT=/path/to/zenodo/data_root     # vote databases, image patches, trained city models
export WT_ROOT=/path/to/zenodo/wt_root            # isolation forests, embeddings and outputs
export PYTHONPATH="$PWD/scripts:$PWD/scripts/city_models:$PYTHONPATH"
pip install -r requirements-models.txt
```

The expected layout under each is described in [`data/README.md`](data/README.md). Command-line flags (`--root-path`, `--out-dir`, …) override them. The two notebooks in `scripts/sat/` set their paths in the first cell; edit them there.

## Results files

| File | Content |
|---|---|
| `random__layerwise_lr1e-04.zip` | Original experiments: zero-shot and finetuned Spearman, accuracy and SAT category shares for every source–target pair and budget (random sampling, layer-wise finetuning, 3 seeds) |
| `informed*.zip`, `random.zip` | Original informed-sampling runs. These used target labels to choose the comparisons; the leak-free rerun in `sampling_v3.zip` replaces them |
| `sampling_v3.zip` | Leak-free sampling rerun: 42 pairs × 4 strategies × 5 seeds × 4 budgets |
| `from_scratch.csv` | Training from scratch on the target city |
| `transfer_long_layerwise.csv`, `transfer_long_match.csv` | Transfer with the from-scratch training budget (layer-wise and full finetuning) |
| `seed_predictions.csv.gz` | Zero-shot predictions of every source model on each target's seed comparisons |
| `iso_scores.zip` | Isolation-forest anomaly scores for every target comparison and image (for re-thresholding) |
| `tier1/` | Coverage metrics, rankings and their validation table |
| `review_stats/` | Outputs of `scripts/review_stats/` |

## Citation

If you use this code or data, please cite the article (reference to be added on publication).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The data on Zenodo are released under CC BY 4.0.
