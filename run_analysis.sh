#!/usr/bin/env bash
# Reproduce every statistic and figure from the files in results/.
# Statistics go to results/review_stats/, figures to figures/. Takes about 15 minutes.
#   bash run_analysis.sh
set -e
cd "$(dirname "$0")/scripts/review_stats"
export MPLBACKEND=Agg
mkdir -p ../../figures ../../results/review_stats

for s in alr_inference iso_sensitivity sat_error_rates coverage_inference coverage_thresholds \
         aggregation_inference source_selection sampling_inference; do
  echo "== $s"; python $s.py
done
echo "== scratch_vs_transfer"
python scratch_vs_transfer.py --long-csv ../../results/transfer_long_layerwise.csv ../../results/transfer_long_match.csv

for f in fig2_coverage_vs_disagreement fig3_source_types fig4_source_selection fig5_sampling \
         fig6_guide figS_sensitivity; do
  echo "== $f"; python $f.py
done
echo "Done: statistics in results/review_stats/, figures in figures/."
