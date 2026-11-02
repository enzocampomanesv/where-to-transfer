#!/usr/bin/env bash
# R3-4: transfer at n = 5,000 with the from-scratch training budget.
# Layerwise first (the paper's method), then match. Extra arguments are passed to
# both runs, e.g. a smoke test:
#   bash run_transfer_long.sh --targets TEM --sources ACC --seeds 42 --max-epochs 3 --out-dir /tmp/tl_test
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1

python transfer_long_budget.py --recipe layerwise "$@"
python transfer_long_budget.py --recipe match "$@"
