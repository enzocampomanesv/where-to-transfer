#!/usr/bin/env bash
# Run run_sampling_experiment.py for all 42 single-city source->target pairs.
# Usage: bash run_all_pairs.sh [parallel_jobs=2] [out_dir]
# Resumable: finished (sampling, seed) blocks are skipped on rerun.
set -u
export PYTHONNOUSERSITE=1          # ignore ~/.local (numpy/h5py clash)
JOBS=${1:-2}
OUT=${2:-${WT_ROOT:-/home/jovyan/private/where_transfer}/results/transferLearning/sampling_v3}
CITIES="ACC NBO BEI TEM KIS CHI MUM"
mkdir -p "$OUT/logs"
cd "$(dirname "$0")"

for s in $CITIES; do for t in $CITIES; do
  [ "$s" = "$t" ] && continue
  echo "$s $t"
done; done | xargs -P "$JOBS" -n 2 sh -c \
  'python run_sampling_experiment.py --source "$0" --target "$1" --out-dir "'"$OUT"'" \
     > "'"$OUT"'/logs/$0_$1.log" 2>&1 && echo "done $0->$1" || echo "FAILED $0->$1 (see log)"'
