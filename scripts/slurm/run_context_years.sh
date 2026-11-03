#!/bin/bash
# exp_5: does the pretraining help when the context is short? One model, the CY-Bench countries in $COUNTRIES, the
# context limited to the last k years before each forecast year (all regions kept): benchmark env TNP_MAX_CONTEXT_YEARS=k
# (local branch `context-years` of the benchmark). Variables: NAME (output folder), TAG (run tag), MODEL, CKPT (checkpoint,
# for the NanoTabICL models), LEVELS (e.g. "2 3 5"), COUNTRIES.   One GPU job, runs every level one after the other.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -uo pipefail
BENCH_DIR=/home/vnedungadi/AgML-CY-Bench-Modeling
for k in $LEVELS; do
    echo "===== context years: $k ($(date +%H:%M:%S))"
    if [ "$k" = all ]; then unset TNP_MAX_CONTEXT_YEARS; else export TNP_MAX_CONTEXT_YEARS=$k; fi
    if [ "${MODEL:-tabicl_wofost}" = tabicl ]; then extra=(model.n_estimators=1); else extra=("model.checkpoint=$CKPT"); fi
    export TAG=${TAG_BASE}_ctx${k}${SEED_SUFFIX:-}
    bash "$BENCH_DIR/cybench/runs/tabicl_wofost/run_europe.sh" "${extra[@]}"
done
