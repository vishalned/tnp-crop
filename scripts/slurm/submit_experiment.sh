#!/bin/bash
# An experiment end to end: training (Hydra experiment config) -> CY-Bench benchmark -> summary (run on a login node: it only calls sbatch).
#
#   bash scripts/slurm/submit_experiment.sh exp_2a_loguniform_points                 # seed 0
#   SEEDS="0 1 2" bash scripts/slurm/submit_experiment.sh exp_2a_loguniform_points   # 3 seeds
#   OVERRIDES="recipe.steps=500" bash scripts/slurm/submit_experiment.sh exp_4_obs_noise
#
# Per seed: run_gpu.sh trains the experiment (configs/experiment/<experiment>.yaml; output paths.runs/<name>) and evaluates the test years; then the
# benchmark runs the checkpoint (last.pt) on the 18 European maize countries (screening + walk-forward, a single pass, REPS=1) in the benchmark
# repo's run_europe.sh; a CPU job writes summary.txt / summary.csv / per_run.csv. The default TabICL reference (deterministic) is copied from
# $REF_FROM (default exp_1_eu_grid_dense_omfix) instead of being run again; REFERENCE=run runs it.
#
# Results: training /projects/prjs1788/tnp-crop/nanotabicl/<name>/, benchmark /projects/prjs1788/tnp-crop/cybench_output/<experiment folder>/. After it:
# `uv run python -m analysis.collect_results` and `uv run python -m analysis.make_plots`.
# Environment: SEEDS, OVERRIDES, AFTER=<jobid> (training waits for it), REF_FROM, REFERENCE, TRAIN_TIME (08:00:00), TRAIN_MEM (120G),
#   SBATCH="echo sbatch" prints the commands (dry run), REPO (this repo).

set -euo pipefail
EXPERIMENT=${1:?experiment name (a file of configs/experiment/)}
REPO=${REPO:-/gpfs/home2/vnedungadi/tnp-crop}
BENCH=/gpfs/home2/vnedungadi/AgML-CY-Bench-Modeling
ROOT=/projects/prjs1788/tnp-crop
SEEDS=${SEEDS:-0}
OVERRIDES=${OVERRIDES:-}
AFTER=${AFTER:-}
TRAIN_TIME=${TRAIN_TIME:-08:00:00}
TRAIN_MEM=${TRAIN_MEM:-120G}
REFERENCE=${REFERENCE:-copy}
REF_FROM=$ROOT/cybench_output/${REF_FROM:-exp_1_eu_grid_dense_omfix}
SBATCH=${SBATCH:-sbatch}

info() {  # info <seed> -> "<name> <prior_fraction> <workers>" of the composed config
    REPO=$REPO uv run --project $REPO python - "$EXPERIMENT" "$1" "$OVERRIDES" <<'PY'
import os, sys
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=os.environ["REPO"] + "/configs", version_base="1.3"):
    c = compose("train", overrides=[f"experiment={sys.argv[1]}", f"seed={sys.argv[2]}", *sys.argv[3].split()])
print(c.name, c.source.prior_fraction, c.source.num_workers)
PY
}

eval_jobs=()
for seed in $SEEDS; do
    read -r name fraction workers <<< "$(info $seed)"
    exp=${name%%/*}; tag=${name#*/}
    LOGS=$REPO/slurm_logs/$exp; BENCH_ROOT=$ROOT/cybench_output/$exp; mkdir -p "$LOGS" "$BENCH_ROOT"
    if [ "$(echo "$fraction > 0" | bc)" = 1 ]; then cpus=16; else cpus=8; fi   # prior tables are generated on the CPU
    train=$($SBATCH --parsable --time=$TRAIN_TIME --mem=$TRAIN_MEM --cpus-per-task=$cpus ${AFTER:+--dependency=afterok:$AFTER} \
        --job-name="$exp-$tag" --output="$LOGS/train_${tag}_%j.out" --error="$LOGS/train_${tag}_%j.err" \
        --export="ALL,REPO=$REPO,EXPERIMENT=$EXPERIMENT,SEED=$seed,OVERRIDES=$OVERRIDES" $REPO/scripts/slurm/run_gpu.sh)
    bench=$(cd "$BENCH" && $SBATCH --parsable --dependency=afterok:$train --job-name="$exp-cyb-$tag" \
        --output="$LOGS/cybench_${tag}_%j.out" --error="$LOGS/cybench_${tag}_%j.err" \
        --export="ALL,NAME=$exp,TAG=$tag" cybench/runs/tabicl_wofost/run_europe.sh "model.checkpoint=$ROOT/nanotabicl/$name/last.pt")
    echo "train $name: $train -> benchmark $bench"
    eval_jobs+=("$bench")
done

if [ "$REFERENCE" = run ]; then   # default TabICL, single pass
    LOGS=$REPO/slurm_logs/$EXPERIMENT
    ref=$(cd "$BENCH" && $SBATCH --parsable --job-name="$EXPERIMENT-cyb-default" --output="$LOGS/cybench_default_%j.out" \
        --error="$LOGS/cybench_default_%j.err" --export="ALL,NAME=$EXPERIMENT,TAG=tabicl_default_single,MODEL=tabicl" \
        cybench/runs/tabicl_wofost/run_europe.sh model.n_estimators=1)
    echo "benchmark default TabICL: $ref"; eval_jobs+=("$ref")
elif [ -n "${exp:-}" ]; then
    mkdir -p "$BENCH_ROOT"; cp -r "$REF_FROM"/*_tabicl_eos_tabicl_default_single "$BENCH_ROOT"/ 2>/dev/null || true
    echo "default TabICL reference: copied from $REF_FROM"
fi

deps=$(IFS=:; echo "${eval_jobs[*]}")
summary=$($SBATCH --parsable --dependency=afterany:$deps --job-name="$exp-summary" --partition=rome --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output="$LOGS/summary_%j.out" --error="$LOGS/summary_%j.err" --wrap="
source $BENCH/cybench/runs/tabicl_wofost/env.sh && cd $BENCH && \
poetry run python cybench/runs/tabicl_wofost/summarize_exp.py $BENCH_ROOT > $BENCH_ROOT/summary.txt")
echo "summary job $summary -> $BENCH_ROOT/summary.txt"
