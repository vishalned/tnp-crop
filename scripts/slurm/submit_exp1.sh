#!/bin/bash
# exp_1: WOFOST vs SCM-prior control on all European maize countries of CY-Bench.
#
#   bash scripts/slurm/submit_exp1.sh          # run on a login node: it only calls sbatch
#   EXP=exp_1_eu_grid TABLE=/projects/prjs1788/tnp-crop/wofost_eu_grid/processed/wofost_maize_daily.parquet \
#       bash scripts/slurm/submit_exp1.sh      # same experiment on another WOFOST table (scaling: EU grid)
#   DATAS=crop       only the WOFOST runs (no SCM control);  AFTER=<jobid>  training waits for that job
#   TRAIN_TIME / TRAIN_MEM  resources of the WOFOST training jobs (default 03:00:00 / 32G)
#
# Training: continued pretraining of the official TabICLv2 weights, 3 seeds each for WOFOST tables
#   and the SCM control (prior tables with the WOFOST batches' shapes; seed N of both draws the same
#   table shapes). Recipe from the LR test (lr_test): AdamW, LR 1e-5, 2000 steps, warmup 10%, cosine
#   to 0, weight decay 0.01, clip 1; evaluated checkpoint = last.pt (annealed).
# Benchmark: every checkpoint on the 18 European maize countries (run_europe.sh: screening +
#   walk-forward, REPS=1), single pass (n_estimators=null); reference = default TabICL single pass
#   (MODEL=tabicl, n_estimators=1). A CPU job then writes summary.csv / per_run.csv.
#
# Outputs:  training   /projects/prjs1788/tnp-crop/nanotabicl/exp_1/<data>_seed<N>/
#           benchmark  /projects/prjs1788/tnp-crop/cybench_output/exp_1/  (+ summary.txt, summary.csv)
#           SLURM logs slurm_logs/exp_1/

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
BENCH=/gpfs/home2/vnedungadi/AgML-CY-Bench-Modeling
EXP=${EXP:-exp_1}
TABLE=${TABLE:-/projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet}  # WOFOST table (also sets the SCM control's shapes)
TRAIN_ROOT=/projects/prjs1788/tnp-crop/nanotabicl/$EXP
BENCH_ROOT=/projects/prjs1788/tnp-crop/cybench_output/$EXP
LOGS=$REPO/slurm_logs/$EXP
SEEDS=${SEEDS:-"0 1 2"}
DATAS=${DATAS:-"crop prior-matched"}
AFTER=${AFTER:-}
TRAIN_TIME=${TRAIN_TIME:-03:00:00}
TRAIN_MEM=${TRAIN_MEM:-32G}
mkdir -p "$LOGS" "$TRAIN_ROOT"

europe() {  # europe <dependency or ""> <tag> <extra env> <hydra overrides...>
    local dep=$1 tag=$2 env=$3; shift 3
    (cd "$BENCH" && sbatch --parsable ${dep:+--dependency=afterok:$dep} --job-name="$EXP-cyb-$tag" \
        --output="$LOGS/cybench_${tag}_%j.out" --error="$LOGS/cybench_${tag}_%j.err" \
        --export="ALL,NAME=$EXP,TAG=$tag$env" cybench/runs/tabicl_wofost/run_europe.sh "$@")
}

eval_jobs=()
# reference: default TabICL (official tabicl package), single pass
job=$(europe "" tabicl_default_single ",MODEL=tabicl" model.n_estimators=1)
echo "benchmark tabicl default single: $job"
eval_jobs+=("$job")

COMMON="STEPS=2000,STOP_AT=2000,LR=1e-5,WD=0.01,WARMUP_FRAC=0.1,EVAL=last.pt,TABLE=$TABLE"
for data in $DATAS; do
    name=$([ "$data" = crop ] && echo wofost || echo scm)
    for seed in $SEEDS; do
        run=${name}_seed$seed
        if [ "$data" = crop ]; then
            resources=(--time=$TRAIN_TIME --mem=$TRAIN_MEM)
            settings="ALL,$COMMON,DATA=crop,SEED=$seed,RUN=$EXP/$run"
        else  # prior generation is CPU-bound
            resources=(--time=04:00:00 --cpus-per-task=16)
            settings="ALL,$COMMON,DATA=prior-matched,WORKERS=14,SEED=$seed,RUN=$EXP/$run"
        fi
        train=$(sbatch --parsable "${resources[@]}" ${AFTER:+--dependency=afterok:$AFTER} --job-name="$EXP-$run" \
            --output="$LOGS/train_${run}_%j.out" --error="$LOGS/train_${run}_%j.err" \
            --export="$settings" "$REPO/scripts/slurm/run_gpu.sh")
        job=$(europe "$train" "$run" "" "model.checkpoint=$TRAIN_ROOT/$run/last.pt")
        echo "train $run: $train -> benchmark $job"
        eval_jobs+=("$job")
    done
done

deps=$(IFS=:; echo "${eval_jobs[*]}")
summary=$(sbatch --parsable --dependency=afterany:$deps --job-name="$EXP-summary" --partition=rome \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output="$LOGS/summary_%j.out" --error="$LOGS/summary_%j.err" --wrap="
for log in $TRAIN_ROOT/*/log.csv; do echo \"== \$log\"; awk -F, 'NR==1 || \$5 != \"\"' \"\$log\"; done > $TRAIN_ROOT/summary_training.txt
source $BENCH/cybench/runs/tabicl_wofost/env.sh && cd $BENCH && \
poetry run python cybench/runs/tabicl_wofost/summarize_exp.py $BENCH_ROOT > $BENCH_ROOT/summary.txt")
echo "summary job $summary -> $BENCH_ROOT/summary.txt"
