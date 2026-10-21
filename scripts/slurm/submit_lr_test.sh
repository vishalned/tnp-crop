#!/bin/bash
# Learning-rate sweep for continued pretraining of the official TabICLv2 weights, WOFOST tables vs
# the SCM-prior control (prior tables with the WOFOST batches' shapes), then CY-Bench for every run.
#
#   bash scripts/slurm/submit_lr_test.sh          # run on a login node: it only calls sbatch
#
# Recipe (official TabICL fine-tuning: AdamW, warmup 10%, cosine to 0, weight decay 0.01, clip 1):
#   LR in {3e-6, 1e-5, 3e-5} x data in {crop, prior-matched}, 3000 steps, the cosine runs to the end,
#   so last.pt (step 3000) is annealed; step1000.pt / step2000.pt kept too.
# Then per run (after it succeeds) the benchmark (screening + walk-forward) on last.pt for maize
# DE/FR/BE, 8 estimators and 1 pass (n_estimators=null). Pick the LR by the screening split (validation years), not the
# walk-forward test years. A final CPU job writes the summaries to $TRAIN_ROOT.
#
# Outputs:  training      /projects/prjs1788/tnp-crop/nanotabicl/lr_test/<run>/
#           benchmark     /projects/prjs1788/tnp-crop/cybench_output/lr_test/
#           SLURM logs    slurm_logs/lr_test/

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
BENCH=/gpfs/home2/vnedungadi/AgML-CY-Bench-Modeling
TRAIN_ROOT=/projects/prjs1788/tnp-crop/nanotabicl/lr_test
BENCH_ROOT=/projects/prjs1788/tnp-crop/cybench_output/lr_test
LOGS=$REPO/slurm_logs/lr_test
STEPS=${STEPS:-3000}
LRS=${LRS:-"3e-6 1e-5 3e-5"}
mkdir -p "$LOGS" "$TRAIN_ROOT"

COMMON="STEPS=$STEPS,STOP_AT=$STEPS,SAVE_STEPS=1000 2000,WD=0.01,WARMUP_FRAC=0.1,EVAL=last.pt"
eval_jobs=()
for data in crop prior-matched; do
    name=$([ "$data" = crop ] && echo wofost || echo scm)
    for lr in $LRS; do
        run=${name}_lr${lr}
        if [ "$data" = crop ]; then
            resources=(--time=04:00:00)
            settings="ALL,$COMMON,DATA=crop,LR=$lr,RUN=lr_test/$run"
        else  # prior generation is CPU-bound
            resources=(--time=06:00:00 --cpus-per-task=16)
            settings="ALL,$COMMON,DATA=prior-matched,WORKERS=14,LR=$lr,RUN=lr_test/$run"
        fi
        train=$(sbatch --parsable "${resources[@]}" --job-name="lr-$run" \
            --output="$LOGS/train_${run}_%j.out" --error="$LOGS/train_${run}_%j.err" \
            --export="$settings" "$REPO/scripts/slurm/run_gpu.sh")
        echo "train $run: $train"

        ckpt=$TRAIN_ROOT/$run/last.pt
        for country in DE FR BE; do
            for est in 8 null; do  # null = one pass on the unchanged table (as the earlier "1 pass" rows)
                tag=$run$([ "$est" = 8 ] && echo _ens8 || echo _single)
                job=$(cd "$BENCH" && sbatch --parsable --dependency=afterok:$train --job-name="lr-cyb-$country" \
                    --output="$LOGS/cybench_${country}_${tag}_%j.out" --error="$LOGS/cybench_${country}_${tag}_%j.err" \
                    --export="ALL,COUNTRY=$country,REPS=1,NAME=lr_test,TAG=$tag" \
                    cybench/runs/tabicl_wofost/run_tabicl_wofost.sh "model.checkpoint=$ckpt" "model.n_estimators=$est")
                eval_jobs+=("$job")
            done
        done
    done
done

# Summary once every benchmark job has ended (also if some failed): training logs + CY-Bench tables.
deps=$(IFS=:; echo "${eval_jobs[*]}")
summary=$(sbatch --parsable --dependency=afterany:$deps --job-name=lr-summary --partition=rome \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output="$LOGS/summary_%j.out" --error="$LOGS/summary_%j.err" --wrap="
for log in $TRAIN_ROOT/*/log.csv; do echo \"== \$log\"; awk -F, 'NR==1 || \$5 != \"\"' \"\$log\"; done > $TRAIN_ROOT/summary_training.txt
for m in $TRAIN_ROOT/*/metrics_*.json; do echo \"== \$m\"; cat \"\$m\"; done > $TRAIN_ROOT/summary_wofost_test.txt
source $BENCH/cybench/runs/tabicl_wofost/env.sh && cd $BENCH && \
poetry run python cybench/runs/tabicl_wofost/summarize.py $BENCH_ROOT > $TRAIN_ROOT/summary_cybench.txt")
echo "${#eval_jobs[@]} benchmark jobs queued; summary job $summary -> $TRAIN_ROOT/summary_*.txt"
