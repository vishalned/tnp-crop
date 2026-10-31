#!/bin/bash

#SBATCH --job-name=nanotabicl
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/nanotabicl%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/nanotabicl%j.err
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

echo "Job started at:  $(date)"
start_ts=$(date +%s)

# Settings via environment variables, e.g. a smoke run:
#   sbatch --export=ALL,STEPS=50,RUN=smoke_maize_structured scripts/slurm/run_gpu.sh
# (on Snellius `STEPS=50 sbatch ...` does NOT reach the job; use --export)
# Control run (prior tables shaped like the crop batches; more CPUs for the prior's generation):
#   sbatch --cpus-per-task=16 --export=ALL,DATA=prior-matched,WORKERS=14,STOP_AT=1000,SAVE_STEPS="500 1000",EVAL=step500.pt,RUN=scm_matched_maize_structured scripts/slurm/run_gpu.sh
cd /home/vnedungadi/tnp-crop
MODE=${MODE:-structured}  # structured | random
DATA=${DATA:-crop}        # crop | prior-matched (control: SCM prior tables with the crop batches' shapes)
STEPS=${STEPS:-5000}      # length of the learning-rate schedule
STOP_AT=${STOP_AT:-$STEPS}  # stop early, keeping the schedule of STEPS
SAVE_STEPS=${SAVE_STEPS:-}  # e.g. "500 1000": also save step<N>.pt
LR=${LR:-3e-5}
WD=${WD:-0}               # AdamW weight decay (official fine-tuning: 0.01)
WARMUP_FRAC=${WARMUP_FRAC:-}  # warmup as a fraction of STEPS (official fine-tuning: 0.1); empty = 200 steps
SEED=${SEED:-0}           # training seed (which tables are drawn; SCM: also the prior)
WORKERS=${WORKERS:-4}     # DataLoader workers (prior generation)
TRAIN_COUNTRIES=${TRAIN_COUNTRIES:-}  # train (and test) on these countries only: names as in the table, ':'-separated
                                     # (not COUNTRIES: the benchmark's run_europe.sh uses that name)
MICRO_BATCH=${MICRO_BATCH:-2}  # 1 if tables reach 4096 rows: 2 x 4096 rows x 8 heads exceeds CUDA's 65535 grid limit in row attention
EVAL=${EVAL:-best.pt}     # checkpoint evaluated on the test years
TABLE=${TABLE:-/projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet}  # CY-Bench features (calendar run)
RUN=${RUN:-continue_maize_$MODE}
OUT=/projects/prjs1788/tnp-crop/nanotabicl/$RUN  # project space: home is nearly full

uv run python -u -m nanotabicl.train --init official --data $DATA --crop-episodes $MODE --table $TABLE \
    --steps $STEPS --stop-at $STOP_AT --save-steps $SAVE_STEPS --lr $LR --weight-decay $WD --seed $SEED \
    ${WARMUP_FRAC:+--warmup-fraction $WARMUP_FRAC} --num-workers $WORKERS \
    ${TRAIN_COUNTRIES:+--countries ${TRAIN_COUNTRIES//:/ }} --micro-batch-size $MICRO_BATCH --checkpoint-activations --out-dir $OUT

# test years (WOFOST tables)
uv run python -u -m nanotabicl.evaluate --checkpoint $OUT/$EVAL --table $TABLE --years 2019 2020

echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
