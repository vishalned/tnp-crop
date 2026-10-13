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

cd /home/vnedungadi/tnp-crop
MODE=structured  # structured | random
TABLE=data/processed/wofost_maize_daily.parquet
OUT=logs/nanotabicl/continue_maize_$MODE

uv run python -m nanotabicl.train --init official --data crop --crop-episodes $MODE --table $TABLE \
    --steps 5000 --lr 3e-5 --micro-batch-size 2 --checkpoint-activations --out-dir $OUT

# test years, best checkpoint (lowest val RMSE)
uv run python -m nanotabicl.evaluate --checkpoint $OUT/best.pt --table $TABLE --years 2019 2020

echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
