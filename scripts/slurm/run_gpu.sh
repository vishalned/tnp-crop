#!/bin/bash
# One training run (continued pretraining of the official TabICLv2 weights, nanotabicl.train) and its evaluation on the WOFOST test years, on one
# A100. The run is a Hydra experiment config (configs/experiment/); settings via --export (on Snellius `VAR=x sbatch ...` does NOT reach the job):
#
#   sbatch --export=ALL,EXPERIMENT=exp_2a_loguniform_points,SEED=0 scripts/slurm/run_gpu.sh
#   sbatch --export=ALL,EXPERIMENT=exp_1_eu_grid_scm,SEED=1 --cpus-per-task=16 scripts/slurm/run_gpu.sh    # SCM control: CPU-bound prior
#   sbatch --export=ALL,EXPERIMENT=exp_2a_loguniform_points,OVERRIDES="recipe.steps=50 recipe.eval_every=25" scripts/slurm/run_gpu.sh   # a smoke run
#
# OVERRIDES: more Hydra overrides (space separated); the output folder is paths.runs/<name> (project space: home is nearly full).
# Memory: the processed tables are ~5 GB of parquet, so the job needs ~120 GB (submit_experiment.sh asks for it).

#SBATCH --job-name=nanotabicl
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/nanotabicl%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/nanotabicl%j.err
#SBATCH --mem=120G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)
cd ${REPO:-/home/vnedungadi/tnp-crop}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT (a file of configs/experiment/)}
SEED=${SEED:-0}
OVERRIDES=${OVERRIDES:-}

ARGS="experiment=$EXPERIMENT seed=$SEED $OVERRIDES"
OUT=$(uv run python -m nanotabicl.train $ARGS --cfg job --resolve | awk '/^out_dir:/ {print $2}')
echo "run: $ARGS -> $OUT"
uv run python -u -m nanotabicl.train $ARGS

# test years (WOFOST table of the run); last.pt is the annealed checkpoint
uv run python -u -m nanotabicl.evaluate --checkpoint $OUT/last.pt --years 2019 2020

echo "Job finished at:  $(date)"
elapsed=$(( $(date +%s) - start_ts ))
echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
