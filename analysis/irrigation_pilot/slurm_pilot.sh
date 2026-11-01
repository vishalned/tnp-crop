#!/bin/bash
# Irrigation pilot on a CPU node (no GPU, no training):  sbatch --export=ALL,N=75,OUT=<csv> analysis/irrigation_pilot/slurm_pilot.sh
#SBATCH --job-name=irr-pilot
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/irr-pilot_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/irr-pilot_%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/vnedungadi/tnp-crop
uv run python -m analysis.irrigation_pilot.run_pilot --n-per-country ${N:-75} --workers 16 \
    --out ${OUT:-/projects/prjs1788/tnp-crop/irrigation_pilot/pilot_seasons.csv}
