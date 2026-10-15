#!/bin/bash
# WOFOST vs CY-Bench distribution plots (all features + yield) for DE, FR, BE:
#   sbatch scripts/slurm/run_distribution_gap.sh
# Needs `uv pip install -e ../AgML-CY-Bench-Modeling --no-deps` (once). Output: docs/figures/distribution_gap/

#SBATCH --job-name=distribution-gap
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/distribution-gap%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/distribution-gap%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
echo "Job started at:  $(date)"
cd /home/vnedungadi/tnp-crop
uv run python -m nanotabicl.distribution_gap --table /projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet \
    --cybench-data /projects/prjs1788/CropBench/data/cybench/cybench-data --countries DE FR BE \
    --out docs/figures/distribution_gap
echo "Job finished at:  $(date)"
