#!/bin/bash
# Compare the WOFOST maize feature table with CY-Bench maize DE, built by the
# benchmark's own code (names in order, per-feature stats, distribution plots
# to docs/figures/), and run the tests incl. the slow real-data one:
#   sbatch scripts/slurm/run_cybench_check.sh
# Needs `uv pip install -e ../AgML-CY-Bench-Modeling --no-deps` (once).

#SBATCH --job-name=cybench-check
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/cybench-check%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/cybench-check%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
echo "Job started at:  $(date)"
cd /home/vnedungadi/tnp-crop
TABLE=/projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet
CYBENCH_DATA=/projects/prjs1788/CropBench/data/cybench/cybench-data

uv run python -m nanotabicl.cybench_check --table $TABLE --cybench-data $CYBENCH_DATA --country DE --out docs/figures
uv run pytest tests/test_cybench_features.py tests/test_nanotabicl.py -p no:cacheprovider

echo "Job finished at:  $(date)"
