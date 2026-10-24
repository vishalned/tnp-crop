#!/bin/bash
# Checks of the EU grid run against the old FR/BE/DK run (scaling_main):
#   1. season sanity per country (yields, near-zero yields, maturity, dates vs the crop calendar);
#   2. feature-yield agreement with CY-Bench per country (nanotabicl.distribution_gap), old and new table.
#   sbatch scripts/slurm/run_scaling_checks.sh

#SBATCH --job-name=scaling-checks
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/scaling-checks%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/scaling-checks%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
echo "Job started at:  $(date)"
cd /home/vnedungadi/tnp-crop
P=/projects/prjs1788/tnp-crop
CYBENCH=/projects/prjs1788/CropBench/data/cybench/cybench-data
COUNTRIES="NL FR DE PL ES IT AT BE BG CZ DK EL HR HU LT RO SE SK"   # the 18 countries of exp_1

[ -f analysis/scaling/sanity/sanity_per_country.csv ] || uv run python -m src.data_checks.season_sanity \
    --run old $P/wofost_calendar/dataset_manifest.csv data/raw/locations/locations_maize.csv \
    --run new $P/wofost_eu_grid/dataset_manifest.csv data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv \
    --out analysis/scaling/sanity --workers $SLURM_CPUS_PER_TASK

for run in old new; do
    table=$([ $run = old ] && echo $P/processed/wofost_maize_daily.parquet || echo $P/wofost_eu_grid/processed/wofost_maize_daily.parquet)
    uv run python -m nanotabicl.distribution_gap --table $table --cybench-data $CYBENCH --countries $COUNTRIES \
        --out $P/wofost_eu_grid/checks/distribution_gap_$run
done
echo "Job finished at:  $(date)"
