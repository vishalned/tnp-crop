#!/bin/bash
# Download the raw SoilGrids values of a locations CSV from Earth Engine into ONE parquet table (the only download
# of the OM-fix rerun; weather and elevation come from the caches). See src/data_pipeline/soil/download_raw_soilgrids.py.
#   sbatch --export=ALL,LOCATIONS=...,OUT=...[,MAX=6] scripts/slurm/download_raw_soil.sh
#SBATCH --job-name=soil-raw
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/soil-raw_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/soil-raw_%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
echo "Job started at:  $(date)"
cd /home/vnedungadi/tnp-crop
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_dense700_s42.csv}
OUT=${OUT:-/projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet}
uv run python -m src.data_pipeline.soil.download_raw_soilgrids --locations-csv $LOCATIONS --out $OUT \
    ${MAX:+--max-locations $MAX} --workers 8
echo "Job finished at:  $(date)"
