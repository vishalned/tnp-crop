#!/bin/bash

#SBATCH --job-name=wofost-sample
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-sample%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-sample%j.err
#SBATCH --mem=16G
#SBATCH --partition=rome
#SBATCH --time=05:00:00

echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv data/raw/locations/locations_maize.csv

echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"