#!/bin/bash
# Small test of the sampled-worlds pipeline (not the full generation): datasets A, A_w, B, C, D on a few
# cached locations, processed tables, the CY-Bench distribution gap per dataset, the checks, and the
# Morris sensitivity analysis. No Earth Engine downloads (cached locations only).
#   sbatch scripts/slurm/run_worlds_test.sh
# Settings (env): LOCS (location indices), WORLDS (per location), OUT.

#SBATCH --job-name=worlds-test
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/worlds-test%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/worlds-test%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
echo "Job started at:  $(date)"
cd /home/vnedungadi/tnp-crop
LOCS=${LOCS:-"67 172 262 343 649 1291 1455"}  # 2 BE, 2 DK, 3 FR (spread over latitude)
WORLDS=${WORLDS:-3}
OUT=${OUT:-/projects/prjs1788/tnp-crop/wofost_worlds_test}
LOCATIONS=data/raw/locations/locations_maize.csv
CYBENCH=/projects/prjs1788/CropBench/data/cybench/cybench-data
NPROC=${SLURM_CPUS_PER_TASK:-8}

for ds in A A_w B C D; do
    echo "===== dataset $ds ($(date +%H:%M:%S))"
    uv run python -m src.data_pipeline.wofost.generate_worlds_dataset --config configs/wofost_worlds/dataset_$ds.yaml \
        --location-indices $LOCS --worlds-per-location $WORLDS --workers $NPROC -o $OUT/$ds
    uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $OUT/$ds/dataset_manifest.csv \
        --locations-csv $LOCATIONS --wofost-dir $OUT/$ds -o $OUT/$ds/processed
done

echo "===== Morris sensitivity ($(date +%H:%M:%S))"
uv run python -m src.data_pipeline.wofost.sensitivity_morris --config configs/wofost_worlds/dataset_C.yaml \
    --out $OUT/sensitivity --n-locations 5 --years 2008 2012 2018 --trajectories 10 --workers $NPROC

for ds in A A_w B C D; do
    echo "===== distribution gap $ds ($(date +%H:%M:%S))"
    uv run python -m nanotabicl.distribution_gap --table $OUT/$ds/processed/wofost_maize_daily.parquet \
        --cybench-data $CYBENCH --countries BE FR DK --out $OUT/$ds/distribution_gap
done

echo "===== checks ($(date +%H:%M:%S))"
uv run python -m src.data_checks.worlds_checks --root $OUT --datasets A A_w B C D \
    --reference-manifest /projects/prjs1788/tnp-crop/wofost_calendar/dataset_manifest.csv \
    --locations-csv $LOCATIONS --out $OUT/checks
echo "Job finished at:  $(date)"
