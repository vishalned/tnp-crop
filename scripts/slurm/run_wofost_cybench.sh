#!/bin/bash
# Re-simulate WOFOST with the crop calendar (campaign from SOS - 60 days to at
# least EOS, sowing at the WorldCereal SOS +- jitter) and build the processed
# table over CY-Bench's feature window, in one job:
#   sbatch scripts/slurm/run_wofost_cybench.sh
# Only the first MAX_LOCATIONS locations are run: their soil and weather are
# cached already (the first batch run stopped there), so no Earth Engine
# calls are needed. Output goes to project space (home is nearly full).

#SBATCH --job-name=wofost-cybench
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-cybench%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-cybench%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
CROP=maize
LOCATIONS=data/raw/locations/locations_${CROP}.csv
MAX_LOCATIONS=1585
OUT=/projects/prjs1788/tnp-crop
RAW=$OUT/wofost_calendar
PROCESSED=$OUT/processed

uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv $LOCATIONS \
    --max-locations $MAX_LOCATIONS --workers $SLURM_CPUS_PER_TASK --seed 42 -o $RAW

uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $RAW/dataset_manifest.csv \
    --locations-csv $LOCATIONS --wofost-dir $RAW -o $PROCESSED

# the CY-Bench feature table NanoTabICL trains on: shape, names, NaN
uv run python -c "
from nanotabicl.crop_tables import CropTables
t = CropTables('$PROCESSED/wofost_${CROP}_daily.parquet')
print(f'{len(t.df)} rows, {len(t.feature_names)} features, regions', {c: len(p) for c, p in t.country_points.items()})
print(t.feature_names)
"

echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))
echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
