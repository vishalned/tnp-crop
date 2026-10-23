#!/bin/bash
# EU grid locations (grid_sample_locations.py: one point per 0.1 deg maize cell, 2000 cells): download soil
# and weather from Earth Engine, simulate WOFOST (same settings as wofost_calendar: 2005-2020, 3 sowing
# jitters, seed 42) and build the processed table, in one job:
#   MAX_LOCATIONS=20 RUN=wofost_eu_grid_test sbatch scripts/slurm/run_wofost_eu_grid.sh   # small test first
#   sbatch scripts/slurm/run_wofost_eu_grid.sh                                            # all locations
# WORKERS = parallel locations = parallel Earth Engine requests; keep it modest. Re-running resumes from
# the soil/weather caches (data/raw/soilgrids_gee, data/raw/weather), so a timed-out job can be resubmitted.
# Earth Engine project: $EE_PROJECT (see ensure_ee_initialized).

#SBATCH --job-name=wofost-eu-grid
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-eu-grid%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-eu-grid%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
CROP=maize
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv}
MAX_LOCATIONS=${MAX_LOCATIONS:-}
WORKERS=${WORKERS:-8}
OUT=/projects/prjs1788/tnp-crop
RAW=$OUT/${RUN:-wofost_eu_grid}
PROCESSED=$RAW/processed

uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv $LOCATIONS \
    ${MAX_LOCATIONS:+--max-locations $MAX_LOCATIONS} --workers $WORKERS --seed 42 -o $RAW

uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $RAW/dataset_manifest.csv \
    --locations-csv $LOCATIONS --wofost-dir $RAW -o $PROCESSED

uv run python -c "
import pandas as pd
m = pd.read_csv('$RAW/dataset_manifest.csv')
print(m['status'].value_counts().to_string())
print(m.groupby('location_index')['status'].apply(lambda s: (s == 'success').all()).value_counts().rename('all seasons ok').to_string())
from nanotabicl.crop_tables import CropTables
t = CropTables('$PROCESSED/wofost_${CROP}_daily.parquet')
print(f'{len(t.df)} rows, {len(t.feature_names)} features, regions', {c: len(p) for c, p in t.country_points.items()})
"

echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))
echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
