#!/bin/bash
# EU grid, densified (densify_grid_locations.py: ~700 usable points per country): simulate and process the
# locations added after the existing 2,000, one chunk per array task. Submitted by submit_eu_grid_dense.sh.
#
# Task k: rows [FIRST0 + k*CHUNK, FIRST0 + (k+1)*CHUNK) of the locations CSV -> $RAW/part_k/ (manifest and
# season files) and $RAW/part_k/processed/ (that chunk's table). Weather comes from the cache where the
# ERA5-Land pixel is already downloaded (only the elevation is fetched), soil per point. A task starts from
# an empty part_k (a resubmitted task never mixes runs; downloads resume from the caches). After processing,
# the season files are packed into part_k/maize.tar.zst (pack_season_files.sh): the project allows ~1.1 M
# files, and the processed table + manifest are all training needs.
# Settings as the EU grid run: 2005-2020, 3 sowing jitters, seed 42 (offsets are drawn for every row, so
# the first rows keep the EU grid run's episodes).

#SBATCH --job-name=wofost-dense
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-dense_%A_%a.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-dense_%A_%a.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_dense700_s42.csv}
RAW=${RAW:-/projects/prjs1788/tnp-crop/wofost_eu_grid_dense}
FIRST0=${FIRST0:-2000}
CHUNK=${CHUNK:-1750}
WORKERS=${WORKERS:-8}
k=$SLURM_ARRAY_TASK_ID
FIRST=$((FIRST0 + k * CHUNK))
LAST=$((FIRST + CHUNK))
PART=$RAW/part_$k
echo "part $k: rows [$FIRST, $LAST) of $LOCATIONS -> $PART"
rm -rf "$PART"

uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv $LOCATIONS \
    --first-location $FIRST --max-locations $LAST --workers $WORKERS --seed 42 -o $PART

uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $PART/dataset_manifest.csv \
    --locations-csv $LOCATIONS --wofost-dir $PART -o $PART/processed

uv run python -c "
import pandas as pd
m = pd.read_csv('$PART/dataset_manifest.csv')
print(m['status'].value_counts().to_string())
ok = m.groupby('location_index')['status'].apply(lambda s: (s == 'success').all())
print(f'locations: {len(ok)}, all seasons ok: {ok.sum()}')
print(m.loc[m['status'] != 'success', 'error'].astype(str).str.slice(0, 80).value_counts().head(5).to_string())
"

bash scripts/slurm/pack_season_files.sh $PART/maize

echo "Job finished at:  $(date)"
elapsed=$(( $(date +%s) - start_ts ))
echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
