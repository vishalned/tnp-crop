#!/bin/bash
# Simulate WOFOST for a locations CSV in chunks (one chunk of rows per array task) and process each chunk to a table. Submitted by
# submit_wofost_run.sh (variants: omfix, pp). Settings via --export:
#
#   LOCATIONS   locations CSV (default: the densified EU grid, 12,382 locations)
#   RAW         output root: part_<k>/{processed/, dataset_manifest.csv, maize.tar.zst[, soil.tar.zst]}
#   FIRST0, CHUNK   task k runs rows [FIRST0 + k*CHUNK, FIRST0 + (k+1)*CHUNK)   (default 0, 1800)
#   WORKERS     processes per task (default 16);  SEED  sowing-jitter seed (default 42)
#   SOIL_TABLE  raw SoilGrids parquet (src/data_pipeline/soil/download_raw_soilgrids.py): the soil is derived from it, nothing is downloaded.
#               Empty: soil is downloaded from Earth Engine per location (new regions).
#   OFFLINE_WEATHER=1   weather and elevation from the caches only (a cache miss is a failed season, not a download); empty: download.
#   GEN_ARGS    more arguments of generate_wofost_dataset (e.g. "--model-name Wofost81_PP" for fully irrigated; "--tsum-factor-range 0.8 1.2")
#   PACK        what to pack into archives (default "maize soil"; "maize" without soil files)
#
# File-count strategy (the project space allows ~1.1 M files): the per-season CSVs (~86,000 per chunk, ~14 GB) and the soil YAMLs are written to the
# node-local scratch ($TMPDIR); the chunk's table is processed there and written to $RAW/part_k/processed/ (parquet) with the manifest; the
# season and soil files are packed into $RAW/part_k/<name>.tar.zst, the count in the archive is checked against the folder, and the local folder
# is removed. Project space per chunk: a handful of files, ~6-7 GB.

#SBATCH --job-name=wofost-chunks
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-chunks_%A_%a.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-chunks_%A_%a.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_dense700_s42.csv}
RAW=${RAW:?set RAW (the output root)}
SOIL_TABLE=${SOIL_TABLE:-}
OFFLINE_WEATHER=${OFFLINE_WEATHER:-}
GEN_ARGS=${GEN_ARGS:-}
PACK=${PACK:-maize soil}
SEED=${SEED:-42}
FIRST0=${FIRST0:-0}
CHUNK=${CHUNK:-1800}
WORKERS=${WORKERS:-16}
k=$SLURM_ARRAY_TASK_ID
FIRST=$((FIRST0 + k * CHUNK))
LAST=$((FIRST + CHUNK))
PART=$RAW/part_$k
LOCAL=${TMPDIR:-/scratch-local/$USER}/chunk_$k
echo "part $k: rows [$FIRST, $LAST) of $LOCATIONS -> $PART (season files in $LOCAL)"
mkdir -p "$LOCAL"
df -h "$LOCAL" | tail -1
free_gb=$(df -P --block-size=1G "$LOCAL" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 60 ]; then echo "only ${free_gb} GB free on the local disk, need ~60"; exit 1; fi
rm -rf "$PART" "$LOCAL"/*
mkdir -p "$PART"

export TNP_SOIL_DIR=$LOCAL/soil   # soil files go to the local disk and are packed with the season files
if [ -n "$SOIL_TABLE" ]; then export TNP_RAW_SOIL=$SOIL_TABLE TNP_SOIL_NO_DOWNLOAD=1; fi
if [ -n "$OFFLINE_WEATHER" ]; then export TNP_NO_GEE_WEATHER=1; fi

uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv $LOCATIONS \
    --first-location $FIRST --max-locations $LAST --workers $WORKERS --seed $SEED $GEN_ARGS -o $LOCAL

uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $LOCAL/dataset_manifest.csv \
    --locations-csv $LOCATIONS --wofost-dir $LOCAL -o $PART/processed
cp $LOCAL/dataset_manifest.csv $PART/dataset_manifest.csv

uv run python -c "
import pandas as pd
m = pd.read_csv('$PART/dataset_manifest.csv')
print(m['status'].value_counts().to_string())
ok = m.groupby('location_index')['status'].apply(lambda s: (s == 'success').all())
print(f'locations: {len(ok)}, all seasons ok: {ok.sum()}')
print(m.loc[m['status'] != 'success', 'error'].astype(str).str.slice(0, 100).value_counts().head(8).to_string())
"

# pack the local season and soil files (project space: one archive each), check the counts, free the local disk
for sub in $PACK; do
    dir=$LOCAL/$sub
    n=$(find "$dir" -type f | wc -l)
    tar -C "$LOCAL" -I "zstd -T16 -3" -cf "$PART/$sub.tar.zst" "$sub"
    m=$(tar -I zstd -tf "$PART/$sub.tar.zst" | grep -cv '/$')
    echo "$sub: $n files -> $PART/$sub.tar.zst ($m listed, $(du -h "$PART/$sub.tar.zst" | cut -f1))"
    [ "$m" -eq "$n" ] || { echo "MISMATCH in $sub archive; keeping $LOCAL"; exit 1; }
done
rm -rf "$LOCAL"
echo "files in $PART: $(find "$PART" -type f | wc -l), size $(du -sh "$PART" | cut -f1)"

echo "Job finished at:  $(date)"
elapsed=$(( $(date +%s) - start_ts ))
echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"
