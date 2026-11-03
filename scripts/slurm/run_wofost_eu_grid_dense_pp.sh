#!/bin/bash
# Densified EU grid simulated as potential production (Wofost81_PP = fully irrigated: no water limitation, no soil input):
# one chunk of the locations CSV per array task, same seasons as the rainfed OM-fix run (same seed, so the same sowing dates).
# Submitted by submit_eu_grid_dense_pp.sh. Like the OM-fix run it never writes season files to the project space:
#
#  * the per-season CSVs (~86,000 per chunk, ~14 GB) and the soil YAMLs are written to the node-local disk ($TMPDIR);
#  * the chunk's table is processed there and written to $RAW/part_k/processed/ (parquet), with the manifest;
#  * the season CSVs and the soil files are packed into $RAW/part_k/maize.tar.zst and soil.tar.zst, the
#    count in the archive is checked against the folder, and the local folder is removed.
# Project space per chunk: 4 entries + processed/ (parquet + columns json) = a handful of files, ~6-7 GB.
#
# Nothing is downloaded: weather and elevation come from the caches ($TNP_NO_GEE_WEATHER=1: a cache miss is a failed
# season, not a download); PP needs no soil. The processed table has no ssm / rsm (no water balance): they are taken
# from the rainfed run when the irrigation-map table is built (analysis/irrigation_map/build_irrigation_table.py).
# Settings as the rainfed run: 2005-2020, 3 sowing jitters (+-10 days), seed 42.

#SBATCH --job-name=wofost-pp
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-pp_%A_%a.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/wofost-pp_%A_%a.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
echo "Job started at:  $(date)"
start_ts=$(date +%s)

cd /home/vnedungadi/tnp-crop
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_dense700_s42.csv}
RAW=${RAW:-/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_pp}
FIRST0=${FIRST0:-0}
CHUNK=${CHUNK:-1800}
WORKERS=${WORKERS:-16}
k=$SLURM_ARRAY_TASK_ID
FIRST=$((FIRST0 + k * CHUNK))
LAST=$((FIRST + CHUNK))
PART=$RAW/part_$k
LOCAL=${TMPDIR:-/scratch-local/$USER}/pp_part_$k
echo "part $k: rows [$FIRST, $LAST) of $LOCATIONS -> $PART (season files in $LOCAL)"
mkdir -p "$LOCAL"
df -h "$LOCAL" | tail -1
free_gb=$(df -P --block-size=1G "$LOCAL" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 60 ]; then echo "only ${free_gb} GB free on the local disk, need ~60"; exit 1; fi
rm -rf "$PART" "$LOCAL"/*
mkdir -p "$PART"

export TNP_NO_GEE_WEATHER=1

uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv $LOCATIONS \
    --first-location $FIRST --max-locations $LAST --workers $WORKERS --seed 42 --model-name Wofost81_PP -o $LOCAL

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

# pack the local season files (project space: one archive each), check the counts, free the local disk
for sub in maize; do
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
