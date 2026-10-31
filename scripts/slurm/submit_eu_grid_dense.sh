#!/bin/bash
# Densified EU grid, end to end (run on a login node: it only calls sbatch):
#
#   bash scripts/slurm/submit_eu_grid_dense.sh
#   TEST=1 bash scripts/slurm/submit_eu_grid_dense.sh     # 6 locations in one part, to wofost_eu_grid_dense_test
#   CHUNKS=2-5 AFTER=<jobid> bash scripts/slurm/submit_eu_grid_dense.sh   # only these chunks, after that job
#       (the merge always takes every chunk 0..NCHUNK-1)
#
# 1. run_wofost_eu_grid_dense.sh: the new locations in chunks (one array task at a time, so Earth Engine
#    sees at most WORKERS requests at once); each chunk simulated and processed on its own.
# 2. merge: the existing EU grid table (the CSV's first 2,000 rows) + every chunk's table -> one table, and
#    the manifests -> one manifest; then the table shares (analysis/scaling/table_share.py).
# 3. exp_1 on the merged table, WOFOST only (DATAS=crop), as exp_1_eu_grid_dense (submit_exp1.sh).

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
LOCATIONS=data/raw/locations/locations_maize_eu27_dense700_s42.csv
OLD=/projects/prjs1788/tnp-crop/wofost_eu_grid
N=$(( $(wc -l < $REPO/$LOCATIONS) - 1 ))
if [ "${TEST:-0}" = 1 ]; then
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_test; CHUNK=6; NCHUNK=1; FIRST0=${FIRST0:-2000}
else
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense; CHUNK=1750; FIRST0=2000
    NCHUNK=$(( (N - FIRST0 + CHUNK - 1) / CHUNK ))
fi
LOGS=$REPO/slurm_logs
mkdir -p $RAW

sim=$(sbatch --parsable --array=${CHUNKS:-0-$((NCHUNK - 1))}%1 ${AFTER:+--dependency=afterok:$AFTER} \
    --export=ALL,LOCATIONS=$LOCATIONS,RAW=$RAW,FIRST0=$FIRST0,CHUNK=$CHUNK $REPO/scripts/slurm/run_wofost_eu_grid_dense.sh)
echo "simulation + processing: array job $sim ($NCHUNK chunks of $CHUNK locations, rows $FIRST0..$((N - 1)))"
[ "${TEST:-0}" = 1 ] && exit 0

parts=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/processed"; done)
manifests=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/dataset_manifest.csv"; done)
merge=$(sbatch --parsable --dependency=afterok:$sim --job-name=wofost-dense-merge --partition=rome \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output=$LOGS/wofost-dense-merge_%j.out --error=$LOGS/wofost-dense-merge_%j.err --wrap="
cd $REPO && \
uv run python -m src.data_pipeline.wofost.merge_processed_tables $OLD/processed$parts -o $RAW/processed \
    --manifests $OLD/dataset_manifest.csv$manifests && \
uv run python -m analysis.scaling.table_share eu_grid_dense=$RAW/processed/wofost_maize_daily.parquet")
echo "merge + table shares: $merge -> $RAW/processed/wofost_maize_daily.parquet"

EXP=exp_1_eu_grid_dense TABLE=$RAW/processed/wofost_maize_daily.parquet DATAS=crop AFTER=$merge \
    TRAIN_TIME=08:00:00 TRAIN_MEM=120G bash $REPO/scripts/slurm/submit_exp1.sh
