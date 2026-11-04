#!/bin/bash
# Cultivar-variation run (TSUM1 and TSUM2 x U(0.8, 1.2) per location) on the densified OM-fix table, end to end
# (run on a login node: it only calls sbatch):
#
#   TEST=1 bash scripts/slurm/submit_eu_grid_dense_tsum.sh   # 12 locations: one small chunk, nothing else
#   bash scripts/slurm/submit_eu_grid_dense_tsum.sh          # everything
#
# 1. run_wofost_eu_grid_dense_tsum.sh: 7 chunks of 1,800 locations (two at a time), simulated + processed per chunk; soil from the
#    raw SoilGrids table, weather from the caches (nothing is downloaded).
# 2. merge the chunks' tables -> $RAW/processed/wofost_maize_daily.parquet.
# 3. exp_7_cultivar_tsum: pretraining seed 0 (log-uniform points per table, the recipe of exp_2a) on that table, then the CY-Bench
#    benchmark and a summary; the default-TabICL reference is copied from exp_1_eu_grid_dense_omfix.
# 4. data analysis (usable filter, diversity): analysis/cultivar_tsum/compare_data.py.

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
LOCATIONS=data/raw/locations/locations_maize_eu27_dense700_s42.csv
N=$(( $(wc -l < $REPO/$LOCATIONS) - 1 ))
LOGS=$REPO/slurm_logs
RAW_SOIL=/projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet
if [ "${TEST:-0}" = 1 ]; then
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_tsum_test; CHUNK=12; NCHUNK=1
else
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_tsum; CHUNK=1800
    NCHUNK=$(( (N + CHUNK - 1) / CHUNK ))
fi
mkdir -p $RAW

sim=$(sbatch --parsable --array=0-$((NCHUNK - 1))%2 \
    --export=ALL,LOCATIONS=$LOCATIONS,RAW=$RAW,RAW_SOIL=$RAW_SOIL,FIRST0=0,CHUNK=$CHUNK $REPO/scripts/slurm/run_wofost_eu_grid_dense_tsum.sh)
echo "simulation + processing: array job $sim ($NCHUNK chunks of $CHUNK rows of $N) -> $RAW"
[ "${TEST:-0}" = 1 ] && exit 0

parts=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/processed"; done)
manifests=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/dataset_manifest.csv"; done)
merge=$(sbatch --parsable --dependency=afterok:$sim --job-name=tsum-merge --partition=rome \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output=$LOGS/tsum-merge_%j.out --error=$LOGS/tsum-merge_%j.err --wrap="
cd $REPO && uv run python -m src.data_pipeline.wofost.merge_processed_tables $parts -o $RAW/processed --manifests $manifests --manifest-out $RAW/dataset_manifest.csv")
echo "merge: $merge -> $RAW/processed/wofost_maize_daily.parquet"

# memory 120G: the table is ~5.6 GB of parquet (the default 32G got the permuted-yield run killed)
export MICRO_BATCH=1
EXP=exp_7_cultivar_tsum REF_FROM=exp_1_eu_grid_dense_omfix TABLE=$RAW/processed/wofost_maize_daily.parquet DATAS=crop SEEDS=0 \
    EXTRA="POINTS_SAMPLING=loguniform,MIN_TABLE_POINTS=1" AFTER=$merge TRAIN_TIME=08:00:00 TRAIN_MEM=120G \
    bash $REPO/scripts/slurm/submit_exp1.sh

# data analysis (usable filter, diversity) once the merged table exists
dist=$(sbatch --parsable --dependency=afterok:$merge --job-name=tsum-data --partition=rome \
    --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --output=$LOGS/tsum-data_%j.out --error=$LOGS/tsum-data_%j.err --wrap="
cd $REPO && uv run python -m analysis.cultivar_tsum.compare_data")
echo "data analysis: $dist -> analysis/cultivar_tsum/"
