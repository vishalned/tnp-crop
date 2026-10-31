#!/bin/bash
# Densified EU grid after the organic-matter fix, end to end (run on a login node: it only calls sbatch):
#
#   TEST=1 bash scripts/slurm/submit_eu_grid_dense_omfix.sh    # 6 locations: soil download + one chunk, nothing else
#   bash scripts/slurm/submit_eu_grid_dense_omfix.sh           # everything
#   SKIP_SOIL=1 ...                                            # raw soil table already downloaded
#
# 1. download_raw_soil.sh: raw SoilGrids of all 12,382 locations -> one parquet (the only download).
# 2. run_wofost_eu_grid_dense_omfix.sh: 7 chunks of 1,800 rows (two at a time), simulated + processed per chunk.
# 3. merge the chunks' tables -> $RAW/processed/wofost_maize_daily.parquet.
# 4. exp_1 on it, WOFOST only (3 seeds) as exp_1_eu_grid_dense_omfix; the default-TabICL reference is copied
#    from exp_1_eu_grid_dense (not rerun). 5. old-vs-new distribution comparison (analysis/soc_fix).

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
LOCATIONS=data/raw/locations/locations_maize_eu27_dense700_s42.csv
N=$(( $(wc -l < $REPO/$LOCATIONS) - 1 ))
LOGS=$REPO/slurm_logs
if [ "${TEST:-0}" = 1 ]; then
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix_test; CHUNK=6; NCHUNK=1; MAX=6
    RAW_SOIL=/projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/test_soilgrids_raw_6.parquet
else
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix; CHUNK=1800; MAX=
    NCHUNK=$(( (N + CHUNK - 1) / CHUNK ))
    RAW_SOIL=/projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet
fi
mkdir -p $RAW

soil=""
if [ "${SKIP_SOIL:-0}" != 1 ]; then
    soil=$(sbatch --parsable --export=ALL,LOCATIONS=$LOCATIONS,OUT=$RAW_SOIL,MAX=$MAX $REPO/scripts/slurm/download_raw_soil.sh)
    echo "raw soil download: $soil -> $RAW_SOIL"
fi
sim=$(sbatch --parsable --array=0-$((NCHUNK - 1))%2 ${soil:+--dependency=afterok:$soil} \
    --export=ALL,LOCATIONS=$LOCATIONS,RAW=$RAW,RAW_SOIL=$RAW_SOIL,FIRST0=0,CHUNK=$CHUNK $REPO/scripts/slurm/run_wofost_eu_grid_dense_omfix.sh)
echo "simulation + processing: array job $sim ($NCHUNK chunks of $CHUNK rows of $N)"
[ "${TEST:-0}" = 1 ] && exit 0

parts=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/processed"; done)
manifests=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/dataset_manifest.csv"; done)
merge=$(sbatch --parsable --dependency=afterok:$sim --job-name=omfix-merge --partition=rome \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output=$LOGS/omfix-merge_%j.out --error=$LOGS/omfix-merge_%j.err --wrap="
cd $REPO && uv run python -m src.data_pipeline.wofost.merge_processed_tables $parts -o $RAW/processed --manifests $manifests --manifest-out $RAW/dataset_manifest.csv")
echo "merge: $merge -> $RAW/processed/wofost_maize_daily.parquet"

EXP=exp_1_eu_grid_dense_omfix REF_FROM=exp_1_eu_grid_dense TABLE=$RAW/processed/wofost_maize_daily.parquet DATAS=crop \
    AFTER=$merge TRAIN_TIME=08:00:00 TRAIN_MEM=120G bash $REPO/scripts/slurm/submit_exp1.sh

dist=$(sbatch --parsable --dependency=afterok:$merge --job-name=omfix-dist --partition=rome \
    --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --output=$LOGS/omfix-dist_%j.out --error=$LOGS/omfix-dist_%j.err --wrap="
cd $REPO && uv run python -m analysis.soc_fix.compare_distributions")
echo "distribution comparison: $dist -> analysis/soc_fix/"
