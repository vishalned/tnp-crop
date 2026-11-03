#!/bin/bash
# Densified EU grid as potential production (fully irrigated, Wofost81_PP), end to end (run on a login node: it only calls sbatch):
#
#   TEST=1 bash scripts/slurm/submit_eu_grid_dense_pp.sh    # 6 locations, one chunk, nothing else
#   bash scripts/slurm/submit_eu_grid_dense_pp.sh           # 7 chunks of 1,800 rows (three at a time), then the merge
#
# Output /projects/prjs1788/tnp-crop/wofost_eu_grid_dense_pp/: per chunk part_k/{dataset_manifest.csv, processed/ (parquet +
# columns json), maize.tar.zst}, and the merged processed/ table + dataset_manifest.csv. Season files only on the node-local disk.

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
LOCATIONS=data/raw/locations/locations_maize_eu27_dense700_s42.csv
N=$(( $(wc -l < $REPO/$LOCATIONS) - 1 ))
LOGS=$REPO/slurm_logs
if [ "${TEST:-0}" = 1 ]; then
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_pp_test; CHUNK=6; NCHUNK=1
else
    RAW=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_pp; CHUNK=1800
    NCHUNK=$(( (N + CHUNK - 1) / CHUNK ))
fi
mkdir -p $RAW
sim=$(sbatch --parsable --array=0-$((NCHUNK - 1))%3 --export=ALL,LOCATIONS=$LOCATIONS,RAW=$RAW,FIRST0=0,CHUNK=$CHUNK \
    $REPO/scripts/slurm/run_wofost_eu_grid_dense_pp.sh)
echo "PP simulation + processing: array job $sim ($NCHUNK chunks of $CHUNK rows of $N)"
[ "${TEST:-0}" = 1 ] && exit 0

parts=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/processed"; done)
manifests=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/dataset_manifest.csv"; done)
merge=$(sbatch --parsable --dependency=afterok:$sim --job-name=pp-merge --partition=rome \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output=$LOGS/pp-merge_%j.out --error=$LOGS/pp-merge_%j.err --wrap="
cd $REPO && uv run python -m src.data_pipeline.wofost.merge_processed_tables $parts -o $RAW/processed --manifests $manifests --manifest-out $RAW/dataset_manifest.csv")
echo "merge: $merge -> $RAW/processed/wofost_maize_daily.parquet"
