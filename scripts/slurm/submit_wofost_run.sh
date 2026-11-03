#!/bin/bash
# Simulate WOFOST for the densified EU grid and merge the chunks into one processed table (run on a login node: it only calls sbatch).
#
#   bash scripts/slurm/submit_wofost_run.sh omfix        # rainfed (water-limited), soil from the raw SoilGrids table: wofost_eu_grid_dense_omfix
#   bash scripts/slurm/submit_wofost_run.sh pp           # fully irrigated (potential production), no soil:        wofost_eu_grid_dense_pp
#   TEST=1 bash scripts/slurm/submit_wofost_run.sh omfix # 12 locations, one small chunk, written to <run>_test
#
# A variant is a RAW folder plus the settings of run_wofost_chunks.sh (soil table, offline weather, generate args, what is packed). Add one for a new
# simulation (e.g. cultivar variation: GEN_ARGS="--tsum-factor-range 0.8 1.2 --tsum-seed 0"). Chunks of 1,800 locations, two at a time. After the
# chunks a merge job writes $RAW/processed/wofost_maize_daily.parquet and the merged manifest. Training on the table: submit_experiment.sh.
# New regions (no raw soil table, nothing cached): leave SOIL_TABLE and OFFLINE_WEATHER empty; soil and weather are downloaded from Earth Engine.

set -euo pipefail
REPO=${REPO:-/gpfs/home2/vnedungadi/tnp-crop}
ROOT=/projects/prjs1788/tnp-crop
LOCATIONS=${LOCATIONS:-data/raw/locations/locations_maize_eu27_dense700_s42.csv}
SOIL_RAW=$ROOT/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet
LOGS=$REPO/slurm_logs
SBATCH=${SBATCH:-sbatch}   # SBATCH="echo sbatch" prints the commands (dry run)

case "${1:?variant: omfix | pp}" in
    omfix) RUN=wofost_eu_grid_dense_omfix; SOIL_TABLE=$SOIL_RAW; OFFLINE_WEATHER=1; GEN_ARGS=""; PACK="maize soil" ;;
    pp)    RUN=wofost_eu_grid_dense_pp;    SOIL_TABLE="";        OFFLINE_WEATHER=1; GEN_ARGS="--model-name Wofost81_PP"; PACK="maize" ;;
    *) echo "unknown variant $1"; exit 1 ;;
esac
LOC_FILE=$LOCATIONS; [ "${LOCATIONS:0:1}" = "/" ] || LOC_FILE=$REPO/$LOCATIONS
N=$(( $(wc -l < $LOC_FILE) - 1 ))
if [ "${TEST:-0}" = 1 ]; then RAW=$ROOT/${RUN}_test; CHUNK=12; NCHUNK=1; else RAW=$ROOT/$RUN; CHUNK=1800; NCHUNK=$(( (N + CHUNK - 1) / CHUNK )); fi
mkdir -p $RAW

sim=$($SBATCH --parsable --array=0-$((NCHUNK - 1))%2 --job-name=$RUN \
    --export=ALL,REPO=$REPO,LOCATIONS=$LOCATIONS,RAW=$RAW,CHUNK=$CHUNK,SOIL_TABLE=$SOIL_TABLE,OFFLINE_WEATHER=$OFFLINE_WEATHER,GEN_ARGS="$GEN_ARGS",PACK="$PACK" \
    $REPO/scripts/slurm/run_wofost_chunks.sh)
echo "simulation + processing: array job $sim ($NCHUNK chunks of $CHUNK rows of $N) -> $RAW"
[ "${TEST:-0}" = 1 ] && exit 0

parts=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/processed"; done)
manifests=$(for k in $(seq 0 $((NCHUNK - 1))); do echo -n " $RAW/part_$k/dataset_manifest.csv"; done)
merge=$($SBATCH --parsable --dependency=afterok:$sim --job-name=$RUN-merge --partition=rome --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output=$LOGS/$RUN-merge_%j.out --error=$LOGS/$RUN-merge_%j.err --wrap="
cd $REPO && uv run python -m src.data_pipeline.wofost.merge_processed_tables $parts -o $RAW/processed --manifests $manifests --manifest-out $RAW/dataset_manifest.csv")
echo "merge: $merge -> $RAW/processed/wofost_maize_daily.parquet"
