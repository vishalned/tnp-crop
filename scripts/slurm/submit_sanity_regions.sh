#!/bin/bash
# Sanity check: exp_1 pretraining on regional subsets of the densified EU grid table, to see which part of
# Europe's WOFOST data the benchmark gain comes from (run on a login node: it only calls sbatch).
#
#   bash scripts/slurm/submit_sanity_regions.sh                 # all four subsets
#   SUBSETS="nw south" bash scripts/slurm/submit_sanity_regions.sh
#
# Per subset: one WOFOST seed (0), the exp_1 recipe (submit_exp1.sh, DATAS=crop), trained only on the subset's
# countries (TRAIN_COUNTRIES -> train.py --countries: rows of other countries are dropped before anything,
# features too), benchmarked on all 18 European CY-Bench countries. Default TabICL is copied from exp_1
# (REFERENCE=copy; deterministic). Micro-batch 1: tables reach 4096 rows (see run_gpu.sh).
# Outputs: nanotabicl/sanity_regions_<subset>/, cybench_output/sanity_regions_<subset>/ (summary.txt, per_run.csv).

set -euo pipefail
REPO=/gpfs/home2/vnedungadi/tnp-crop
TABLE=/projects/prjs1788/tnp-crop/wofost_eu_grid_dense/processed/wofost_maize_daily.parquet

# subset -> countries (names as in the table, ':'-separated); together the 19 countries of the table
declare -A REGION=(
    [nw]="France:Belgium:Netherlands:Germany:Denmark"
    [ce]="Poland:Czechia:Slovakia:Hungary:Romania:Bulgaria:Austria:Croatia:Slovenia:Lithuania"
    [south]="Spain:Portugal:Italy:Greece"
    [frbedk]="France:Belgium:Denmark"
)
SUBSETS=${SUBSETS:-"nw ce south frbedk"}

for subset in $SUBSETS; do
    echo "== sanity_regions_$subset: ${REGION[$subset]}"
    EXP=sanity_regions_$subset TABLE=$TABLE DATAS=crop SEEDS=0 REFERENCE=copy \
        TRAIN_COUNTRIES=${REGION[$subset]} MICRO_BATCH=1 TRAIN_TIME=08:00:00 TRAIN_MEM=120G \
        bash $REPO/scripts/slurm/submit_exp1.sh
done
