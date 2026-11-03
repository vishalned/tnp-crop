#!/bin/bash
# exp_5 (context length): 4 models x {2,3,5 years} on DE FR HU ES, and "all years" on Senegal (SN). 4 GPU jobs (one per model),
# run in sequence-free parallel only 4 at a time at most. Results: /projects/prjs1788/tnp-crop/cybench_output/exp_5_context_years
set -euo pipefail
T=/projects/prjs1788/tnp-crop/nanotabicl
LOG=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/exp_5; mkdir -p $LOG
submit() {  # submit <tag base> <model> <ckpt or ""> <seed suffix>
    local tag=$1 model=$2 ckpt=$3 sfx=${4:-}
    for part in "ctx:DE FR HU ES:2 3 5" "sn:SN:all"; do
        IFS=: read -r pn countries levels <<<"$part"
        sbatch --parsable --job-name=exp5-$tag-$pn --output=$LOG/${tag}_${pn}_%j.out --error=$LOG/${tag}_${pn}_%j.err \
          --export="ALL,NAME=exp_5_context_years,TAG_BASE=$tag,SEED_SUFFIX=$sfx,MODEL=$model,CKPT=$ckpt,LEVELS=$levels,COUNTRIES=$countries" \
          /gpfs/home2/vnedungadi/tnp-crop/scripts/slurm/run_context_years.sh
    done
}
submit tabicl_default_single tabicl ""
submit wofost tabicl_wofost $T/exp_2a_loguniform_points/wofost_seed0/last.pt _seed0
submit wofost_irrmap tabicl_wofost $T/exp_3_irrigation_map/wofost_seed0/last.pt _seed0
submit scm tabicl_wofost $T/exp_1_eu_grid/scm_seed0/last.pt _seed0
