#!/bin/bash
# CPU prep: dump the CY-Bench walk-forward folds of the 7 countries (benchmark pipeline, no model) and
# build the 100 held-out simulated tables. Outputs in /projects/prjs1788/tnp-crop/feature_importance/.
#SBATCH --job-name=fi-prep
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/feature_importance/prep_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/feature_importance/prep_%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -uo pipefail
source /home/vnedungadi/AgML-CY-Bench-Modeling/cybench/runs/tabicl_wofost/env.sh
export PYTHONPATH=$TNP_DIR
OUT=/projects/prjs1788/tnp-crop/feature_importance
FI=$TNP_DIR/analysis/feature_importance
cd "$BENCH_DIR"
for c in ${COUNTRIES:-DE FR IT ES RO HU PL}; do
    [ -f $OUT/folds/folds_$c.npz ] && { echo "skip $c"; continue; }
    echo "===== $c $(date +%H:%M:%S)"
    poetry run python $FI/dump_folds.py --country $c --out $OUT/folds || echo "FAILED $c"
done
# simulated tables: the repo's uv env (it has pyarrow; the benchmark's poetry env does not)
cd "$TNP_DIR"
[ -f $OUT/sim_tables/index.json ] || uv run --no-sync python -m analysis.feature_importance.build_sim_tables \
    --table /projects/prjs1788/tnp-crop/wofost_eu_grid_dense/processed/wofost_maize_daily.parquet \
    --out $OUT/sim_tables --n-tables 100 --shards 4
