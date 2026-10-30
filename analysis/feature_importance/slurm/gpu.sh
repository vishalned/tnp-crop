#!/bin/bash
# One GPU job; WORK = space-separated "<model>:<data>" units, run one after the other (resumable).
#   sbatch --export=ALL,WORK="default_tabicl:cybench scm:cybench wofost:cybench wofost:sim" gpu.sh
#SBATCH --job-name=fi-gpu
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/feature_importance/gpu_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/feature_importance/gpu_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -uo pipefail
source /home/vnedungadi/AgML-CY-Bench-Modeling/cybench/runs/tabicl_wofost/env.sh
export PYTHONPATH=$TNP_DIR
cd "$BENCH_DIR"
for unit in $WORK; do
    echo "===== $unit $(date +%H:%M:%S)"
    poetry run python $TNP_DIR/analysis/feature_importance/run_importance.py --model ${unit%%:*} --data ${unit##*:} || echo "FAILED $unit"
done
echo "done $(date +%H:%M:%S)"
