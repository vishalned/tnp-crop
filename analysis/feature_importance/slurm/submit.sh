#!/bin/bash
# Prep (CPU) -> two GPU jobs (A: cybench for all 3 models + sim for default TabICL; B: sim for SCM and WOFOST).
set -euo pipefail
D=/gpfs/home2/vnedungadi/tnp-crop/analysis/feature_importance/slurm
prep=$(sbatch --parsable $D/prep.sh)
a=$(sbatch --parsable --dependency=afterok:$prep --export=ALL,WORK="wofost:cybench scm:cybench default_tabicl:cybench default_tabicl:sim" $D/gpu.sh)
b=$(sbatch --parsable --dependency=afterok:$prep --export=ALL,WORK="wofost:sim scm:sim" $D/gpu.sh)
echo "prep $prep -> GPU A $a, GPU B $b"
