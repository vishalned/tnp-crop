#!/bin/bash

#SBATCH --job-name=run_cybench_cropfm_best_params
#SBATCH --output=/gpfs/home2/vnedungadi/CropBench/slurm_logs/run_cybench_cropfm_best_params_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/CropBench/slurm_logs/run_cybench_cropfm_best_params_%j.err
#SBATCH --mem=16G
#SBATCH --partition=rome
#SBATCH --time=05:00:00

echo "Job started at:  $(date)"
start_ts=$(date +%s)

source ~/.bashrc
conda activate cropbench

python train.py \
    +experiment=cybench/cropfm \
    preprocessing.normalize.force_recompute=true \
    seed=32 \
    trainer=walk_forward \
    dataset.dataset_name=maize_DE \
    trainer.max_epochs=50 \
    trainer.stage2.lr=1e-4 \
    trainer.warmup_epochs=0 \
    trainer.wrapped_trainer_type=lightning_fm \
    model.finetune_method=full


echo "Job finished at:  $(date)"
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

echo "Job runtime: ${elapsed} seconds (~$((elapsed/60)) minutes)"