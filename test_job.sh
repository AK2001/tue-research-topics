#!/bin/bash
# Short test on a MIG slice: original TD3+BC, our file with beta=0, our file with beta=0.1.
# Submit with:  sbatch test_job.sh
#SBATCH --job-name=td3bc_test
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $HOME/Repositories/CORL
export WANDB_MODE=disabled

ARGS="--env hopper-medium-replay-v2 --seed 0 --max_timesteps 10000 --eval_freq 5000 --n_episodes 5"

echo "===== 1. ORIGINAL CORL TD3+BC ====="
time python algorithms/offline/td3_bc.py $ARGS

echo "===== 2. NEG FILE, beta = 0 (should match 1) ====="
time python algorithms/offline/td3_bc_neg.py $ARGS --beta 0

echo "===== 3. NEG FILE, beta = 0.1 (your method) ====="
time python algorithms/offline/td3_bc_neg.py $ARGS --beta 0.1
