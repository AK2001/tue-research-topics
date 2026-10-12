#!/bin/bash
# WALKER2D-MEDIUM-REPLAY: pre-train the negative agent (two danger critics), seeds 0, 1, 2, on one MIG slice.
# Each run saves its most dangerous snapshot to models/neg_pretrained/<dataset>__seed<N>__neg.pt
# (Hopper used run_pretrain_neg.sh; this file is only for walker2d-medium-replay-v2.)
#
# Submit:  sbatch run_pretrain_neg_walker2d.sh
# Logs:    logs/pretrain_walker2d/neg__<dataset>__s<seed>.log

#SBATCH --job-name=neg_pretrain_walker
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# ------------------------- settings -------------------------
ENV_NAME=walker2d-medium-replay-v2
SEEDS=(0 1 2)
PRETRAIN_STEPS=500000
CORL_DIR=$HOME/Repositories/CORL
# ------------------------------------------------------------

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p logs/pretrain_walker2d models/neg_pretrained

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3     # 3 runs share the 9 CPU cores

echo "$(date '+%F %T')  Pre-training negative agents for $ENV_NAME on $(hostname)"
for SEED in "${SEEDS[@]}"; do
  python algorithms/offline/td3_bc_neg_v2.py --mode pretrain_neg --env $ENV_NAME --seed $SEED \
    --max_timesteps $PRETRAIN_STEPS --eval_freq 10000 --neg_eval_episodes 3 \
    > logs/pretrain_walker2d/neg__${ENV_NAME}__s${SEED}.log 2>&1 &
done
wait
echo "$(date '+%F %T')  Pre-training finished"
ls -l models/neg_pretrained
