#!/bin/bash
# JOB A: pre-train the negative agent (two danger critics) for seeds 0, 1, 2 on one MIG slice.
# Each run saves its most dangerous snapshot to models/neg_pretrained/<env>__seed<N>__neg.pt
#
# Submit:  sbatch run_pretrain_neg.sh
# Logs:    logs/pretrain/neg__<env>__s<seed>.log

#SBATCH --job-name=neg_pretrain
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out

ENV_NAME=hopper-medium-replay-v2
SEEDS=(0 1 2)
PRETRAIN_STEPS=500000
CORL_DIR=$HOME/Repositories/CORL

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p logs/pretrain models/neg_pretrained

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3     # 3 runs share the 9 CPU cores

echo "$(date '+%F %T')  Pre-training negative agents on $(hostname)"
for SEED in "${SEEDS[@]}"; do
  python algorithms/offline/td3_bc_neg_v2.py --mode pretrain_neg --env $ENV_NAME --seed $SEED \
    --max_timesteps $PRETRAIN_STEPS --eval_freq 10000 --neg_eval_episodes 3 \
    > logs/pretrain/neg__${ENV_NAME}__s${SEED}.log 2>&1 &
done
wait
echo "$(date '+%F %T')  Pre-training finished"
ls -l models/neg_pretrained
