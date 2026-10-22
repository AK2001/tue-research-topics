#!/bin/bash
# Pre-train the negative agent (two danger critics) for seeds 0, 1, 2 on ANY dataset.
# Same settings as run_pretrain_neg.sh (hopper) and run_pretrain_neg_walker2d.sh.
# Each run saves its most dangerous snapshot to models/neg_pretrained/<dataset>__seed<N>__neg.pt
#
# Submit:  sbatch jobs/run_pretrain_dataset.sh <dataset>
#   e.g.   sbatch jobs/run_pretrain_dataset.sh ant-medium-replay-v2
# Logs:    logs/pretrain_<dataset>/neg__<dataset>__s<seed>.log

#SBATCH --job-name=neg_pretrain
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out

ENV_NAME=${1:?"give the dataset name, e.g. sbatch jobs/run_pretrain_dataset.sh ant-medium-replay-v2"}
SEEDS=(0 1 2)
PRETRAIN_STEPS=500000
REPO_DIR=${SLURM_SUBMIT_DIR:-$PWD}   # submit from the repository root
LOGDIR=logs/pretrain_${ENV_NAME}

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $REPO_DIR
mkdir -p $LOGDIR models/neg_pretrained

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3     # 3 runs share the 9 CPU cores

echo "$(date '+%F %T')  Pre-training negative agents for $ENV_NAME on $(hostname)"
for SEED in "${SEEDS[@]}"; do
  python td3_bc_neg.py --mode pretrain_neg --env $ENV_NAME --seed $SEED \
    --max_timesteps $PRETRAIN_STEPS --eval_freq 10000 --neg_eval_episodes 3 \
    > $LOGDIR/neg__${ENV_NAME}__s${SEED}.log 2>&1 &
done
wait
echo "$(date '+%F %T')  Pre-training finished"
ls -l models/neg_pretrained | grep "$ENV_NAME"
