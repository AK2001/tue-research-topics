#!/bin/bash
# WALKER2D-MEDIUM-REPLAY final comparison: 1M steps, seeds 0, 1, 2, one MIG job, 6 runs in parallel:
#   v2_b0    TD3+BC (beta = 0), with safety logging
#   v2_b0.5  fixed method (two danger critics, frozen pre-trained negative policy), beta = 0.5
# Run run_pretrain_neg_walker2d.sh first.
# (Hopper used run_final_v2.sh; this file is only for walker2d-medium-replay-v2.)
#
# Submit:   sbatch run_final_walker2d.sh
# Results:  python collect_results.py --logdir logs/final_walker2d-medium-replay-v2
#           python plot_runs.py --logdir logs/final_walker2d-medium-replay-v2 --dataset walker2d-medium-replay-v2 --out final_walker2d

#SBATCH --job-name=final_walker
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# ------------------------- settings -------------------------
ENV_NAME=walker2d-medium-replay-v2
SEEDS=(0 1 2)
BETAS=(0 0.5)
CORL_DIR=$HOME/Repositories/CORL
# ------------------------------------------------------------
LOGDIR=logs/final_${ENV_NAME}

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p $LOGDIR models/final_walker2d

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "$(date '+%F %T')  Final run for $ENV_NAME (betas ${BETAS[*]}, seeds ${SEEDS[*]}, 1M steps) on $(hostname)"
for BETA in "${BETAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python algorithms/offline/td3_bc_neg_v2.py --mode train --env $ENV_NAME --seed $SEED \
      --beta $BETA --save_final_dir models/final_walker2d \
      > $LOGDIR/v2_b${BETA}__${ENV_NAME}__s${SEED}.log 2>&1 &
  done
done
wait
echo "$(date '+%F %T')  Final run finished"
