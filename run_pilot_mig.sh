#!/bin/bash
# Pilot on one MIG slice: hopper-medium-replay-v2
#   normal TD3+BC  (td3_bc_neg.py --beta 0)    seeds 0, 1, 2
#   our method     (td3_bc_neg.py --beta 0.5)  seeds 0, 1, 2
# All 6 runs start at the same time and share the MIG slice.
#
# Submit:  sbatch run_pilot_mig.sh
# Logs:    logs/runs/<method>__hopper-medium-replay-v2__s<seed>.log

#SBATCH --job-name=td3bc_pilot
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# ------------------------- settings -------------------------
ENV_NAME=hopper-medium-replay-v2
SEEDS=(0 1 2)
BETA_NEG=0.5
CORL_DIR=$HOME/Repositories/CORL
# ------------------------------------------------------------

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p logs/runs

export WANDB_MODE=disabled   # results are read from the logs
export PYTHONUNBUFFERED=1    # write output to the log immediately
export OMP_NUM_THREADS=1     # 6 parallel runs share the 9 CPU cores

echo "$(date '+%F %T')  Pilot: $ENV_NAME, seeds ${SEEDS[*]}, beta 0 and $BETA_NEG, node $(hostname)"

for SEED in "${SEEDS[@]}"; do
  python algorithms/offline/td3_bc_neg.py --env $ENV_NAME --seed $SEED --beta 0 \
    > logs/runs/baseline__${ENV_NAME}__s${SEED}.log 2>&1 &
  python algorithms/offline/td3_bc_neg.py --env $ENV_NAME --seed $SEED --beta $BETA_NEG \
    > logs/runs/neg_b${BETA_NEG}__${ENV_NAME}__s${SEED}.log 2>&1 &
done
wait   # keep the job alive until all 6 runs have finished

echo "$(date '+%F %T')  Pilot finished"
