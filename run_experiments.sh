#!/bin/bash
# Full experiments: 9 D4RL datasets x 2 methods = 18 array tasks.
# Each task runs all seeds of one (dataset, method) IN PARALLEL on one MIG GPU,
# so several runs share the cost of one GPU slice.
#
# Submit everything:              sbatch run_experiments.sh
# Submit only hopper-medium-replay (baseline + neg):  sbatch --array=8-9 run_experiments.sh
#
# Task numbering: task = dataset_index * 2 + method_index
#   dataset 0 halfcheetah-medium        -> tasks 0,1
#   dataset 1 halfcheetah-medium-replay -> tasks 2,3
#   dataset 2 halfcheetah-medium-expert -> tasks 4,5
#   dataset 3 hopper-medium             -> tasks 6,7
#   dataset 4 hopper-medium-replay      -> tasks 8,9
#   dataset 5 hopper-medium-expert      -> tasks 10,11
#   dataset 6 walker2d-medium           -> tasks 12,13
#   dataset 7 walker2d-medium-replay    -> tasks 14,15
#   dataset 8 walker2d-medium-expert    -> tasks 16,17
#   method 0 = baseline (original CORL td3_bc.py), method 1 = negative policy

#SBATCH --job-name=td3bc_exp
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00            # ADJUST after the test job tells us the real run time
#SBATCH --array=0-17
#SBATCH --output=logs/slurm/%x_%A_%a.out

# ------------------------- settings you may change -------------------------
SEEDS=(0 1 2)          # 3 seeds; use (0 1 2 3 4) for 5 seeds
BETA_NEG=0.1           # beta for the negative-policy method
CORL_DIR=$HOME/Repositories/CORL
# ---------------------------------------------------------------------------

ENVS=(
  halfcheetah-medium-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-expert-v2
  hopper-medium-v2      hopper-medium-replay-v2      hopper-medium-expert-v2
  walker2d-medium-v2    walker2d-medium-replay-v2    walker2d-medium-expert-v2
)
METHODS=(baseline neg)

ENV_NAME=${ENVS[$((SLURM_ARRAY_TASK_ID / 2))]}
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID % 2))]}

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p logs/runs

export WANDB_MODE=disabled   # we read results from the logs; skips writing wandb files every step
export OMP_NUM_THREADS=2     # share the 9 CPU cores between the parallel runs (use 1 for 5 seeds)

if [ "$METHOD" = "baseline" ]; then
  SCRIPT=algorithms/offline/td3_bc.py
  EXTRA=""
  TAG=baseline
else
  SCRIPT=algorithms/offline/td3_bc_neg.py
  EXTRA="--beta $BETA_NEG"
  TAG=neg_b${BETA_NEG}
fi

echo "Task $SLURM_ARRAY_TASK_ID: $TAG on $ENV_NAME, seeds ${SEEDS[*]}, node $(hostname)"

for SEED in "${SEEDS[@]}"; do
  LOG=logs/runs/${TAG}__${ENV_NAME}__s${SEED}.log
  python $SCRIPT --env $ENV_NAME --seed $SEED $EXTRA > "$LOG" 2>&1 &
done
wait   # keep the job alive until all seeds have finished

echo "Task $SLURM_ARRAY_TASK_ID finished"
