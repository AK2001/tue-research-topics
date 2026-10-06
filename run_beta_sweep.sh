#!/bin/bash
# JOB B: beta screening with the FROZEN pre-trained negative policies (run JOB A first).
# 5 beta values x 3 seeds = 15 runs of 300k steps, in 2 MIG jobs (9 + 6 runs).
# beta = 0 is plain TD3+BC (with safety logging) as the reference.
#
# Submit:   sbatch run_beta_sweep.sh
# Results:  python collect_results.py --logdir logs/sweep
#           python plot_runs.py --logdir logs/sweep --out sweep
# (collect_results.py will note "only 60/200 evaluations": expected, these are 300k-step runs)

#SBATCH --job-name=beta_sweep
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/%x_%A_%a.out

ENV_NAME=hopper-medium-replay-v2
SEEDS=(0 1 2)
STEPS=300000
CORL_DIR=$HOME/Repositories/CORL
LOGDIR=logs/sweep

case $SLURM_ARRAY_TASK_ID in
  0) BETAS=(0 0.05 0.1) ;;
  1) BETAS=(0.25 0.5) ;;
esac

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p $LOGDIR models/sweep

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "$(date '+%F %T')  Beta sweep job $SLURM_ARRAY_TASK_ID (betas ${BETAS[*]}) on $(hostname)"
for BETA in "${BETAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python algorithms/offline/td3_bc_neg_v2.py --mode train --env $ENV_NAME --seed $SEED \
      --beta $BETA --max_timesteps $STEPS --save_final_dir models/sweep \
      > $LOGDIR/v2_b${BETA}__${ENV_NAME}__s${SEED}.log 2>&1 &
  done
done
wait
echo "$(date '+%F %T')  Beta sweep job $SLURM_ARRAY_TASK_ID finished"
