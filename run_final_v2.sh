#!/bin/bash
# FINAL comparison at full length (1M steps) on seeds 0, 1, 2, one MIG job (6 runs in parallel):
#   v2_b0     plain TD3+BC (with safety logging)
#   v2_b0.5   fixed method: two danger critics + frozen pre-trained negative policy, beta 0.5
# Uses the negative agents already pre-trained for seeds 0-2 (models/neg_pretrained).
#
# Submit:   sbatch run_final_v2.sh
# Results:  python collect_results.py --logdir logs/final
#           python plot_runs.py --logdir logs/final --out final

#SBATCH --job-name=final_v2
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out

ENV_NAME=hopper-medium-replay-v2
SEEDS=(0 1 2)
BETAS=(0 0.5)
CORL_DIR=$HOME/Repositories/CORL
LOGDIR=logs/final

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $CORL_DIR
mkdir -p $LOGDIR models/final

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "$(date '+%F %T')  Final runs (betas ${BETAS[*]}, seeds ${SEEDS[*]}) on $(hostname)"
for BETA in "${BETAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python algorithms/offline/td3_bc_neg_v2.py --mode train --env $ENV_NAME --seed $SEED \
      --beta $BETA --save_final_dir models/final \
      > $LOGDIR/v2_b${BETA}__${ENV_NAME}__s${SEED}.log 2>&1 &
  done
done
wait
echo "$(date '+%F %T')  Final runs finished"
