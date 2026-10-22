#!/bin/bash
# Final comparison on ANY dataset: 1M steps, seeds 0, 1, 2, one MIG job, 6 runs in parallel:
#   v2_b0    TD3+BC (beta = 0)
#   v2_b0.5  our method (two danger critics, frozen pre-trained negative policy), beta = 0.5
# Same settings as run_final_v2.sh (hopper) and run_final_walker2d.sh.
# Needs the negative agents from run_pretrain_dataset.sh for the same dataset first.
#
# Submit:   sbatch jobs/run_final_dataset.sh <dataset>
#   e.g.    sbatch jobs/run_final_dataset.sh ant-medium-replay-v2
# Results:  python analysis/collect_results.py --logdir logs/final_<dataset>
#           python analysis/plot_runs.py --logdir logs/final_<dataset> --dataset <dataset> --out final_<dataset>

#SBATCH --job-name=final_v2
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out

ENV_NAME=${1:?"give the dataset name, e.g. sbatch jobs/run_final_dataset.sh ant-medium-replay-v2"}
SEEDS=(0 1 2)
BETAS=(0 0.5)
REPO_DIR=${SLURM_SUBMIT_DIR:-$PWD}   # submit from the repository root
LOGDIR=logs/final_${ENV_NAME}
MODELDIR=models/final_${ENV_NAME}

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd $REPO_DIR
mkdir -p $LOGDIR $MODELDIR

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "$(date '+%F %T')  Final runs for $ENV_NAME (betas ${BETAS[*]}, seeds ${SEEDS[*]}) on $(hostname)"
for BETA in "${BETAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python td3_bc_neg.py --mode train --env $ENV_NAME --seed $SEED \
      --beta $BETA --save_final_dir $MODELDIR \
      > $LOGDIR/v2_b${BETA}__${ENV_NAME}__s${SEED}.log 2>&1 &
  done
done
wait
echo "$(date '+%F %T')  Final runs finished"
