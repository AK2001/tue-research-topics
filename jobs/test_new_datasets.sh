#!/bin/bash
# QUICK TEST (a few minutes, free) before submitting the real jobs.
# For each new dataset: a tiny pre-training (2,000 steps) and a tiny main training with beta 0.5
# (2,000 steps) that loads that pre-trained negative policy.
# Test models go to models/neg_test, so they can never be mixed up with the real ones.
#
# Run on int3 (no sbatch), from the repository root:  bash jobs/test_new_datasets.sh

DATASETS=(hopper-medium-v2 ant-medium-replay-v2 ant-medium-v2)

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate td3bc
cd "$(dirname "$0")/.."   # repository root
mkdir -p logs/test models/neg_test
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

for D in "${DATASETS[@]}"; do
  echo "=================== $D ==================="
  python td3_bc_neg.py --mode pretrain_neg --env $D --seed 0 \
    --max_timesteps 2000 --eval_freq 1000 --neg_eval_episodes 1 --neg_dir models/neg_test \
    > logs/test/pretrain__$D.log 2>&1
  echo "pre-training exit code: $?   (0 = ok)"
  grep -E "Pretrain step|Saved negative|Error|Traceback" logs/test/pretrain__$D.log | tail -n 4

  python td3_bc_neg.py --mode train --env $D --seed 0 --beta 0.5 \
    --max_timesteps 2000 --eval_freq 1000 --n_episodes 1 --neg_dir models/neg_test \
    > logs/test/train__$D.log 2>&1
  echo "main training exit code: $?   (0 = ok)"
  grep -E "Loaded frozen|D4RL score|Episode lengths|Error|Traceback" logs/test/train__$D.log | tail -n 5
done
echo "=================== done ==================="
ls -l models/neg_test
