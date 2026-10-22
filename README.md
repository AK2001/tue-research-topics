# TD3+BC with a negative policy

This is the code for our project in 2AMM20 Safe Reinforcement Learning (Group 13).

The idea is simple. We first train a "negative policy" with TD3+BC on the inverted reward, so it learns
bad behaviour. We then freeze it and train a normal TD3+BC policy with an extra loss term that pushes it
away from the actions of the negative policy. With `--beta 0` the code is just plain TD3+BC.

The code is based on the TD3+BC implementation from [CORL](https://github.com/corl-team/CORL), but it runs
on its own.

## Folders

- `td3_bc_neg.py` is the method itself.
- `jobs/` has the scripts we used to run the experiments on Snellius.
- `analysis/` has the scripts we used to make the tables and plots from the logs.
- `environment/` has the conda environment and the pip packages we used.
- `results/logs/` has the logs of the runs we report in the paper.

## Setup

We used Python 3.9 with MuJoCo 2.1 and D4RL. You can recreate our environment with:

```
conda env create -f environment/environment.yml
conda activate td3bc
```

This file was exported on Snellius (Linux), so if it does not work on your machine you can also try
`pip install -r environment/requirements.txt`.

The datasets have to be downloaded first, on a machine with internet access:

```
python -c "import gym, d4rl; gym.make('hopper-medium-replay-v2').get_dataset()"
```

## Running

All commands are run from the main folder of the repository. The examples use hopper-medium-v2 and
seed 0, but any D4RL locomotion dataset and seed works the same way.

We did not use Weights & Biases, so turn it off first:

```
export WANDB_MODE=disabled
```

**1. Pre-train the negative policy.** This trains it for 500,000 steps and saves its most dangerous
version to `models/neg_pretrained/`.

```
mkdir -p logs/pretrain_hopper-medium-v2

python td3_bc_neg.py --mode pretrain_neg --env hopper-medium-v2 --seed 0 --max_timesteps 500000 --eval_freq 10000 --neg_eval_episodes 3 > logs/pretrain_hopper-medium-v2/neg__hopper-medium-v2__s0.log
```

**2. Train TD3+BC and our method.** `--beta 0` is plain TD3+BC, `--beta 0.5` is our method, which loads
the negative policy from step 1. Both train for 1 million steps.

```
mkdir -p logs/final_hopper-medium-v2

python td3_bc_neg.py --mode train --env hopper-medium-v2 --seed 0 --beta 0 --save_final_dir models/final_hopper-medium-v2 > logs/final_hopper-medium-v2/v2_b0__hopper-medium-v2__s0.log

python td3_bc_neg.py --mode train --env hopper-medium-v2 --seed 0 --beta 0.5 --save_final_dir models/final_hopper-medium-v2 > logs/final_hopper-medium-v2/v2_b0.5__hopper-medium-v2__s0.log
```

Keep the log names in this form (`<method>__<dataset>__s<seed>.log`), because the analysis scripts read
the method, dataset and seed from the file name.

**3. Collect the results.** This prints the score, the fall rate and the CVaR@10 of both methods,
averaged over the seeds in the folder, and makes the learning curves.

```
python analysis/collect_results.py --logdir logs/final_hopper-medium-v2

python analysis/plot_runs.py --logdir logs/final_hopper-medium-v2 --dataset hopper-medium-v2 --out final_hopper-medium-v2
```

**On Snellius** we did not run these by hand. The job scripts in `jobs/` do steps 1 and 2 for seeds 0,
1 and 2 at once, and the second job starts by itself when the first one is done:

```
mkdir -p logs/slurm
J=$(sbatch --parsable jobs/run_pretrain_dataset.sh hopper-medium-v2)

sbatch --dependency=afterok:$J jobs/run_final_dataset.sh hopper-medium-v2
```

On one GPU slice the pre-training took about 1 hour and the training about 4 hours.

## Notes

- `td3_bc_neg.py` was called `td3_bc_neg_v2.py` during the project. The first version is still in the
  git history.
- The scripts in `jobs/original/` are the exact ones we used for hopper-medium-replay and
  walker2d-medium-replay. They still use the old folder layout of our Snellius setup.
- In `results/logs/`, the folders `final` and `pretrain` belong to hopper-medium-replay.
