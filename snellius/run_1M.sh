#!/bin/bash
#SBATCH --job-name=iql-neg-1M
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
# ---------------------------------------------------------------------------
# 4 configurations x SEEDS seeds, 1,000,000 gradient steps each.
#
# CPU, not GPU, deliberately: the networks are two 256-unit layers, so each
# step is dominated by per-operation overhead rather than arithmetic. One
# process per core with a single thread each beats a GPU here, and a whole
# Genoa node gives us 192 cores for 60 concurrent runs.
# ---------------------------------------------------------------------------
set -e

# ---------------------------------------------------------------------------
# Override without editing this file:
#     sbatch --export=ALL,SEEDS=30 snellius/run_1M.sh
#
# A Genoa node has 192 cores and is billed whole, so unused cores are wasted
# money. 4 configurations x SEEDS runs must stay at or under 192:
#      5 seeds ->  20 runs   (first real run; 172 cores idle)
#     30 seeds -> 120 runs   (same wall time, same cost, far better statistics)
#     48 seeds -> 192 runs   (full node)
SEEDS=${SEEDS:-5}
# ---------------------------------------------------------------------------

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/venvs/iql/bin/activate"

# one thread per process, or the 60 runs fight each other for the same cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_1M

run () {   # mode  beta_neg  iql_tau  seed
    python -u iql_neg.py \
        --device cpu \
        --dataset mujoco/walker2d/medium-v0 \
        --steps 1000000 \
        --eval_freq 250000 \
        --n_episodes 100 \
        --neg_mode "$1" --beta_neg "$2" --iql_tau "$3" --seed "$4" \
        --out results_1M \
        > "logs/${1}_b${2}_tau${3}_seed${4}.log" 2>&1
}

echo "start $(date)  on $(hostname)  with $SLURM_CPUS_PER_TASK cores"
echo "$((4 * SEEDS)) runs, 1,000,000 steps each"

for seed in $(seq 0 $((SEEDS - 1))); do
    run none 0   0.7 "$seed" &     # standard IQL, the baseline
    run none 0   0.5 "$seed" &     # pessimistic IQL, the cheap alternative
    run awr  0.5 0.7 "$seed" &     # ours, IQL-native form
    run dist 0.5 0.7 "$seed" &     # ours, the TD3+BC port
done
wait

echo "done  $(date)"
echo "$(ls results_1M | wc -l) of $((4 * SEEDS)) runs produced a result file"
