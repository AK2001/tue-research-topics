#!/bin/bash
#SBATCH --job-name=iql-smoke
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke-%j.out
# ---------------------------------------------------------------------------
# Two short runs to prove the environment works before spending the allocation
# on the real thing. Checks the imports, that the dataset is in the cache (the
# compute nodes have no internet), that both code paths train, and that a
# result file lands on disk. A few minutes.
# ---------------------------------------------------------------------------
set -e

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/venvs/iql/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_smoke

echo "node $(hostname)"
python -c "import torch, minari, gymnasium; print('imports OK, torch', torch.__version__)"
python -c "
import minari
ds = minari.load_dataset('mujoco/walker2d/medium-v0')     # no download= : must already be cached
print('dataset OK:', ds.total_episodes, 'episodes')
"

for mode in none awr; do
    python -u iql_neg.py --device cpu --dataset mujoco/walker2d/medium-v0 \
        --steps 2000 --eval_freq 2000 --n_episodes 5 \
        --neg_mode "$mode" --beta_neg 0.5 --seed 0 --out results_smoke
done

echo
echo "result files: $(ls results_smoke | wc -l)  (expected 2)"
ls -la results_smoke
echo "smoke test passed -- now: sbatch snellius/run_1M.sh"
