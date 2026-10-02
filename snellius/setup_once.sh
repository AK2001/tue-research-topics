#!/bin/bash
# Run this ONCE on the Snellius LOGIN node (not in a job).
# Compute nodes have no internet, so the environment and the dataset must both
# exist before any job starts.
set -e

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv "$HOME/venvs/iql"
source "$HOME/venvs/iql/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install minari "gymnasium[mujoco]" h5py

# Pull the dataset now, on the login node, into the shared home directory.
# ~200 MB. The job will then find it in the Minari cache without any network.
python - <<'PY'
import minari
ds = minari.load_dataset("mujoco/walker2d/medium-v0", download=True)
print("cached:", ds.total_episodes, "episodes,", ds.total_steps, "steps")
PY

echo
echo "Done. Dataset cached under ~/.minari — submit the job with:"
echo "    sbatch snellius/run_1M.sh"
