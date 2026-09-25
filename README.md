# Data-scarcity sweep for offline RL

How BC, TD3+BC, IQL, BCQ and CQL behave on hopper-medium as the dataset shrinks
from 1327 episodes to 13.

## Reproducing

```bash
python offline_rl_sweep.py --workers 8      # the full sweep, ~12h on 8 cores
python report_tables.py                     # tables from whatever has finished
```

The sweep is resumable: a job whose result JSON already exists is skipped, so
`results_scarcity/` is both the output and the checkpoint. With it in place
`report_tables.py` runs in a second and no training is needed.

## Two things worth knowing before you use this code

**Do not load D4RL data with `d3rlpy.datasets.get_minari()`.** As of 2.8.1 it
misaligns observations against actions, one step further per episode, and fails
silently. `load_dataset()` in the sweep script is the corrected loader. Check
any existing run by ridge-regressing actions on observations: correct data
explains about 36% of the variance, the broken path explains 2%.

**CQL diverges on d3rlpy's defaults.** Its Lagrange auto-tuning of the
conservative coefficient runs away; every seed collapses to ~0 after 10-15k
steps. `build_algo()` pins `alpha_learning_rate=0.0, conservative_weight=5.0`,
which is what the CQL paper uses for MuJoCo.

Results carry a `config_version` stamp. Change an algorithm's config and the
sweep re-runs anything produced under the old one, rather than silently reusing
it.

## Reading the numbers

Scores pool the last 5 checkpoints. Single-checkpoint numbers swing by up to
2000 points on this task, so reading the final checkpoint measures where
training stopped as much as it measures the algorithm.

hopper returns are bimodal: a policy either runs the full 1000 steps (~3600) or
falls partway (~1400). The mean is therefore close to a success rate, and
`report_tables.py` reports that share directly alongside it.
