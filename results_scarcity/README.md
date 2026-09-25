One JSON per run, named `<algo>__frac<NNN>__seed<N>.json`.

Each file holds the run's config, the full evaluation curve, and every
individual episode return at each checkpoint (20 checkpoints, 10 episodes
each). `report_tables.py` reads them; nothing else is needed.

The raw hopper-medium data is not here. It is a public Minari dataset
(`mujoco/hopper/medium-v0`, ~136 MB) that `offline_rl_sweep.py` downloads by
id on first run.
