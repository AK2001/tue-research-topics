"""Data-scarcity sweep for offline RL on hopper-medium.

Trains CQL / IQL / BCQ / TD3+BC (+ BC as reference) on nested random subsets of
`mujoco/hopper/medium-v0` and records how each one degrades as data shrinks.

Design notes
------------
* Subsets are *nested*: one permutation of episodes per seed, prefixes taken for
  each fraction. So the 10% subset is contained in the 25% subset, etc. This
  removes subset-composition noise from the comparison across data levels.
* The subset depends only on `seed`, never on the algorithm, so at a given
  (fraction, seed) all algorithms see byte-identical data.
* Each job runs in its own single-threaded process. Benchmarked on this machine,
  torch threads give almost no speedup on these small MLPs, so throughput comes
  from running many processes in parallel instead.
* Jobs are resumable: a job whose result JSON already exists is skipped.
* The dataset is loaded with `load_dataset()` below, NOT with
  `d3rlpy.datasets.get_minari()`. As of d3rlpy 2.8.1 that helper flattens every
  episode's observations including the terminal one (Minari stores T+1
  observations for T actions) and then re-slices by T, so episode k's
  observations end up shifted k steps against its actions. Episode 0 is fine and
  every later episode drifts further. It fails silently: BC's loss just sits at
  the action variance. Verified here by ridge-regressing actions on
  observations: R2 = 0.02 via get_minari vs 0.36 via the loader below.

Usage
-----
    python offline_rl_sweep.py --smoke            # ~3 min end-to-end check
    python offline_rl_sweep.py                    # full sweep
    python offline_rl_sweep.py --workers 6        # leave cores free
    python offline_rl_sweep.py --job IQL 0.25 0   # single job (used internally)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Must be set before torch is imported by anything else.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DATASET_ID = "mujoco/hopper/medium-v0"
RESULTS_DIR = Path(__file__).parent / "results_scarcity"
SMOKE_DIR = Path(__file__).parent / "results_smoke"

# Reference points for the normalized score, both measured on this machine
# (see notebook). RANDOM is the mean return of a uniform-random policy on
# Hopper-v5 over 30 episodes; BEHAVIOR is the mean episodic return of the full
# hopper-medium dataset. A normalized score of 100 therefore means "matches the
# policy that generated the data", not "matches a D4RL expert".
RANDOM_RETURN = 19.98
BEHAVIOR_RETURN = 2817.8

# Down to 1% (13 episodes) because the 10-100% sweep showed every algorithm
# sitting at the ceiling: 10% is still 133 episodes, which is not scarce.
# Bumped whenever build_algo() changes. Every result records the value it was
# run under, and drive() re-runs anything stored under an older one. Without
# this the resumable-jobs logic silently mixes configurations: the pre-fix CQL
# runs (auto-tuned alpha, collapsed to ~0) would have been reused as-is.
CONFIG_VERSION = "2026-09-24-cql-fixed-alpha"

FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
# 5 seeds, not 3: at the small fractions the between-seed spread is the whole
# story, and 3 seeds cannot separate a real effect from one unlucky subset.
SEEDS = [0, 1, 2, 3, 4]
ALGOS = ["BC", "TD3+BC", "IQL", "BCQ", "CQL"]


def load_dataset():
    """Minari -> d3rlpy ReplayBuffer with observations aligned to actions."""
    import minari
    import numpy as np
    from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer

    md = minari.load_dataset(DATASET_ID)
    episodes = []
    for e in md.iterate_episodes():
        episodes.append(Episode(
            observations=np.asarray(e.observations, dtype=np.float32)[:-1],
            actions=np.asarray(e.actions, dtype=np.float32),
            rewards=np.asarray(e.rewards, dtype=np.float32).reshape(-1, 1),
            terminated=bool(np.asarray(e.terminations)[-1]),
        ))
    return ReplayBuffer(InfiniteBuffer(), episodes=episodes)


def normalized(raw: float) -> float:
    return 100.0 * (raw - RANDOM_RETURN) / (BEHAVIOR_RETURN - RANDOM_RETURN)


def build_algo(name: str):
    import d3rlpy
    from d3rlpy.preprocessing import StandardObservationScaler

    # Same observation scaling for every algorithm so the comparison is fair.
    common = dict(batch_size=256, observation_scaler=StandardObservationScaler())

    if name == "BC":
        cfg = d3rlpy.algos.BCConfig(**common)
    elif name == "TD3+BC":
        cfg = d3rlpy.algos.TD3PlusBCConfig(**common)
    elif name == "IQL":
        cfg = d3rlpy.algos.IQLConfig(**common)
    elif name == "BCQ":
        # n_action_samples=10 matches the BCQ paper's training-time setting and
        # is ~3x faster than d3rlpy's default of 100.
        cfg = d3rlpy.algos.BCQConfig(n_action_samples=10, **common)
    elif name == "CQL":
        # alpha_learning_rate=0.0 disables d3rlpy's Lagrange auto-tuning of the
        # conservative coefficient. With it on (the default) CQL peaks around
        # 10-15k steps and then diverges to ~0 on every seed and every data
        # level; with a fixed alpha it holds ~3500 out to 40k. Fixed alpha is
        # also what the CQL paper uses for its MuJoCo results.
        cfg = d3rlpy.algos.CQLConfig(alpha_learning_rate=0.0,
                                     conservative_weight=5.0, **common)
    else:
        raise ValueError(f"unknown algo: {name}")
    return cfg.create(device="cpu")


def subsample(dataset, frac: float, seed: int):
    """Nested random subset of episodes; depends on `seed` only."""
    import numpy as np
    from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer

    episodes = dataset.episodes
    order = np.random.default_rng(1000 + seed).permutation(len(episodes))
    n = max(1, int(round(frac * len(episodes))))
    chosen = [episodes[i] for i in order[:n]]
    return ReplayBuffer(InfiniteBuffer(), episodes=chosen)


def result_path(algo: str, frac: float, seed: int, base: Path = RESULTS_DIR) -> Path:
    tag = algo.replace("+", "plus")
    return base / f"{tag}__frac{int(frac * 100):03d}__seed{seed}.json"


def evaluate_episodes(algo, env, n_trials: int, seed_base: int) -> list[float]:
    """Per-episode returns, not just their mean.

    The report needs median and worst over individual evaluation episodes, so
    d3rlpy's EnvironmentEvaluator (which only hands back the mean) is not enough.
    """
    import numpy as np

    returns = []
    for i in range(n_trials):
        obs, _ = env.reset(seed=seed_base + i)
        total = 0.0
        while True:
            action = algo.predict(np.expand_dims(np.asarray(obs, dtype=np.float32), 0))[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return returns


def run_job(algo_name: str, frac: float, seed: int, n_steps: int, eval_every: int,
            n_trials: int) -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import d3rlpy
    import gymnasium as gym
    import torch

    torch.set_num_threads(1)
    d3rlpy.seed(seed)

    dataset = load_dataset()
    subset = subsample(dataset, frac, seed)

    env = gym.make("Hopper-v5")
    env.reset(seed=10_000 + seed)

    algo = build_algo(algo_name)
    algo.build_with_dataset(subset)

    # Every checkpoint keeps its individual episode returns. Evaluation episodes
    # are seeded off the run seed so a re-run reproduces them exactly.
    checkpoints: list[dict] = []

    def on_epoch_end(trained, epoch, total_step):
        rets = evaluate_episodes(trained, env, n_trials,
                                 seed_base=20_000 + seed * 1_000 + epoch)
        checkpoints.append({"step": int(total_step), "returns": rets})

    t0 = time.time()
    algo.fit(
        subset,
        n_steps=n_steps,
        n_steps_per_epoch=eval_every,
        show_progress=False,
        logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
        epoch_callback=on_epoch_end,
    )
    elapsed = time.time() - t0

    curve = [(c["step"], float(np.mean(c["returns"]))) for c in checkpoints]
    raw_final = curve[-1][1]
    raw_best = max(s for _, s in curve)

    return {
        "algo": algo_name,
        "fraction": frac,
        "seed": seed,
        "config_version": CONFIG_VERSION,
        "n_episodes": subset.size(),
        "n_transitions": int(subset.transition_count),
        "n_steps": n_steps,
        "n_trials": n_trials,
        "raw_final": raw_final,
        "raw_best": raw_best,
        "norm_final": normalized(raw_final),
        "norm_best": normalized(raw_best),
        "curve": curve,
        "checkpoints": checkpoints,
        "final_returns": checkpoints[-1]["returns"],
        "elapsed_sec": elapsed,
    }


def drive(algos, fractions, seeds, n_steps, eval_every, n_trials, workers,
          results_dir: Path = RESULTS_DIR):
    results_dir.mkdir(exist_ok=True)

    def is_current(path: Path) -> bool:
        """True if `path` holds a result produced by the current build_algo()."""
        try:
            return json.loads(path.read_text()).get("config_version") == CONFIG_VERSION
        except (OSError, json.JSONDecodeError):
            return False

    jobs, stale = [], []
    for a in algos:
        for f in fractions:
            for s in seeds:
                path = result_path(a, f, s, results_dir)
                if not path.exists():
                    jobs.append((a, f, s))
                elif not is_current(path):
                    jobs.append((a, f, s))
                    stale.append(path.name)
    total = len(algos) * len(fractions) * len(seeds)
    if not jobs:
        print(f"All {total} jobs already done -> {results_dir}")
        return

    if stale:
        print(f"{len(stale)} result(s) predate {CONFIG_VERSION} and will be "
              f"re-run: {', '.join(sorted(stale)[:4])}"
              f"{' ...' if len(stale) > 4 else ''}")

    print(f"{len(jobs)} job(s) to run ({total - len(jobs)} already done), "
          f"{workers} parallel worker(s), {n_steps} steps each")

    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    running: list[tuple] = []
    done = 0
    t_start = time.time()

    def launch(job):
        algo, frac, seed = job
        cmd = [sys.executable, __file__, "--job", algo, str(frac), str(seed),
               "--steps", str(n_steps), "--eval-every", str(eval_every),
               "--trials", str(n_trials), "--results-dir", str(results_dir)]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        log = log_dir / (result_path(algo, frac, seed, results_dir).stem + ".log")
        handle = log.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                stderr=handle, text=True)
        return (job, proc, time.time(), handle, log)

    queue = list(jobs)
    while queue or running:
        while queue and len(running) < workers:
            running.append(launch(queue.pop(0)))

        time.sleep(2.0)

        still = []
        for entry in running:
            job, proc, started, handle, log = entry
            if proc.poll() is None:
                still.append(entry)
                continue
            handle.close()
            done += 1
            algo, frac, seed = job
            label = f"{algo} frac={frac} seed={seed}"
            if proc.returncode == 0:
                res = json.loads(result_path(algo, frac, seed, results_dir).read_text())
                print(f"[{done}/{len(jobs)}] {label:<28} "
                      f"norm_final={res['norm_final']:6.1f}  "
                      f"({time.time() - started:.0f}s)", flush=True)
            else:
                err = log.read_text().strip().splitlines()
                tail = err[-1] if err else "unknown error"
                print(f"[{done}/{len(jobs)}] {label:<28} FAILED: {tail[:160]}\n"
                      f"       full log: {log}", flush=True)
        running = still

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min -> {results_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", nargs=3, metavar=("ALGO", "FRAC", "SEED"),
                   help="run a single job and write its result JSON")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--algos", nargs="*", default=ALGOS,
                   help="algorithms to sweep (default: all)")
    p.add_argument("--fractions", nargs="*", type=float, default=FRACTIONS,
                   help="data fractions to sweep (default: all)")
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS,
                   help="seeds to sweep (default: all)")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--smoke", action="store_true",
                   help="tiny config to verify the pipeline end to end")
    args = p.parse_args()

    if args.job:
        algo, frac, seed = args.job[0], float(args.job[1]), int(args.job[2])
        res = run_job(algo, frac, seed, args.steps, args.eval_every, args.trials)
        args.results_dir.mkdir(exist_ok=True)
        result_path(algo, frac, seed, args.results_dir).write_text(json.dumps(res, indent=2))
        return

    if args.smoke:
        drive(["BC", "IQL"], [0.10, 1.00], [0], n_steps=2_000, eval_every=1_000,
              n_trials=5, workers=4, results_dir=SMOKE_DIR)
    else:
        drive(args.algos, args.fractions, args.seeds, args.steps, args.eval_every,
              args.trials, args.workers, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
