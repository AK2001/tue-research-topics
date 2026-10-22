"""
Reads all run logs in logs/runs/ and prints the results table (mean +- std over seeds).

Log file names must look like:  <method>__<dataset>__s<seed>.log
(run_experiments.sh creates them that way).

Usage (from the repository root):
    python analysis/collect_results.py                      # pilot logs (logs/runs)
    python analysis/collect_results.py --logdir logs/diag   # diagnostic study
    python analysis/collect_results.py --last 5             # use the last 5 evaluations instead
Also writes results_per_run.csv with one row per run.

If the logs contain per-episode lines ("Episode scores:", "Episode lengths:"),
a second table with SAFETY metrics is printed, computed over all episodes of
the last N evaluations of each run:
    fall rate = share of episodes that ended before MAX_STEPS (the robot fell)
    CVaR@10   = mean normalized score of the worst 10% of those episodes
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

SCORE_RE = re.compile(r"D4RL score:\s*(-?[\d.]+)")
FULL_RUN_EVALS = 200  # 1,000,000 steps / evaluation every 5,000 steps
MAX_STEPS = 1000      # episode limit in the D4RL locomotion tasks
EP_SCORES_RE = re.compile(r"Episode scores:\s*(.+)")
EP_LENGTHS_RE = re.compile(r"Episode lengths:\s*(.+)")

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/runs")
parser.add_argument("--last", type=int, default=10, help="average the last N evaluations")
args = parser.parse_args()

per_run = []  # (method, dataset, seed, final_score, n_evals)
safety = {}   # (method, dataset, seed) -> (fall_rate, cvar10)
for path in sorted(glob.glob(os.path.join(args.logdir, "*.log"))):
    name = os.path.basename(path)[:-4]
    try:
        method, dataset, seed = name.split("__")
        seed = int(seed.lstrip("s"))
    except ValueError:
        print(f"skipping {name}: unexpected file name")
        continue
    with open(path) as f:
        text = f.read()
    scores = [float(m.group(1)) for m in SCORE_RE.finditer(text)]
    ep_scores = [[float(x) for x in m.group(1).split(",")] for m in EP_SCORES_RE.finditer(text)]
    ep_lengths = [[int(x) for x in m.group(1).split(",")] for m in EP_LENGTHS_RE.finditer(text)]
    if not scores:
        print(f"skipping {name}: no evaluations found (still running or crashed?)")
        continue
    if len(scores) < FULL_RUN_EVALS:
        print(f"note: {name} has only {len(scores)}/{FULL_RUN_EVALS} evaluations (unfinished?)")
    final = float(np.mean(scores[-args.last:]))
    per_run.append((method, dataset, seed, final, len(scores)))
    if ep_scores and ep_lengths:
        eps = np.concatenate([np.array(x) for x in ep_scores[-args.last:]])
        lens = np.concatenate([np.array(x) for x in ep_lengths[-args.last:]])
        fall_rate = 100.0 * float(np.mean(lens < MAX_STEPS))
        k = max(1, int(round(0.1 * len(eps))))
        cvar10 = float(np.mean(np.sort(eps)[:k]))
        safety[(method, dataset, seed)] = (fall_rate, cvar10)

if not per_run:
    raise SystemExit("No results found.")

with open("results_per_run.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "dataset", "seed", "final_score", "n_evals"])
    w.writerows(per_run)

# group: (dataset, method) -> list of final scores
groups = defaultdict(list)
for method, dataset, seed, final, _ in per_run:
    groups[(dataset, method)].append(final)

datasets = sorted({d for d, _ in groups})
methods = sorted({m for _, m in groups})


def fmt(vals):
    if not vals:
        return "-"
    mean = np.mean(vals)
    std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:6.1f} +- {std:4.1f} (n={len(vals)})"


col = 24
print()
print(f"Final D4RL score = mean of last {args.last} evaluations; mean +- std over seeds")
print("dataset".ljust(32) + "".join(m.ljust(col) for m in methods))
print("-" * (32 + col * len(methods)))
for d in datasets:
    print(d.ljust(32) + "".join(fmt(groups.get((d, m), [])).ljust(col) for m in methods))

# Total row: sum of per-dataset means, only over datasets every method has
common = [d for d in datasets if all(groups.get((d, m)) for m in methods)]
print("-" * (32 + col * len(methods)))
totals = "".join(f"{sum(np.mean(groups[(d, m)]) for d in common):6.1f}".ljust(col) for m in methods)
print(f"Total ({len(common)} datasets)".ljust(32) + totals)
# ---------------- safety table (only if per-episode lines exist) ----------------
if safety:
    for title, idx, unit in (("Fall rate (% of episodes that fell)", 0, "%"),
                             ("CVaR@10 (mean score of the worst 10% of episodes)", 1, "")):
        print()
        print(f"{title}; last {args.last} evaluations per run; mean +- std over seeds")
        print("dataset".ljust(32) + "".join(m.ljust(col) for m in methods))
        print("-" * (32 + col * len(methods)))
        for d in datasets:
            row = d.ljust(32)
            for m in methods:
                vals = [v[idx] for (mm, dd, _), v in safety.items() if mm == m and dd == d]
                row += fmt(vals).ljust(col)
            print(row)

print("\nPer-run scores saved to results_per_run.csv")