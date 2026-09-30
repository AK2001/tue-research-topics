"""
Reads all run logs in logs/runs/ and prints the results table (mean +- std over seeds).

Log file names must look like:  <method>__<dataset>__s<seed>.log
(run_experiments.sh creates them that way).

Usage (from the CORL folder):
    python collect_results.py              # final score = mean of last 10 evaluations
    python collect_results.py --last 5     # use the last 5 evaluations instead
Also writes results_per_run.csv with one row per run.
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

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/runs")
parser.add_argument("--last", type=int, default=10, help="average the last N evaluations")
args = parser.parse_args()

per_run = []  # (method, dataset, seed, final_score, n_evals)
for path in sorted(glob.glob(os.path.join(args.logdir, "*.log"))):
    name = os.path.basename(path)[:-4]
    try:
        method, dataset, seed = name.split("__")
        seed = int(seed.lstrip("s"))
    except ValueError:
        print(f"skipping {name}: unexpected file name")
        continue
    with open(path) as f:
        scores = [float(m.group(1)) for m in SCORE_RE.finditer(f.read())]
    if not scores:
        print(f"skipping {name}: no evaluations found (still running or crashed?)")
        continue
    if len(scores) < FULL_RUN_EVALS:
        print(f"note: {name} has only {len(scores)}/{FULL_RUN_EVALS} evaluations (unfinished?)")
    final = float(np.mean(scores[-args.last:]))
    per_run.append((method, dataset, seed, final, len(scores)))

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
print("\nPer-run scores saved to results_per_run.csv")
