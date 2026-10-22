"""
Plots learning curves (D4RL score over training) and the training statistics
from the run logs, to see HOW the methods differ, not just their final score.

Usage (from the folder that contains logs/runs/):
    python analysis/plot_runs.py
    python analysis/plot_runs.py --dataset hopper-medium-replay-v2
    python analysis/plot_runs.py --logdir logs/diag --out diag
Creates:
    plot_scores.png   mean +- std D4RL score over training, per method
                      (dashed: score of the negative policy itself, if evaluated)
    plot_stats.png    training statistics over time (losses, Q, distances), per method
Needs: numpy, matplotlib   (pip install matplotlib)
"""
import argparse
import ast
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"Time steps:\s*(\d+)")
SCORE_RE = re.compile(r"D4RL score:\s*(-?[\d.]+)")
STATS_RE = re.compile(r"Train stats:\s*(.+)")
NEG_RE = re.compile(r"Negative policy normalized score:\s*(-?[\d.]+)")

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/runs")
parser.add_argument("--dataset", default="hopper-medium-replay-v2")
parser.add_argument("--out", default="plot", help="prefix of the output png files")
args = parser.parse_args()


def parse_stats(text):
    """Accepts both  {'a': 1.0, ...}  and  a=1.0, b=2.0  formats."""
    text = text.strip()
    if text.startswith("{"):
        return {k: float(v) for k, v in ast.literal_eval(text).items()}
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


# method -> list of runs; each run = (steps, scores, stats_list)
runs = defaultdict(list)
for path in sorted(glob.glob(os.path.join(args.logdir, f"*__{args.dataset}__s*.log"))):
    method = os.path.basename(path).split("__")[0]
    steps, scores, stats, neg = [], [], [], []
    current_step = None
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
            m = SCORE_RE.search(line)
            if m and current_step is not None:
                steps.append(current_step)
                scores.append(float(m.group(1)))
            m = STATS_RE.search(line)
            if m:
                stats.append(parse_stats(m.group(1)))
            m = NEG_RE.search(line)
            if m:
                neg.append(float(m.group(1)))
    if scores:
        runs[method].append((np.array(steps), np.array(scores), stats, np.array(neg)))
        print(f"{method:14s} seed file {os.path.basename(path)}: {len(scores)} evaluations")

if not runs:
    raise SystemExit(f"No logs found for {args.dataset} in {args.logdir}")


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


# ---------- 1. learning curves ----------
plt.figure(figsize=(8, 5))
for method, rs in sorted(runs.items()):
    n = min(len(r[1]) for r in rs)
    steps = rs[0][0][:n]
    scores = np.stack([r[1][:n] for r in rs])
    mean, std = scores.mean(0), scores.std(0)
    plt.plot(steps / 1e6, mean, label=f"{method} (n={len(rs)})")
    plt.fill_between(steps / 1e6, mean - std, mean + std, alpha=0.2)
    negs = [r[3] for r in rs if len(r[3])]
    if negs:  # score of the negative policy itself (dashed)
        m = min(len(x) for x in negs)
        plt.plot(steps[:m] / 1e6, np.stack([x[:m] for x in negs]).mean(0), "--",
                 color=plt.gca().lines[-1].get_color(), alpha=0.7, label=f"{method}: negative policy")
plt.xlabel("training steps (millions)")
plt.ylabel("D4RL normalized score")
plt.title(f"Learning curves: {args.dataset} (mean +- std over seeds)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{args.out}_scores.png", dpi=150)
print(f"saved {args.out}_scores.png")

# ---------- 2. training statistics ----------
keys = sorted({k for rs in runs.values() for r in rs for s in r[2] for k in s})
if keys:
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.6 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for method, rs in sorted(runs.items()):
            for i, (steps, _, stats, _neg) in enumerate(rs):
                vals = [s.get(key, np.nan) for s in stats]
                if np.all(np.isnan(vals)):
                    continue
                x = steps[: len(vals)] / 1e6
                y = smooth(np.array(vals[: len(x)]))
                ax.plot(x[: len(y)], y, alpha=0.7,
                        color=f"C{sorted(runs).index(method)}",
                        label=method if i == 0 else None)
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("training steps (millions)")
    fig.suptitle(f"Training statistics: {args.dataset} (smoothed, one line per seed)")
    fig.tight_layout()
    fig.savefig(f"{args.out}_stats.png", dpi=150)
    print(f"saved {args.out}_stats.png")
else:
    print(f"no 'Train stats' lines found; only {args.out}_scores.png was made")