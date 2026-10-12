"""
Plots the PRE-TRAINING statistics of the negative agent over training, one line per seed,
to see what happens before the danger critic breaks down.

Usage (from the CORL folder):
    python plot_pretrain.py --logdir logs/pretrain_walker2d --out pretrain_walker2d
    python plot_pretrain.py --logdir logs/pretrain          --out pretrain_hopper
Creates <out>.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"Pretrain step (\d+): Negative policy normalized score: (-?[\d.]+)")
STATS_RE = re.compile(r"Train stats:\s*(.+)")

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/pretrain_walker2d")
parser.add_argument("--out", default="pretrain_walker2d")
args = parser.parse_args()

KEYS = [
    ("neg_q_mean", "average danger value\n(should be negative)", True),
    ("danger_critic_loss", "danger critic loss", True),
    ("neg_to_data", "distance neg. policy\n<-> data", False),
    ("neg_lambda", "danger scaling factor", True),
    ("neg_actor_loss", "negative actor loss", False),
    ("score", "negative policy score\n(lower = more dangerous)", False),
]

runs = {}
for path in sorted(glob.glob(os.path.join(args.logdir, "*.log"))):
    seed = os.path.basename(path)[:-4].split("__")[-1]
    steps, rows = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        m = STEP_RE.search(line)
        if not m:
            continue
        row = {"score": float(m.group(2))}
        for nxt in lines[i + 1:i + 4]:
            s = STATS_RE.search(nxt)
            if s:
                for part in s.group(1).split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        row[k.strip()] = float(v)
        steps.append(int(m.group(1)))
        rows.append(row)
    snap = re.search(r"snapshot from step (\d+)", "\n".join(lines))
    runs[seed] = (np.array(steps), rows, int(snap.group(1)) if snap else None)
    print(f"{seed}: {len(steps)} evaluations, snapshot at {runs[seed][2]}")

fig, axes = plt.subplots(len(KEYS), 1, figsize=(9, 2.6 * len(KEYS)), sharex=True)
for ax, (key, label, symlog) in zip(axes, KEYS):
    for c, (seed, (steps, rows, snap)) in enumerate(sorted(runs.items())):
        vals = np.array([r.get(key, np.nan) for r in rows])
        ax.plot(steps / 1e3, vals, color=f"C{c}", label=seed)
        if snap is not None:
            ax.axvline(snap / 1e3, color=f"C{c}", ls=":", alpha=0.7)
    if symlog:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].legend(fontsize=8, title="seed (dotted = saved snapshot)", title_fontsize=8)
axes[-1].set_xlabel("pre-training steps (thousands)")
fig.suptitle(f"Negative-agent pre-training: {args.logdir}")
fig.tight_layout()
fig.savefig(f"{args.out}.png", dpi=150)
print(f"saved {args.out}.png")
