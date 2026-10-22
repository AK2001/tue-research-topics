"""
Summarizes the diagnostic values in the logs, per setting, to explain WHY a setting
scores as it does. Uses the 'Train stats' and 'Negative policy normalized score' lines.

Usage (from the repository root):
    python analysis/diag_summary.py --logdir logs/diag

For each value: mean over the last N evaluations of each run, then mean +- std over seeds.
For the danger values it also shows how much they swing during training.
"""
import argparse
import ast
import glob
import os
import re
from collections import defaultdict

import numpy as np

STATS_RE = re.compile(r"Train stats:\s*(.+)")
NEG_RE = re.compile(r"Negative policy normalized score:\s*(-?[\d.]+)")

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/diag")
parser.add_argument("--dataset", default="hopper-medium-replay-v2")
parser.add_argument("--last", type=int, default=10)
args = parser.parse_args()


def parse_stats(text):
    text = text.strip()
    if text.startswith("{"):
        return {k: float(v) for k, v in ast.literal_eval(text).items()}
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


# method -> seed -> {"stats": [dict per eval], "neg": [score per eval]}
data = defaultdict(dict)
for path in sorted(glob.glob(os.path.join(args.logdir, f"*__{args.dataset}__s*.log"))):
    method, _, seed = os.path.basename(path)[:-4].split("__")
    stats, neg = [], []
    with open(path) as f:
        for line in f:
            m = STATS_RE.search(line)
            if m:
                stats.append(parse_stats(m.group(1)))
            m = NEG_RE.search(line)
            if m:
                neg.append(float(m.group(1)))
    data[method][seed] = {"stats": stats, "neg": neg}

ROWS = [
    ("bc_mse",        "distance policy <-> data          (A: does it leave the data?)"),
    ("q_mean",        "critic value of policy actions    (A: is the critic over-optimistic?)"),
    ("dist_to_neg",   "distance policy <-> neg. policy   (is it pushed away?)"),
    ("neg_to_data",   "distance neg. policy <-> data     (D: is the 'bad' policy just the data?)"),
    ("neg_q_mean",    "average danger value              (B: close to zero?)"),
    ("neg_lambda",    "danger scaling factor             (B: large / jumping?)"),
    ("neg_score",     "score of the negative policy      (C: is it actually bad?)"),
]

methods = sorted(data)


def last_mean(values):
    v = [x for x in values[-args.last:] if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else None


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3g}"
    return f"{np.mean(vals):.3g} +- {np.std(vals, ddof=1):.2g}"


col = 22
print(f"\nDiagnostics, {args.dataset}: mean of last {args.last} evaluations, mean +- std over seeds")
print("value".ljust(14) + "".join(m.ljust(col) for m in methods))
print("-" * (14 + col * len(methods)))
for key, desc in ROWS:
    row = key.ljust(14)
    for m in methods:
        per_seed = []
        for s in data[m].values():
            if key == "neg_score":
                per_seed.append(last_mean(s["neg"]) if s["neg"] else None)
            else:
                per_seed.append(last_mean([d.get(key, np.nan) for d in s["stats"]]))
        row += fmt(per_seed).ljust(col)
    print(row)
print()
for key, desc in ROWS:
    print(f"  {key:12s} {desc}")

# how much do the danger values swing over the whole run?
print(f"\nDanger values over the WHOLE run (min .. max, all seeds together):")
for m in methods:
    q = [d["neg_q_mean"] for s in data[m].values() for d in s["stats"] if "neg_q_mean" in d]
    lam = [d["neg_lambda"] for s in data[m].values() for d in s["stats"] if "neg_lambda" in d]
    if q:
        print(f"  {m:16s} neg_q_mean {min(q):9.2f} .. {max(q):9.2f}    "
              f"neg_lambda {min(lam):8.3f} .. {max(lam):8.3f}")
