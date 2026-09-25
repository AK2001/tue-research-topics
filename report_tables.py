"""Render the data-scarcity sweep as per-algorithm tables.

Numbers are pooled over the LAST FEW CHECKPOINTS, not the final one. On hopper
every algorithm's evaluation score swings between ~1400 and ~3600 from one
checkpoint to the next, so a single-checkpoint table mostly measures which
checkpoint you happened to stop on: at 100k steps TD3+BC's final checkpoint and
its last-half average agree, but BC's final checkpoint (2342) sits 250 below its
own last-half average, and the 25k-step ranking is essentially a coin toss.
Averaging a window of checkpoints removes that.

    mean       mean over every evaluation episode in the window, seeds pooled
    +/-seed std  standard deviation of the per-seed windowed mean returns
    median     median over every evaluation episode in the window
    worst      single worst evaluation episode in the window
    ok%        share of evaluation episodes scoring >= 3000

`ok%` is there because hopper returns are bimodal, not spread out: a policy
either runs the full 1000 steps (~3600) or falls over partway (~1400), with
little in between. So the mean is really a disguised success rate, and reading
it directly is clearer than inferring it from a mean/median gap.

Works on a partial sweep, so it can be run while jobs are still going.

Usage:
    python report_tables.py                # last 5 checkpoints (default)
    python report_tables.py --window 1     # final checkpoint only (old behaviour)
    python report_tables.py --at 25000     # one specific training step
    python report_tables.py --markdown     # fenced blocks, ready to paste
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results_scarcity"
RANDOM_RETURN = 19.98
BEHAVIOR_RETURN = 2817.8
# Midpoint of the empty band between hopper's two outcomes (falls ~1400-2000,
# runs the full episode ~3500+). Anything above this ran to the time limit.
SUCCESS_RETURN = 3000.0
DEFAULT_WINDOW = 5
ALGO_ORDER = ["BC", "TD3+BC", "IQL", "BCQ", "CQL"]


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {path.name})")
    return out


def returns_for(run: dict, step: int | None, window: int) -> list[float]:
    """Per-episode returns at `step`, or pooled over the last `window` checkpoints."""
    if step is not None:
        for cp in run["checkpoints"]:
            if cp["step"] == step:
                return cp["returns"]
        return []
    return [r for cp in run["checkpoints"][-window:] for r in cp["returns"]]


def stats(runs: list[dict], step: int | None, window: int) -> dict | None:
    per_seed_means, pooled = [], []
    for run in runs:
        rets = returns_for(run, step, window)
        if not rets:
            continue
        per_seed_means.append(float(np.mean(rets)))
        pooled.extend(rets)
    if not pooled:
        return None
    return {
        "mean": float(np.mean(pooled)),
        "seed_std": float(np.std(per_seed_means)),
        "median": float(np.median(pooled)),
        "worst": float(np.min(pooled)),
        "success": 100.0 * float(np.mean(np.asarray(pooled) >= SUCCESS_RETURN)),
        "n_seeds": len(per_seed_means),
        "n_episodes": len(pooled),
    }


def normalized(x: float) -> float:
    return 100.0 * (x - RANDOM_RETURN) / (BEHAVIOR_RETURN - RANDOM_RETURN)


def format_table(rows: list[tuple[float, dict]]) -> str:
    head = (f"{'data%':>6} | {'mean':>9} | {'±seed std':>9} | {'median':>9} | "
            f"{'worst':>9} | {'ok%':>5}")
    lines = [head, "-" * len(head)]
    for frac, st in rows:
        lines.append(
            f"{frac * 100:>5.0f}% | {st['mean']:>9.1f} | {st['seed_std']:>9.1f} | "
            f"{st['median']:>9.1f} | {st['worst']:>9.1f} | {st['success']:>4.0f}%"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--at", type=int, default=None,
                    help="report a single training step instead of a window")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"checkpoints to pool from the end of training "
                         f"(default {DEFAULT_WINDOW}; 1 = final checkpoint only)")
    ap.add_argument("--markdown", action="store_true",
                    help="wrap tables in fenced code blocks for pasting")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="algorithms to leave out of the report")
    ap.add_argument("--fractions", nargs="*", type=float, default=None,
                    help="restrict the report to these data fractions")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict the report to these seeds")
    ap.add_argument("--no-summary", action="store_true",
                    help="omit the retention/normalized summary blocks")
    args = ap.parse_args()

    if args.window < 1:
        ap.error("--window must be at least 1")

    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir} yet.")
        return

    by_algo: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in results:
        if run["algo"] in args.exclude:
            continue
        if args.fractions is not None and run["fraction"] not in args.fractions:
            continue
        if args.seeds is not None and run["seed"] not in args.seeds:
            continue
        by_algo[run["algo"]][run["fraction"]].append(run)
    if not by_algo:
        print("Every algorithm was excluded.")
        return

    expected = max(len(v) for a in by_algo.values() for v in a.values())
    total_runs = len(results)
    if args.at is not None:
        label = f"step {args.at}"
    elif args.window == 1:
        label = "final checkpoint"
    else:
        label = f"last {args.window} checkpoints pooled"
    print(f"hopper-medium data-scarcity sweep - {label}")
    print(f"{total_runs} runs loaded; higher is better "
          f"(random ~{RANDOM_RETURN:.0f}, data-generating policy ~{BEHAVIOR_RETURN:.0f})")
    print(f"ok% = evaluation episodes scoring >= {SUCCESS_RETURN:.0f}, i.e. never fell over")
    print()

    summary: dict[str, dict[float, float]] = {}

    for algo in [a for a in ALGO_ORDER if a in by_algo] + \
                [a for a in by_algo if a not in ALGO_ORDER]:
        rows = []
        for frac in sorted(by_algo[algo]):
            st = stats(by_algo[algo][frac], args.at, args.window)
            if st:
                rows.append((frac, st))
        if not rows:
            continue

        incomplete = [f"{f * 100:.0f}%" for f, st in rows if st["n_seeds"] < expected]
        note = f"  (incomplete: {', '.join(incomplete)})" if incomplete else ""
        print(f"{algo}{note}")
        table = format_table(rows)
        print(f"```\n{table}\n```" if args.markdown else table)
        print()

        summary[algo] = {f: st["mean"] for f, st in rows}

    # The actual research question: who holds up as the data shrinks?
    full = [s for s in summary.values() if 1.0 in s]
    if full and not args.no_summary:
        small = sorted({f for s in summary.values() for f in s if f < 1.0})
        print("Degradation relative to 100% data (mean return, % retained)")
        header = f"{'algo':>8} | " + " | ".join(f"{f * 100:>5.0f}%" for f in small)
        print(header)
        print("-" * len(header))
        for algo, per_frac in summary.items():
            if 1.0 not in per_frac:
                continue
            base = per_frac[1.0]
            cells = []
            for f in small:
                if f in per_frac and base > 0:
                    cells.append(f"{100.0 * per_frac[f] / base:>5.0f}%")
                else:
                    cells.append(f"{'-':>6}")
            print(f"{algo:>8} | " + " | ".join(cells))
        print()
        print("Normalized score at 100% data (100 = matches the data-generating policy)")
        for algo, per_frac in summary.items():
            if 1.0 in per_frac:
                print(f"  {algo:>8}: {normalized(per_frac[1.0]):6.1f}")


if __name__ == "__main__":
    main()
