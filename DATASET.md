# Dataset

Every run uses one dataset:

```
mujoco/walker2d/medium-v0        (Minari)
```

## It is not committed here

It is a single **201 MB** HDF5 file and GitHub rejects anything over 100 MB. It
does not need to be committed: Minari downloads it on demand, and
`dataset/walker2d_medium_v0_metadata.json` pins exactly which version we used.

```python
import minari
ds = minari.load_dataset("mujoco/walker2d/medium-v0", download=True)
```

The notebook does this for you.

## What it is

| | |
|---|---|
| environment | `Walker2d-v5` (Gymnasium + MuJoCo 3.2.3), 1000-step limit |
| episodes | 1,044 |
| transitions | 999,613 |
| observation / action | 17 / 6 |
| behaviour policy | SB3 `SAC`, trained 6e6 steps |
| minari version | 0.5.2 |
| return mean / median | 5701.6 / 5960.6 |
| return p1 / min | 1.0 / 0.2 |
| episodes ending in a fall | 58 (5.6%) |

## Why `medium` and not `simple` or `expert`

A worst-case study needs a dataset where things actually go wrong. Only
`medium` has a real catastrophic tail:

| dataset | median | p5 | p1 | min | falls |
|---|---|---|---|---|---|
| simple | 4149 | 4006 | 492 | 193 | 2.8% |
| **medium** | **5960** | **4570** | **1.0** | **0.2** | **5.6%** |
| expert | 6870 | 6756 | 6328 | 2664 | 0.4% |

On `expert` there is nothing to improve, and no unsafe behaviour for a danger
policy to learn from.

## Why Minari and not D4RL

D4RL needs `mujoco-py`, the old `gym` and Python <= 3.9, which does not install
on current runtimes or on Apple silicon. Minari is the maintained successor and
hosts the same locomotion data.

`iql_neg.py` reads either, with identical output: a dataset id containing a
slash (`mujoco/walker2d/medium-v0`) goes to Minari, one without
(`walker2d-medium-v2`) goes to D4RL. The same script therefore runs both
locally and on a cluster that does have D4RL.

## Normalised score

Raw returns are `steps per episode x reward per step`, so they are not
comparable across environments — Pendulum gives hundreds, walker2d thousands.
Results are reported as

```
score = 100 * (return - 2.1) / (6847.8 - 2.1)
```

* `2.1` — a uniformly random policy on `Walker2d-v5`, mean over 30 episodes
* `6847.8` — the mean return of `mujoco/walker2d/expert-v0`

Fixed constants, not measured at run time, so the notebook and the report give
identical numbers. **A low score means the agent fell early**, not that it
walked badly.
