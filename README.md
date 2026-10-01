# IQL with a negative policy

Carries the negative-policy penalty from `feature/td3_bc_neg` over to **IQL**,
on `mujoco/walker2d/medium-v0`. Answers the supervisor's note on the first
draft: *"consider other offline algorithms as baseline."*

**Result: the improvement does not survive more seeds.** At 5 seeds the
negative policy looks like a clear win; at 15 it is a coin flip.

## Files

| | |
|---|---|
| `IQL_negative_policy_walker2d.ipynb` | self-contained Colab reproduction — start here |
| `iql_neg.py` | IQL + negative policy, `--neg_mode none/awr/dist` |
| `iql.py` | the unmodified CORL / IQL-PyTorch reference, for comparison |
| `DATASET.md` | which dataset, why, and how to get it |

## Why IQL needs its own formulation

TD3+BC maximises Q *through the actor*, so a negative policy slots in as a
distance penalty in action space. IQL never queries the critic on its own
actions — its policy step is advantage-weighted regression, a weighted
imitation of the dataset. So there are two ways to add the penalty:

| `--neg_mode` | |
|---|---|
| `none` | ordinary IQL — the baseline everything is compared against |
| `dist` | the direct port: train a danger actor, push the main actor away from it |
| `awr` | the IQL-native form: fold the danger advantage into the imitation weight |

```
stock IQL:  weight = exp( beta * A )
awr:        weight = exp( beta * A  -  beta_neg * A_neg )
```

`A_neg` is not simply `-A`. IQL's value step takes an upper expectile
(`tau=0.7`), so the danger stack ends up using a lower one. The two stacks
bootstrap through different continuations: `Q` assumes you act well afterwards,
`-Q_neg` assumes you act badly. That gap is a risk signal the ordinary
advantage does not carry, and TD3+BC cannot express it at all.

## Results

Normalised score, 0 = random policy, 100 = expert data. 25,000 steps,
100 evaluation episodes per seed.

**5 seeds**

| Configuration | Mean | Median | Worst | Best |
|---|---|---|---|---|
| Baseline (IQL) | 86.7 | 88.8 | 34.2 | 90.8 |
| **Negative [awr], β=0.5** | **88.8** | **89.0** | **65.2** | **90.7** |
| Negative [dist], β=0.5 | 88.8 | 89.1 | 74.7 | 90.8 |

**15 seeds**

| Configuration | Mean | Median | Worst | Best |
|---|---|---|---|---|
| Baseline (IQL), τ=0.7 | 87.6 | 88.7 | 53.3 | 90.8 |
| **Pessimistic IQL, τ=0.5** | **87.6** | **88.3** | **62.5** | **89.9** |
| Pessimistic IQL, τ=0.3 | 86.3 | 87.6 | 52.8 | 89.3 |
| Negative [awr], β=0.5 | 87.6 | 88.6 | 57.1 | 90.5 |

The gap on `Worst` collapses from 31 points to 3.8. A paired bootstrap over
seeds puts the chance the negative policy beats the baseline at **0.53** — at
5 seeds it was 0.99.

This is a seed lottery, not a bug: seeds 0–4 reproduce exactly between the two
sweeps. The baseline's per-seed worst-case score across 15 seeds is

```
15.0, 59.1, 72.1, 73.5, 76.2, 81.0, 81.4, 82.1, 84.7, 85.9, 87.1, 87.1, 87.4, 87.6, 87.9
^^^^ this one landed inside the first five
```

so the first 5 averaged 67.7 while all 15 average 76.5.

Note that the cheap alternative — plain IQL at a lower expectile, one changed
number and no second network — is the best row at 15 seeds. And `τ=0.3` is
*worse* than the baseline, so "more pessimism is better" is false.

## Why this benchmark cannot settle it

The baseline falls in only **2.1%** of episodes (32 in 1500), and **5 of 15
seeds never fell once**. The quantity we are trying to reduce is already zero
for a third of the runs, and the seed-to-seed spread is several times larger
than any effect being chased — detecting it would need roughly 140 seeds. The
next experiment needs an explicit hazard, so the measurement becomes a count
per episode rather than a rare event.

## Running it

```bash
python iql_neg.py --dataset mujoco/walker2d/medium-v0 --neg_mode awr --beta_neg 0.5 \
    --steps 25000 --n_episodes 100 --seed 0
```

`--neg_mode none` for the baseline. The dataset id may also be a D4RL one
(`walker2d-medium-v2`) on a machine that has D4RL installed.

## Fixes needed in the draft, independent of these results

1. **Eq. 3 has the wrong sign.** Under `argmax`, `-β(π - π_neg)²` *minimises*
   the distance, pulling the policy *towards* danger. The code and the
   pseudocode are correct; the equation should read `+β(π(s) - π_neg(s))²`.
2. **The penalty is unbounded**, the likely cause of the β=1.0 collapse in
   Figure 2. `iql_neg.py` uses a bounded hinge, `max(0, margin - d)`.
3. **"Worst" in Figure 2 is a minimum over 20 episodes and 3 seeds.** Say which
   definition is meant, and report CVaR alongside it.
