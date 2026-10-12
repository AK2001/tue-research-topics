"""
Debug check for the negative-policy method. NO training: it only loads the saved models and the
dataset, so it runs on the free int3 node in a few minutes.

Our actor loss with the avoidance term is equivalent to imitating a SHIFTED target action
    a* = a + beta/(1-beta) * (a - a_neg)          (beta = 0.5  ->  a* = 2a - a_neg)
where a is the dataset action and a_neg the frozen negative policy's action in the same state.
The actor can only output actions in [-1, 1] (tanh), so any part of a* outside [-1, 1] cannot
be reached and pushes the actor to the edge of its action range.

Part A (negative policy, per seed):
  neg_to_data  : mean squared distance between a_neg and the dataset action
  shift        : mean absolute size of the push |a - a_neg| * beta/(1-beta), per action component
  outside      : % of target components a* outside [-1, 1]  (unreachable targets)
Part B (final main policies, beta = 0 vs beta, per seed):
  to_data      : mean squared distance between the policy's action and the dataset action
  saturated    : % of action components at the edge (|action| > 0.99)
  to_neg       : mean squared distance to the negative policy

Run from the CORL folder, e.g.
    python check_target_shift.py --env hopper-medium-replay-v2   --final_dir models/final
    python check_target_shift.py --env walker2d-medium-replay-v2 --final_dir models/final_walker2d
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms", "offline"))
import td3_bc_neg_v2 as v2  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="walker2d-medium-replay-v2")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--neg_dir", default="models/neg_pretrained")
parser.add_argument("--final_dir", default="models/final_walker2d")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
k = args.beta / (1.0 - args.beta)


def act(actor, states, bs=50_000):
    with torch.no_grad():
        return torch.cat([actor(states[i:i + bs]) for i in range(0, len(states), bs)])


def load_main_actor(path, state_dim, action_dim, max_action):
    if not os.path.exists(path):
        return None
    actor = v2.Actor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(path, map_location=device)["actor"])
    actor.eval()
    return actor


rows_a, rows_b = [], []
for seed in [int(s) for s in args.seeds.split(",")]:
    config = v2.TrainConfig(env=args.env, seed=seed, device=device, neg_dir=args.neg_dir)
    env, buf, state_dim, action_dim, max_action, _, _ = v2.setup(config)
    n = buf._size
    states, actions = buf._states[:n], buf._actions[:n]

    saved = torch.load(v2.neg_file_path(config), map_location=device)
    neg = v2.Actor(state_dim, action_dim, max_action).to(device)
    neg.load_state_dict(saved["neg_actor"])
    neg.eval()
    a_neg = act(neg, states)

    target = actions + k * (actions - a_neg)
    rows_a.append((seed,
                   ((a_neg - actions) ** 2).mean().item(),
                   (k * (actions - a_neg)).abs().mean().item(),
                   100.0 * (target.abs() > max_action).float().mean().item()))

    for beta in (0.0, args.beta):
        name = f"{args.env}__seed{seed}__beta{beta}__alpha{config.alpha}__v2.pt"
        actor = load_main_actor(os.path.join(args.final_dir, name), state_dim, action_dim, max_action)
        if actor is None:
            print(f"  (missing {name})")
            continue
        a_pi = act(actor, states)
        rows_b.append((seed, beta,
                       ((a_pi - actions) ** 2).mean().item(),
                       100.0 * (a_pi.abs() > 0.99 * max_action).float().mean().item(),
                       ((a_pi - a_neg) ** 2).mean().item()))

print(f"\n=== {args.env}, beta = {args.beta} (whole dataset, {n} states) ===")
print("\nA. Negative policy and the shifted target a*")
print(f"{'seed':6s}{'neg_to_data':>13s}{'shift':>10s}{'outside':>10s}")
for s, d, sh, out in rows_a:
    print(f"{s:<6d}{d:>13.3f}{sh:>10.3f}{out:>9.1f}%")
print("\nB. Final main policies")
print(f"{'seed':6s}{'beta':>6s}{'to_data':>10s}{'saturated':>11s}{'to_neg':>9s}")
for s, b, d, sat, tn in rows_b:
    print(f"{s:<6d}{b:>6.2f}{d:>10.3f}{sat:>10.1f}%{tn:>9.3f}")
print()