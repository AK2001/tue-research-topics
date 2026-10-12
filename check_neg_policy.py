"""
Re-evaluates the saved (frozen) negative policies over MORE episodes and with NEW seeds,
to check whether they are really dangerous or were selected by luck.

Run from the CORL folder:
    python check_neg_policy.py
    python check_neg_policy.py --episodes 30
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms", "offline"))
import td3_bc_neg_v2 as v2  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="hopper-medium-replay-v2")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--neg_dir", default="models/neg_pretrained")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nNegative policies on {args.env}: {args.episodes} new evaluation episodes each\n")
print(f"{'seed':6s}{'selected score':>16s}{'new score':>12s}{'avg length':>12s}{'fell':>8s}")
for seed in [int(s) for s in args.seeds.split(",")]:
    config = v2.TrainConfig(env=args.env, seed=seed, device=device, neg_dir=args.neg_dir)
    env, _, state_dim, action_dim, max_action, state_mean, state_std = v2.setup(config)
    saved = torch.load(v2.neg_file_path(config), map_location=device)

    neg_actor = v2.Actor(state_dim, action_dim, max_action).to(device)
    neg_actor.load_state_dict(saved["neg_actor"])

    # new evaluation seed (different from the one used to select the snapshot)
    scores, lengths = v2.eval_actor(env, neg_actor, device=device, n_episodes=args.episodes,
                                    seed=seed + 1000, return_lengths=True)
    norm = env.get_normalized_score(scores.mean()) * 100.0
    fell = 100.0 * np.mean(lengths < 1000)
    print(f"{seed:<6d}{saved['score']:>16.2f}{norm:>12.2f}{lengths.mean():>12.1f}{fell:>7.0f}%")
print()
