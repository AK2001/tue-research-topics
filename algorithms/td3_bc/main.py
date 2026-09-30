import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torchrl.data.datasets import load_dataset
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, ObservationNorm
from torchrl.record.loggers import get_logger
from tqdm import trange

from algorithms.td3_bc.core import TD3_BC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = Path(__file__).parent / "results"
BASE_PATH.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")


@torch.no_grad()
def eval_policy(
    policy: TD3_BC,
    eval_env: ParallelEnv | TransformedEnv,
    seed: int,
    seed_offset: int = 100,
):
    eval_env.set_seed(seed + seed_offset)

    policy.actor.to("cpu")
    policy.actor.eval()

    tensor_dict = eval_env.reset()
    eval_episodes = eval_env.num_workers

    episode_returns = torch.zeros(eval_episodes, device="cpu")
    done_mask = torch.zeros(eval_episodes, dtype=torch.bool, device="cpu")

    while not done_mask.all():
        tensor_dict = policy.select_action(tensor_dict)
        tensor_dict = eval_env.step(tensor_dict)

        next_td = tensor_dict["next"]
        rewards = next_td["reward"].squeeze(-1)
        dones = next_td["done"].squeeze(-1) | next_td.get(
            "terminated", torch.zeros_like(done_mask)
        ).squeeze(-1)

        # Accumulate reward only for episodes that haven't finished yet
        episode_returns += rewards * (~done_mask)
        done_mask = done_mask | dones

        tensor_dict = next_td

    policy.actor.to(DEVICE)
    policy.actor.train()

    avg_reward = episode_returns.mean().item()

    # d4rl metrics for hopper dataset
    # source: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py
    random_return = -20.272305
    expert_return = 3234.3

    d4rl_score = 100.0 * (avg_reward - random_return) / (expert_return - random_return)

    return d4rl_score


def train(args):
    logger = get_logger(
        logger_type="csv",
        logger_name="td3_bc_results",
        experiment_name=f"{args.env_id.lower().replace('-', '_')}_s{args.seed}",
        # extra_kwargs={"config": vars(args)},
    )

    replay_buffer = load_dataset(dataset_id=args.dataset_id, batch_size=args.batch_size)

    mean = replay_buffer["observation"].mean(dim=0, keepdim=False)
    std = replay_buffer["observation"].std(dim=0, keepdim=False) + 1e-3

    if args.normalize:
        obs_norm = ObservationNorm(
            in_keys=["observation", ("next", "observation")],
            loc=mean.squeeze(0),
            scale=std.squeeze(0),
            standard_normal=True,
        )
        replay_buffer.append_transform(obs_norm)

    # Create base env
    env = GymEnv(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # set seeds
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    env.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Init policy
    policy = TD3_BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        logger=logger,
        # TD3
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        # TD3 + BC
        alpha=args.alpha,
    )

    # Create eval environments for parallel compute
    # Unfortunately Hopper is CPU-bound, greatly limits efficiency
    make_env = EnvCreator(lambda: GymEnv(args.env_id))
    eval_env = ParallelEnv(args.eval_episodes, make_env)

    transforms = []
    if args.normalize:
        obs_norm = ObservationNorm(
            in_keys=["observation"],
            loc=mean.squeeze(0),
            scale=std.squeeze(0),
            standard_normal=True,
        )
        transforms = [obs_norm]

    transforms.append(DoubleToFloat(in_keys=["observation"]))
    eval_env = TransformedEnv(eval_env, Compose(*transforms))

    # Eval pipeline
    try:
        for t in trange(int(args.max_timesteps), desc="Training TD3+BC"):
            policy.train(replay_buffer.sample())

            if (t + 1) % args.eval_freq == 0:
                d4rl_score = eval_policy(
                    policy=policy,
                    eval_env=eval_env,
                    seed=args.seed,
                )

                logger.log_scalar("eval/d4rl_score", d4rl_score, step=t + 1)
    finally:
        eval_env.close()  # Close parallel env workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Hopper-v5")
    parser.add_argument("--dataset_id", default="d4rl:hopper-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)

    # TD3
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)

    # TD3 + BC
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--normalize", default=True, type=bool)

    args = parser.parse_args()

    train(args)
