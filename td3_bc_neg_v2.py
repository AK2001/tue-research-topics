# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
#
# VERSION 2 of the negative-policy method (td3_bc_neg_v2.py).
# Same objective as td3_bc_neg.py (paper Eq. 4-6, "mse" mode), with two implementation fixes:
#   1. The danger critic uses TWO networks and the minimum of both (clipped double-Q, as in TD3),
#      to prevent the overestimation seen in version 1.
#   2. The negative agent is trained FIRST (--mode pretrain_neg), the most dangerous snapshot is
#      saved, and the main TD3+BC training then loads it and keeps it FROZEN (--mode train).
# With beta = 0 the training mode is exactly TD3+BC.
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Mode
    mode: str = "train"  # V2: "pretrain_neg" = train + save the negative agent; "train" = TD3+BC (+ frozen negative policy)
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps (in pretrain_neg mode: pre-training steps)
    checkpoints_path: Optional[str] = None  # Save path
    save_final_dir: Optional[str] = None  # if set, save only the final model of the run here
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Negative policy (V2: pre-trained with two danger critics, then frozen)
    beta: float = 0.0  # avoidance weight; 0 = plain TD3+BC. Must be < 1
    neg_dir: str = "models/neg_pretrained"  # where pre-trained negative agents are saved / loaded
    neg_eval_episodes: int = 3  # pretrain_neg mode: episodes per evaluation of the negative policy
    neg_select_after: float = 0.5  # pretrain_neg mode: only snapshots after this fraction of training are candidates
    # Wandb logging
    project: str = "CORL"
    group: str = "TD3_BC-D4RL"
    name: str = "TD3_BC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:  # NEG: back to the original CORL version (only stores data)
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int,
    return_lengths: bool = False,
):
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    episode_lengths = []  # NEG: number of steps per episode (< max steps = the robot fell)
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    actor.train()
    if return_lengths:
        return np.asarray(episode_rewards), np.asarray(episode_lengths)
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class NegPretrainer:
    """Trains the negative ("danger-seeking") agent on its own: TD3+BC on the inverted reward -r,
    with TWO danger critics and the minimum of both in the target (clipped double-Q)."""

    def __init__(
        self,
        max_action: float,
        neg_actor: nn.Module,
        neg_actor_optimizer: torch.optim.Optimizer,
        danger_critic_1: nn.Module,
        danger_critic_1_optimizer: torch.optim.Optimizer,
        danger_critic_2: nn.Module,
        danger_critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
    ):
        self.neg_actor = neg_actor
        self.neg_actor_target = copy.deepcopy(neg_actor)
        self.neg_actor_optimizer = neg_actor_optimizer
        self.danger_critic_1 = danger_critic_1
        self.danger_critic_1_target = copy.deepcopy(danger_critic_1)
        self.danger_critic_1_optimizer = danger_critic_1_optimizer
        self.danger_critic_2 = danger_critic_2
        self.danger_critic_2_target = copy.deepcopy(danger_critic_2)
        self.danger_critic_2_optimizer = danger_critic_2_optimizer
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1 - done
        danger_reward = -reward  # danger = negated reward (paper Eq. 4)

        # ===== danger critics: two networks, minimum of both in the target =====
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.neg_actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_d1 = self.danger_critic_1_target(next_state, next_action)
            target_d2 = self.danger_critic_2_target(next_state, next_action)
            target_d = danger_reward + not_done * self.discount * torch.min(target_d1, target_d2)

        current_d1 = self.danger_critic_1(state, action)
        current_d2 = self.danger_critic_2(state, action)
        danger_critic_loss = F.mse_loss(current_d1, target_d) + F.mse_loss(current_d2, target_d)
        log_dict["danger_critic_loss"] = danger_critic_loss.item()
        self.danger_critic_1_optimizer.zero_grad()
        self.danger_critic_2_optimizer.zero_grad()
        danger_critic_loss.backward()
        self.danger_critic_1_optimizer.step()
        self.danger_critic_2_optimizer.step()

        # ===== delayed negative actor update: TD3+BC on the danger critic (paper Eq. 5) =====
        if self.total_it % self.policy_freq == 0:
            neg_pi = self.neg_actor(state)
            neg_q = self.danger_critic_1(state, neg_pi)
            neg_lmbda = self.alpha / neg_q.abs().mean().detach()
            neg_actor_loss = -neg_lmbda * neg_q.mean() + F.mse_loss(neg_pi, action)
            log_dict["neg_actor_loss"] = neg_actor_loss.item()
            log_dict["neg_q_mean"] = neg_q.mean().item()
            log_dict["neg_lambda"] = neg_lmbda.item()
            log_dict["neg_to_data"] = F.mse_loss(neg_pi.detach(), action).item()
            self.neg_actor_optimizer.zero_grad()
            neg_actor_loss.backward()
            self.neg_actor_optimizer.step()

            soft_update(self.danger_critic_1_target, self.danger_critic_1, self.tau)
            soft_update(self.danger_critic_2_target, self.danger_critic_2, self.tau)
            soft_update(self.neg_actor_target, self.neg_actor, self.tau)

        return log_dict

    def snapshot(self) -> Dict[str, Any]:
        return {
            "neg_actor": copy.deepcopy(self.neg_actor.state_dict()),
            "danger_critic_1": copy.deepcopy(self.danger_critic_1.state_dict()),
            "danger_critic_2": copy.deepcopy(self.danger_critic_2.state_dict()),
        }


class TD3_BC:
    """TD3+BC, optionally with the paper's avoidance term against a FROZEN negative policy."""

    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
        frozen_neg_actor: Optional[nn.Module] = None,  # V2: pre-trained negative policy, never updated here
        beta: float = 0.0,
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

        # loss = mse(pi, a) - beta * mse(pi, a_neg); beta >= 1 removes the BC constraint
        assert 0.0 <= beta < 1.0, "beta must be in [0, 1)"
        self.beta = beta
        self.use_neg = beta > 0 and frozen_neg_actor is not None
        if self.use_neg:
            self.neg_actor = frozen_neg_actor
            self.neg_actor.eval()
            for p in self.neg_actor.parameters():  # frozen: never trained in this mode
                p.requires_grad_(False)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        # ===== Normal TD3 critic update (unchanged) =====
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            bc_mse = F.mse_loss(pi, action)
            actor_loss = -lmbda * q.mean() + bc_mse
            log_dict["q_mean"] = q.mean().item()
            log_dict["bc_mse"] = bc_mse.item()

            # ===== avoidance term against the frozen negative policy (paper Eq. 6) =====
            if self.use_neg:
                with torch.no_grad():
                    neg_action = self.neg_actor(state)
                avoidance = -F.mse_loss(pi, neg_action)
                actor_loss = actor_loss + self.beta * avoidance
                log_dict["avoidance"] = avoidance.item()
                with torch.no_grad():
                    log_dict["dist_to_neg"] = F.mse_loss(pi, neg_action).item()
                    log_dict["neg_to_data"] = F.mse_loss(neg_action, action).item()

            log_dict["actor_loss"] = actor_loss.item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)
        self.total_it = state_dict["total_it"]


# ---------------------------------------------------------------------------------------------
# helpers shared by both modes
# ---------------------------------------------------------------------------------------------
def neg_file_path(config: TrainConfig) -> str:
    return os.path.join(config.neg_dir, f"{config.env}__seed{config.seed}__neg.pt")


def print_config(config: TrainConfig):
    print("Config:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    print("---------------------------------------")


def stats_line(stat_sums: Dict[str, float], stat_counts: Dict[str, int]) -> str:
    return ", ".join(f"{k}={stat_sums[k] / stat_counts[k]:.4f}" for k in sorted(stat_sums))


def setup(config: TrainConfig):
    """Loads the data and environment exactly the same way in both modes."""
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)
    if config.normalize_reward:
        modify_reward(dataset, config.env)
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)
    max_action = float(env.action_space.high[0])

    set_seed(config.seed, env)
    return env, replay_buffer, state_dim, action_dim, max_action, state_mean, state_std


def as_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------------------------
# mode 1: pre-train the negative agent and save its most dangerous snapshot
# ---------------------------------------------------------------------------------------------
def pretrain_neg(config: TrainConfig):
    env, replay_buffer, state_dim, action_dim, max_action, state_mean, state_std = setup(config)

    neg_actor = Actor(state_dim, action_dim, max_action).to(config.device)
    neg_actor_optimizer = torch.optim.Adam(neg_actor.parameters(), lr=3e-4)
    danger_critic_1 = Critic(state_dim, action_dim).to(config.device)
    danger_critic_1_optimizer = torch.optim.Adam(danger_critic_1.parameters(), lr=3e-4)
    danger_critic_2 = Critic(state_dim, action_dim).to(config.device)
    danger_critic_2_optimizer = torch.optim.Adam(danger_critic_2.parameters(), lr=3e-4)

    trainer = NegPretrainer(
        max_action=max_action,
        neg_actor=neg_actor,
        neg_actor_optimizer=neg_actor_optimizer,
        danger_critic_1=danger_critic_1,
        danger_critic_1_optimizer=danger_critic_1_optimizer,
        danger_critic_2=danger_critic_2,
        danger_critic_2_optimizer=danger_critic_2_optimizer,
        discount=config.discount,
        tau=config.tau,
        policy_noise=config.policy_noise * max_action,
        noise_clip=config.noise_clip * max_action,
        policy_freq=config.policy_freq,
        alpha=config.alpha,
    )

    print("---------------------------------------")
    print(f"PRE-TRAINING the negative agent, Env: {config.env}, Seed: {config.seed}, "
          f"steps: {config.max_timesteps}")
    print("---------------------------------------")
    print_config(config)

    select_from = int(config.neg_select_after * config.max_timesteps)
    best_score, best_step, best_snapshot = None, None, None
    neg_norm = float("nan")
    stat_sums, stat_counts = {}, {}

    for t in range(int(config.max_timesteps)):
        batch = [b.to(config.device) for b in replay_buffer.sample(config.batch_size)]
        log_dict = trainer.train(batch)
        for k, v in log_dict.items():
            stat_sums[k] = stat_sums.get(k, 0.0) + v
            stat_counts[k] = stat_counts.get(k, 0) + 1

        if (t + 1) % config.eval_freq == 0:
            scores, lengths = eval_actor(
                env, trainer.neg_actor, device=config.device,
                n_episodes=config.neg_eval_episodes, seed=config.seed, return_lengths=True,
            )
            neg_norm = env.get_normalized_score(scores.mean()) * 100.0
            print(f"Pretrain step {t + 1}: Negative policy normalized score: {neg_norm:.3f}")
            print("Episode lengths: " + ", ".join(str(int(x)) for x in lengths))
            print(f"Train stats: {stats_line(stat_sums, stat_counts)}")
            stat_sums, stat_counts = {}, {}

            # keep the most dangerous (lowest-scoring) snapshot from the second part of training
            if t + 1 >= select_from and (best_score is None or neg_norm < best_score):
                best_score, best_step, best_snapshot = neg_norm, t + 1, trainer.snapshot()
                print(f"  -> new most dangerous snapshot (score {neg_norm:.3f})")

    if best_snapshot is None:  # e.g. very short test runs
        best_score, best_step, best_snapshot = neg_norm, int(config.max_timesteps), trainer.snapshot()

    os.makedirs(config.neg_dir, exist_ok=True)
    path = neg_file_path(config)
    torch.save(
        {
            **best_snapshot,
            "step": best_step,
            "score": float(best_score),
            "state_mean": as_tensor(state_mean),
            "state_std": as_tensor(state_std),
            "config": asdict(config),
        },
        path,
    )
    print("---------------------------------------")
    print(f"Saved negative agent: {path}")
    print(f"  snapshot from step {best_step}, negative policy score {best_score:.3f}")
    print("---------------------------------------")


# ---------------------------------------------------------------------------------------------
# mode 2: TD3+BC, with the frozen pre-trained negative policy if beta > 0
# ---------------------------------------------------------------------------------------------
def train_main(config: TrainConfig):
    env, replay_buffer, state_dim, action_dim, max_action, state_mean, state_std = setup(config)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    # load the frozen negative policy (only when it is used)
    frozen_neg_actor = None
    if config.beta > 0:
        path = neg_file_path(config)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No pre-trained negative agent at {path}. Run --mode pretrain_neg for this env/seed first."
            )
        saved = torch.load(path, map_location=config.device)
        if not (np.allclose(saved["state_mean"].cpu().numpy(), np.asarray(state_mean, dtype=np.float32), atol=1e-4)
                and np.allclose(saved["state_std"].cpu().numpy(), np.asarray(state_std, dtype=np.float32), atol=1e-4)):
            raise ValueError("State normalization of the saved negative agent does not match this run.")
        frozen_neg_actor = Actor(state_dim, action_dim, max_action).to(config.device)
        frozen_neg_actor.load_state_dict(saved["neg_actor"])
        print(f"Loaded frozen negative policy: {path} "
              f"(snapshot step {saved['step']}, negative policy score {saved['score']:.3f})")

    trainer = TD3_BC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        discount=config.discount,
        tau=config.tau,
        device=config.device,
        policy_noise=config.policy_noise * max_action,
        noise_clip=config.noise_clip * max_action,
        policy_freq=config.policy_freq,
        alpha=config.alpha,
        frozen_neg_actor=frozen_neg_actor,
        beta=config.beta,
    )

    print("---------------------------------------")
    print(f"Training TD3 + BC (v2), Env: {config.env}, Seed: {config.seed}, "
          f"beta: {config.beta}, alpha: {config.alpha}")
    print("---------------------------------------")

    if config.load_model != "":
        trainer.load_state_dict(torch.load(Path(config.load_model)))
        actor = trainer.actor

    print_config(config)
    wandb_init(asdict(config))

    evaluations = []
    stat_sums, stat_counts = {}, {}
    for t in range(int(config.max_timesteps)):
        batch = [b.to(config.device) for b in replay_buffer.sample(config.batch_size)]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        for k, v in log_dict.items():
            stat_sums[k] = stat_sums.get(k, 0.0) + v
            stat_counts[k] = stat_counts.get(k, 0) + 1

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, eval_lengths = eval_actor(
                env, actor, device=config.device, n_episodes=config.n_episodes,
                seed=config.seed, return_lengths=True,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print(f"Train stats: {stats_line(stat_sums, stat_counts)}")
            stat_sums, stat_counts = {}, {}
            ep_norm = [env.get_normalized_score(r) * 100.0 for r in eval_scores]
            print("Episode scores: " + ", ".join(f"{x:.2f}" for x in ep_norm))
            print("Episode lengths: " + ", ".join(str(int(x)) for x in eval_lengths))
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(trainer.state_dict(),
                           os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"))
            wandb.log({"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it)

    if config.save_final_dir is not None:
        os.makedirs(config.save_final_dir, exist_ok=True)
        final_name = f"{config.env}__seed{config.seed}__beta{config.beta}__alpha{config.alpha}__v2.pt"
        torch.save(trainer.state_dict(), os.path.join(config.save_final_dir, final_name))
        print(f"Final model saved: {os.path.join(config.save_final_dir, final_name)}")


@pyrallis.wrap()
def main(config: TrainConfig):
    assert config.mode in ("train", "pretrain_neg"), "mode must be 'train' or 'pretrain_neg'"
    if config.mode == "pretrain_neg":
        pretrain_neg(config)
    else:
        train_main(config)


if __name__ == "__main__":
    main()
