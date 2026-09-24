"""
Self-contained TD3+BC with an optional NEGATIVE POLICY (Option B).

No D4RL, no MuJoCo. Runs on a laptop CPU.
Only needs:  pip install torch gymnasium

What it does:
  1. Collects an offline dataset from the Pendulum environment (once).
  2. Trains TD3+BC on that fixed dataset (offline RL).
  3. Optionally adds a "negative policy" that learns to SEEK danger
     (danger = negated reward), and penalizes the main policy for
     choosing actions similar to what the danger-seeker would choose.
  4. Evaluates and reports mean / median / worst-case scores.

Run the baseline:            python td3bc_negpolicy.py
Run with negative policy:    python td3bc_negpolicy.py --neg
Tune the penalty strength:   python td3bc_negpolicy.py --neg --beta 0.1
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ----------------------------------------------------------------------
# Simple replay buffer (holds the fixed offline dataset)
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, states, actions, rewards, next_states, dones, device):
        self.states = torch.tensor(states, dtype=torch.float32, device=device)
        self.actions = torch.tensor(actions, dtype=torch.float32, device=device)
        self.rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        self.next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        self.dones = torch.tensor(dones, dtype=torch.float32, device=device)
        self.size = self.states.shape[0]

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


# ----------------------------------------------------------------------
# Step 1a: collect a TOY offline dataset (Pendulum) with a scripted,
# mixed-quality controller. Pure-random data makes TD3+BC diverge, so we
# use a rough energy-based swing-up controller plus exploration noise.
# This produces "medium"-style data the algorithm can actually learn from.
# ----------------------------------------------------------------------
def scripted_pendulum_action(state, max_action, noise_scale, rng):
    """A rough swing-up/balance controller for Pendulum-v1.
    state = [cos(theta), sin(theta), theta_dot]. Not optimal, just decent."""
    cos_th, sin_th, theta_dot = state
    # energy-shaping-ish: push in the direction that adds energy when down,
    # and damp when near the top. Deliberately imperfect -> mixed quality.
    if cos_th < 0.3:  # pendulum is low: pump energy in direction of motion
        action = np.sign(theta_dot + 1e-6) * max_action
    else:             # near the top: damp to balance
        action = -2.0 * theta_dot - 4.0 * sin_th
    action = np.clip(action, -max_action, max_action)
    action = action + rng.normal(0.0, noise_scale * max_action)  # exploration noise
    return np.clip(np.array([action], dtype=np.float32), -max_action, max_action)


def collect_dataset(env_name, n_transitions, seed):
    env = gym.make(env_name)
    env.reset(seed=seed)
    max_action = float(env.action_space.high[0])
    rng = np.random.default_rng(seed)

    S, A, R, NS, D = [], [], [], [], []
    state, _ = env.reset()
    for _ in range(n_transitions):
        # vary the noise so the dataset is a mix of good and sloppy behavior
        noise_scale = rng.choice([0.1, 0.3, 0.6])
        action = scripted_pendulum_action(state, max_action, noise_scale, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        S.append(state); A.append(action); R.append([reward])
        NS.append(next_state); D.append([float(terminated)])
        state = next_state
        if done:
            state, _ = env.reset()
    env.close()
    return (np.array(S, dtype=np.float32), np.array(A, dtype=np.float32),
            np.array(R, dtype=np.float32), np.array(NS, dtype=np.float32),
            np.array(D, dtype=np.float32), max_action)


# ----------------------------------------------------------------------
# Step 1b: load a REAL D4RL dataset (for the cluster). Only imported when
# actually used, so the toy path needs no d4rl/mujoco installed.
# ----------------------------------------------------------------------
def load_d4rl_dataset(env_name):
    import gym as _gym       # old gym, required by d4rl
    import d4rl              # noqa: F401  (registers the datasets)
    env = _gym.make(env_name)
    ds = d4rl.qlearning_dataset(env)
    max_action = float(env.action_space.high[0])
    S = ds["observations"].astype(np.float32)
    A = ds["actions"].astype(np.float32)
    R = ds["rewards"].astype(np.float32).reshape(-1, 1)
    NS = ds["next_observations"].astype(np.float32)
    D = ds["terminals"].astype(np.float32).reshape(-1, 1)
    return S, A, R, NS, D, max_action


# ----------------------------------------------------------------------
# Step 1c: load an EXISTING d3rlpy dataset (e.g. Pendulum). This is an
# externally-provided dataset (more trustworthy than our home-made one),
# and needs only d3rlpy installed (no mujoco/d4rl). Handles both the
# newer d3rlpy API (episodes with .observations/.actions/...) and older
# MDPDataset-style attributes.
# ----------------------------------------------------------------------
def load_d3rlpy_dataset(which="pendulum"):
    import d3rlpy
    if which == "pendulum":
        dataset, env = d3rlpy.datasets.get_pendulum()
    elif which == "cartpole":
        dataset, env = d3rlpy.datasets.get_cartpole()
    else:
        raise ValueError(f"unknown d3rlpy dataset: {which}")

    S, A, R, NS, D = [], [], [], [], []
    for ep in dataset.episodes:
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        rew = np.asarray(ep.rewards, dtype=np.float32).reshape(-1)
        # build (s, a, r, s') transitions within the episode
        n = len(obs)
        for t in range(n - 1):
            S.append(obs[t]); A.append(act[t]); R.append([rew[t]])
            NS.append(obs[t + 1]); D.append([0.0])
        # last step of the episode is terminal
        S.append(obs[-1]); A.append(act[-1]); R.append([rew[-1]])
        NS.append(obs[-1]); D.append([1.0])

    S = np.asarray(S, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    R = np.asarray(R, dtype=np.float32)
    NS = np.asarray(NS, dtype=np.float32)
    D = np.asarray(D, dtype=np.float32)
    max_action = float(np.abs(A).max())
    return S, A, R, NS, D, max_action


# ----------------------------------------------------------------------
# The algorithm: TD3+BC, with optional negative policy
# ----------------------------------------------------------------------
class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action, device,
                 use_neg=False, beta=0.1, alpha=2.5,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = device
        self.max_action = max_action
        self.use_neg = use_neg
        self.beta = beta
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.total_it = 0

        # main actor + two critics (standard TD3)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        # negative policy + its danger critic (only if enabled)
        if use_neg:
            self.neg_actor = Actor(state_dim, action_dim, max_action).to(device)
            self.neg_actor_target = copy.deepcopy(self.neg_actor)
            self.neg_actor_opt = torch.optim.Adam(self.neg_actor.parameters(), lr=3e-4)

            self.danger_critic = Critic(state_dim, action_dim).to(device)
            self.danger_critic_target = copy.deepcopy(self.danger_critic)
            self.danger_critic_opt = torch.optim.Adam(self.danger_critic.parameters(), lr=3e-4)

    def train_step(self, batch):
        self.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1.0 - done
        log = {}

        # ----- main critics update (standard TD3) -----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            tq1 = self.critic_1_target(next_state, next_action)
            tq2 = self.critic_2_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(tq1, tq2)

        cq1 = self.critic_1(state, action)
        cq2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)
        self.critic_1_opt.zero_grad(); self.critic_2_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_1_opt.step(); self.critic_2_opt.step()
        log["critic_loss"] = critic_loss.item()

        # ----- danger critic update (learns value of DANGER = -reward) -----
        if self.use_neg:
            danger_reward = -reward
            with torch.no_grad():
                neg_noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip)
                neg_next_action = (self.neg_actor_target(next_state) + neg_noise).clamp(
                    -self.max_action, self.max_action)
                dt = self.danger_critic_target(next_state, neg_next_action)
                danger_target = danger_reward + not_done * self.discount * dt
            cd = self.danger_critic(state, action)
            danger_loss = F.mse_loss(cd, danger_target)
            self.danger_critic_opt.zero_grad()
            danger_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.danger_critic.parameters(), 1.0)
            self.danger_critic_opt.step()
            log["danger_critic_loss"] = danger_loss.item()

        # ----- delayed actor updates -----
        if self.total_it % self.policy_freq == 0:

            # negative actor: learn to SEEK danger
            if self.use_neg:
                neg_pi = self.neg_actor(state)
                neg_q = self.danger_critic(state, neg_pi)
                neg_lmbda = self.alpha / neg_q.abs().mean().detach()
                neg_actor_loss = -neg_lmbda * neg_q.mean() + F.mse_loss(neg_pi, action)
                self.neg_actor_opt.zero_grad()
                neg_actor_loss.backward()
                self.neg_actor_opt.step()
                log["neg_actor_loss"] = neg_actor_loss.item()

            # main actor: TD3+BC, plus (optionally) avoid the negative policy
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)

            if self.use_neg:
                with torch.no_grad():
                    neg_action = self.neg_actor(state)
                # penalize being CLOSE to the danger-seeker's action
                avoidance = -F.mse_loss(pi, neg_action)
                actor_loss = actor_loss + self.beta * avoidance
                log["avoidance"] = avoidance.item()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            log["actor_loss"] = actor_loss.item()

            # target updates
            self._soft_update(self.critic_1_target, self.critic_1)
            self._soft_update(self.critic_2_target, self.critic_2)
            self._soft_update(self.actor_target, self.actor)
            if self.use_neg:
                self._soft_update(self.danger_critic_target, self.danger_critic)
                self._soft_update(self.neg_actor_target, self.neg_actor)

        return log

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    @torch.no_grad()
    def act(self, state):
        s = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        return self.actor(s).cpu().numpy().flatten()


# ----------------------------------------------------------------------
# Evaluation: play the real game many times, return all scores
# ----------------------------------------------------------------------
def evaluate(agent, env_name, n_episodes, seed):
    env = gym.make(env_name)
    scores = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + 1000 + i)
        done = False
        total = 0.0
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return np.array(scores)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neg", action="store_true", help="enable the negative policy")
    p.add_argument("--beta", type=float, default=0.1, help="avoidance penalty strength")
    p.add_argument("--steps", type=int, default=30000, help="training steps")
    p.add_argument("--data", type=int, default=50000, help="offline transitions to collect")
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--d4rl", action="store_true",
                   help="load a real D4RL dataset instead of the toy one "
                        "(requires d4rl+mujoco; use --env e.g. halfcheetah-medium-v2)")
    p.add_argument("--d3rlpy", action="store_true",
                   help="load an existing d3rlpy dataset (Pendulum) instead of the "
                        "home-made one (needs only d3rlpy installed)")
    args = p.parse_args()

    device = "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Config: neg={args.neg}, beta={args.beta}, steps={args.steps}, "
          f"data={args.data}, seed={args.seed}")

    # 1. get the fixed offline dataset (toy by default; D4RL or d3rlpy if requested)
    if args.d4rl:
        print(f"Loading D4RL dataset: {args.env} ...")
        S, A, R, NS, D, max_action = load_d4rl_dataset(args.env)
    elif args.d3rlpy:
        print("Loading existing d3rlpy Pendulum dataset ...")
        S, A, R, NS, D, max_action = load_d3rlpy_dataset("pendulum")
    else:
        print("Collecting toy offline dataset (scripted mixed-quality policy)...")
        S, A, R, NS, D, max_action = collect_dataset(args.env, args.data, args.seed)
    print(f"Dataset size: {S.shape[0]} transitions")

    # normalize states (as TD3+BC does)
    mean = S.mean(0, keepdims=True)
    std = S.std(0, keepdims=True) + 1e-3
    S = (S - mean) / std
    NS = (NS - mean) / std

    # normalize rewards to unit std -> keeps Q-values in a sane range and
    # prevents the critic from exploding (the saturation bug we hit).
    reward_std = R.std() + 1e-6
    R = R / reward_std
    print(f"Reward std (used to normalize): {reward_std:.3f}")

    state_dim = S.shape[1]
    action_dim = A.shape[1]
    buffer = ReplayBuffer(S, A, R, NS, D, device)

    # 2. train
    agent = TD3_BC(state_dim, action_dim, max_action, device,
                   use_neg=args.neg, beta=args.beta)

    print("Training...")
    for step in range(args.steps):
        batch = buffer.sample(256)
        log = agent.train_step(batch)
        if (step + 1) % 5000 == 0:
            msg = f"  step {step+1}/{args.steps}"
            for k, v in log.items():
                msg += f" | {k}={v:.3f}"
            print(msg)

    # 3. evaluate (need normalized states at eval time too)
    class NormAgent:
        def __init__(self, agent, mean, std):
            self.agent = agent; self.mean = mean; self.std = std
        def act(self, state):
            return self.agent.act((state - self.mean.flatten()) / self.std.flatten())

    norm_agent = NormAgent(agent, mean, std)
    scores = evaluate(norm_agent, args.env, args.eval_episodes, args.seed)

    print("\n----- RESULTS -----")
    print(f"scores: {[round(float(s), 1) for s in scores]}")
    print(f"mean:   {scores.mean():.1f}")
    print(f"median: {np.median(scores):.1f}")
    print(f"worst:  {scores.min():.1f}")
    print(f"best:   {scores.max():.1f}")


if __name__ == "__main__":
    main()