"""
IQL with a negative-policy regulariser, for worst-case offline RL.

Adapted from the CORL / IQL-PyTorch reference implementation
(https://github.com/gwthomas/IQL-PyTorch, https://arxiv.org/abs/2110.06169)
and extended with the negative policy from our TD3+BC work.

Why IQL needs its own formulation
---------------------------------
TD3+BC extracts its policy by maximising Q through the actor, so a negative
policy slots in as a distance penalty in action space:

    pi = argmax  lambda*Q(s,pi(s)) - (pi(s)-a)^2 + beta*(pi(s)-pi_neg(s))^2

IQL never queries the critic on the actor's own actions. Its policy step is
advantage-weighted regression (AWR): a weighted behaviour clone, where the
weight is exp(beta * A(s,a)). So there are two ways to add a negative policy,
and we implement both:

  --neg_mode dist   The direct port of the TD3+BC penalty. Train a negative
                    actor pi_neg by AWR on the danger advantage, then push the
                    main actor's mean away from it. Keeps the paper's story
                    identical across the two algorithms.

  --neg_mode awr    The IQL-native form. Fold the danger advantage straight
                    into the AWR weight:

                        w(s,a) = exp(beta*A(s,a) - beta_neg*A_neg(s,a))

                    No second actor at all; the regulariser is a reweighting
                    of the behaviour clone.

Why A_neg is not just -A
------------------------
The negative stack is trained on the danger reward r_neg = -r, with its own
expectile. IQL's value step gives V = expectile_tau(Q), an UPPER expectile
(tau=0.7), so V_neg = expectile_tau(-Q) = -expectile_{1-tau}(Q), a LOWER one.
The two stacks therefore bootstrap through different continuations:

    Q(s,a)      ~ return-to-go if you act WELL after (s,a)
    -Q_neg(s,a) ~ return-to-go if you act BADLY after (s,a)

An action with high Q but very low -Q_neg is one that usually works and
occasionally ends the episode. That gap is the risk signal, and it is
information the ordinary advantage does not carry. This is why the negative
policy is a better bet in IQL than it was in TD3+BC.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import gymnasium as gym

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ----------------------------------------------------------------------
# dataset loading: Minari (laptop, Colab) or D4RL (cluster), same output
# ----------------------------------------------------------------------
def load_transitions(dataset_id: str):
    """Load a flat (s, a, r, s', done) dataset.

    A slash in the id means Minari ("mujoco/walker2d/medium-v0"), otherwise
    D4RL ("walker2d-medium-v2"). D4RL needs mujoco-py, the old gym and
    Python <= 3.9, so it only works on the cluster; Minari is the maintained
    successor and hosts the same locomotion data.

    Returns (data_dict, gymnasium_env_id).
    """
    if "/" in dataset_id:
        import minari

        ds = minari.load_dataset(dataset_id, download=True)
        obs, act, rew, nobs, term, ep_len = [], [], [], [], [], []
        for ep in ds.iterate_episodes():
            o = np.asarray(ep.observations, dtype=np.float32)   # (T+1, d)
            obs.append(o[:-1]); nobs.append(o[1:])
            act.append(np.asarray(ep.actions, dtype=np.float32))
            rew.append(np.asarray(ep.rewards, dtype=np.float32))
            # bootstrapping must use terminations only, never truncations
            term.append(np.asarray(ep.terminations, dtype=bool))
            ep_len.append(len(ep.actions))
        data = {
            "observations":      np.concatenate(obs, 0),
            "actions":           np.concatenate(act, 0),
            "rewards":           np.concatenate(rew, 0),
            "next_observations": np.concatenate(nobs, 0),
            "terminals":         np.concatenate(term, 0).astype(np.float32),
            "episode_ends":      np.cumsum(ep_len) - 1,
        }
        return data, ds.env_spec.id

    import d4rl        # noqa: F401  (registers the datasets)
    import gym as old_gym

    ds = d4rl.qlearning_dataset(old_gym.make(dataset_id))
    data = {
        "observations":      ds["observations"].astype(np.float32),
        "actions":           ds["actions"].astype(np.float32),
        "rewards":           ds["rewards"].astype(np.float32),
        "next_observations": ds["next_observations"].astype(np.float32),
        "terminals":         ds["terminals"].astype(np.float32),
    }
    n = len(data["rewards"])
    ends = set(np.flatnonzero(data["terminals"] > 0.5).tolist())
    ends.update(range(999, n, 1000))      # d4rl locomotion times out at 1000
    ends.add(n - 1)
    data["episode_ends"] = np.array(sorted(ends), dtype=np.int64)
    base = dataset_id.split("-")[0]
    env_id = {"walker2d": "Walker2d-v5", "hopper": "Hopper-v5",
              "halfcheetah": "HalfCheetah-v5"}.get(base, dataset_id)
    return data, env_id


# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def normalize_reward_locomotion(data: dict) -> dict:
    """CORL's locomotion reward scaling: map the dataset's return range onto
    roughly [0, 1000] so the advantage lands in the range beta=3.0 expects."""
    ends = data["episode_ends"]
    starts = np.r_[0, ends[:-1] + 1]
    returns = np.add.reduceat(data["rewards"], starts)
    span = returns.max() - returns.min()
    data["rewards"] = data["rewards"] / span * 1000.0
    return {"min_ret": float(returns.min()), "max_ret": float(returns.max()),
            "scale": float(1000.0 / span)}


# ----------------------------------------------------------------------
# networks (unchanged from the IQL reference)
# ----------------------------------------------------------------------
class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU, output_activation_fn=None,
                 squeeze_output: bool = False, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_action: float,
                 hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation_fn=nn.Tanh, dropout=dropout)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        s = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(s)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.v = MLP([state_dim, *([hidden_dim] * n_hidden), 1], squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


# ----------------------------------------------------------------------
# replay buffer (the fixed offline dataset, resident on device)
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, data: dict, device: str):
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.states = t(data["observations"])
        self.actions = t(data["actions"])
        self.rewards = t(data["rewards"])
        self.next_states = t(data["next_observations"])
        self.dones = t(data["terminals"])
        self.size = self.states.shape[0]
        self.device = device

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


# ----------------------------------------------------------------------
# IQL + negative policy
# ----------------------------------------------------------------------
class ImplicitQLearning:
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 device: str, max_steps: int,
                 beta: float = 3.0, iql_tau: float = 0.7,
                 discount: float = 0.99, tau: float = 0.005,
                 actor_dropout: float = 0.0, actor_lr: float = 3e-4,
                 vf_lr: float = 3e-4, qf_lr: float = 3e-4,
                 # --- negative policy ---
                 neg_mode: str = "none", beta_neg: float = 0.0,
                 iql_tau_neg: float = 0.7,
                 margin: float = 0.25, normalize_weights: bool = True):
        self.device = device
        self.max_action = max_action
        self.beta = beta
        self.iql_tau = iql_tau
        self.discount = discount
        self.tau = tau
        self.neg_mode = neg_mode
        self.beta_neg = beta_neg
        self.iql_tau_neg = iql_tau_neg
        self.margin = margin
        self.normalize_weights = normalize_weights
        self.total_it = 0

        # ---- the ordinary IQL stack ----
        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = ValueFunction(state_dim).to(device)
        self.actor = GaussianPolicy(state_dim, action_dim, max_action,
                                    dropout=actor_dropout).to(device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=qf_lr)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=vf_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)

        # ---- the danger stack, mirrored, trained on r_neg ----
        self.use_neg = neg_mode != "none"
        if self.use_neg:
            self.qf_neg = TwinQ(state_dim, action_dim).to(device)
            self.q_neg_target = copy.deepcopy(self.qf_neg).requires_grad_(False).to(device)
            self.vf_neg = ValueFunction(state_dim).to(device)
            self.q_neg_optimizer = torch.optim.Adam(self.qf_neg.parameters(), lr=qf_lr)
            self.v_neg_optimizer = torch.optim.Adam(self.vf_neg.parameters(), lr=vf_lr)

        # a second actor is only needed for the distance-penalty variant
        if neg_mode == "dist":
            self.neg_actor = GaussianPolicy(state_dim, action_dim, max_action,
                                            dropout=actor_dropout).to(device)
            self.neg_actor_optimizer = torch.optim.Adam(self.neg_actor.parameters(),
                                                        lr=actor_lr)

    # ---------------- value / critic steps ----------------
    def _update_v(self, vf, q_target, obs, act, iql_tau, optimizer, log, key):
        with torch.no_grad():
            target_q = q_target(obs, act)
        v = vf(obs)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, iql_tau)
        optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        optimizer.step()
        log[key] = v_loss.item()
        return adv

    def _update_q(self, qf, q_target, next_v, obs, act, rew, dones,
                  optimizer, log, key):
        # next_v is evaluated BEFORE the value update this iteration, matching
        # the IQL reference implementation; using the post-update value here
        # would make the Q target depend on a step the reference has not taken.
        targets = rew + (1.0 - dones) * self.discount * next_v
        qs = qf.both(obs, act)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        optimizer.step()
        soft_update(q_target, qf, self.tau)
        log[key] = q_loss.item()

    # ---------------- policy step ----------------
    def _awr_weights(self, adv: torch.Tensor, beta: float) -> torch.Tensor:
        """exp(beta*adv), clamped in log space so it never overflows."""
        return torch.exp((beta * adv).clamp(max=math.log(EXP_ADV_MAX)))

    def _bc_losses(self, actor, obs, act):
        dist = actor(obs)
        return -dist.log_prob(act).sum(-1)

    def _update_policy(self, adv, adv_neg, obs, act, log):
        # AWR weight, optionally carrying the danger advantage (neg_mode=awr).
        # adv_neg is near 0 for the most dangerous in-support actions and very
        # negative for safe ones, so subtracting it upweights the safe ones.
        logits = self.beta * adv.detach()
        if self.neg_mode == "awr":
            logits = logits - self.beta_neg * adv_neg.detach()
        w = torch.exp(logits.clamp(max=math.log(EXP_ADV_MAX)))
        if self.normalize_weights:
            # Hold the mean weight at 1 so that changing beta_neg changes which
            # actions are favoured, not the effective learning rate. Without
            # this, a beta_neg sweep is confounded by a step-size sweep.
            w = w / (w.mean() + 1e-8)
        log["awr_weight_max"] = w.max().item()

        dist = self.actor(obs)
        bc = -dist.log_prob(act).sum(-1)
        policy_loss = torch.mean(w * bc)

        if self.neg_mode == "dist":
            with torch.no_grad():
                mu_neg = self.neg_actor(obs).mean
            mu = dist.mean
            # Mean squared per-dim distance, so the margin does not depend on
            # the action dimension.
            d = ((mu - mu_neg) ** 2).mean(dim=-1)
            # Bounded hinge: the pressure stops once the policy is far enough
            # away. The unbounded -beta*||pi-pi_neg||^2 used in the TD3+BC
            # version has no such stop, which is what made large beta collapse.
            pen = torch.clamp(self.margin - d, min=0.0)
            policy_loss = policy_loss + self.beta_neg * pen.mean()
            log["neg_dist"] = d.mean().item()
            log["neg_penalty"] = pen.mean().item()

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log["actor_loss"] = policy_loss.item()

    def _update_neg_policy(self, adv_neg, obs, act, log):
        """The negative actor is an ordinary AWR fit against the danger
        advantage: it clones the dataset actions that are good at being unsafe."""
        w = self._awr_weights(adv_neg.detach(), self.beta)
        if self.normalize_weights:
            w = w / (w.mean() + 1e-8)
        bc = self._bc_losses(self.neg_actor, obs, act)
        loss = torch.mean(w * bc)
        self.neg_actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.neg_actor_optimizer.step()
        log["neg_actor_loss"] = loss.item()

    # ---------------- one training iteration ----------------
    def train(self, batch) -> Dict[str, float]:
        self.total_it += 1
        obs, act, rew, next_obs, dones = batch
        log: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_obs)
            next_v_neg = self.vf_neg(next_obs) if self.use_neg else None

        adv = self._update_v(self.vf, self.q_target, obs, act, self.iql_tau,
                             self.v_optimizer, log, "value_loss")
        self._update_q(self.qf, self.q_target, next_v, obs, act, rew,
                       dones, self.q_optimizer, log, "q_loss")

        adv_neg = None
        if self.use_neg:
            rew_neg = -rew          # the danger reward
            adv_neg = self._update_v(self.vf_neg, self.q_neg_target, obs, act,
                                     self.iql_tau_neg, self.v_neg_optimizer,
                                     log, "value_loss_neg")
            self._update_q(self.qf_neg, self.q_neg_target, next_v_neg, obs, act,
                           rew_neg, dones, self.q_neg_optimizer,
                           log, "q_loss_neg")
            log["adv_neg_mean"] = adv_neg.mean().item()
            if self.neg_mode == "dist":
                self._update_neg_policy(adv_neg, obs, act, log)

        self._update_policy(adv, adv_neg, obs, act, log)
        log["adv_mean"] = adv.mean().item()
        return log


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_actor(env, actor, device, n_episodes, seed, state_mean, state_std):
    """Roll the deterministic policy out and keep every episode separately.

    The worst case is a property of the return DISTRIBUTION, so we return the
    whole per-episode vector rather than a summary.
    """
    actor.eval()
    returns_raw, returns_pen, violations, lengths, fell = [], [], [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        r_raw = r_pen = v = 0.0
        steps = 0
        terminated = False
        while not done:
            a = actor.act((obs - state_mean) / state_std, device)
            obs, reward, terminated, truncated, info = env.step(a)
            r_pen += float(reward)
            r_raw += float(info.get("reward_raw", reward))
            v += float(info.get("cost", 0.0))
            steps += 1
            done = terminated or truncated
        returns_raw.append(r_raw); returns_pen.append(r_pen)
        violations.append(v); lengths.append(steps); fell.append(float(terminated))
    actor.train()
    return {
        "returns": np.asarray(returns_raw, dtype=np.float64),
        "returns_penalised": np.asarray(returns_pen, dtype=np.float64),
        "violations": np.asarray(violations, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.float64),
        "fell": np.asarray(fell, dtype=np.float64),
    }


def cvar(x: np.ndarray, alpha: float) -> float:
    """Mean of the worst alpha-fraction of episodes: a stable left-tail
    statistic. The plain minimum over N episodes has enormous variance, which
    is why our earlier worst-case numbers were not separable from noise."""
    k = max(1, int(math.ceil(alpha * len(x))))
    return float(np.sort(x)[:k].mean())


def summarise(ev: dict) -> Dict[str, float]:
    r = ev["returns"]
    return {
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "worst": float(r.min()),
        "best": float(r.max()),
        "std": float(r.std()),
        "cvar10": cvar(r, 0.10),
        "cvar25": cvar(r, 0.25),
        "p5": float(np.percentile(r, 5)),
        "fall_rate": float(ev["fell"].mean()),
        "mean_length": float(ev["lengths"].mean()),
        "violations_per_ep": float(ev["violations"].mean()),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IQL with a negative-policy regulariser")
    p.add_argument("--dataset", default="mujoco/walker2d/medium-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=300_000, help="offline gradient steps")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--n_episodes", type=int, default=100)
    # IQL
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--iql_tau", type=float, default=0.7)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_dropout", type=float, default=0.0)
    p.add_argument("--normalize_reward", type=int, default=1)
    p.add_argument("--no_weight_norm", action="store_true",
                   help="disable AWR weight normalisation (matches stock CORL)")
    # negative policy
    p.add_argument("--neg_mode", default="none", choices=["none", "awr", "dist"])
    p.add_argument("--beta_neg", type=float, default=0.0)
    p.add_argument("--iql_tau_neg", type=float, default=0.7)
    p.add_argument("--margin", type=float, default=0.25,
                   help="hinge margin for --neg_mode dist")
    p.add_argument("--out", default="results_iql")
    p.add_argument("--tag", default="")
    return p


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    data, env_id = load_transitions(args.dataset)
    reward_info = normalize_reward_locomotion(data) if args.normalize_reward else {}

    state_mean = data["observations"].mean(0)
    state_std = data["observations"].std(0) + 1e-3
    data["observations"] = (data["observations"] - state_mean) / state_std
    data["next_observations"] = (data["next_observations"] - state_mean) / state_std

    state_dim = data["observations"].shape[1]
    action_dim = data["actions"].shape[1]
    env = gym.make(env_id)
    max_action = float(env.action_space.high[0])

    # the tag becomes the filename, so every swept hyper-parameter that can
    # differ between runs has to appear in it or runs silently overwrite
    tag = args.tag or (
        f"{args.neg_mode}"
        + (f"_b{args.beta_neg:g}" if args.neg_mode != "none" else "")
        + (f"_tau{args.iql_tau:g}" if args.iql_tau != 0.7 else "")
        + (f"_tauneg{args.iql_tau_neg:g}" if args.neg_mode != "none" and args.iql_tau_neg != 0.7 else "")
    )
    run_name = f"IQL__{tag}__seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    print("=" * 66)
    print(f"{run_name}")
    print(f"  dataset  {args.dataset}  ({data['observations'].shape[0]:,} transitions)")
    print(f"  env      {env_id}   device {device}")
    print(f"  neg_mode {args.neg_mode}  beta_neg {args.beta_neg}")
    print("=" * 66)

    buffer = ReplayBuffer(data, device)
    trainer = ImplicitQLearning(
        state_dim, action_dim, max_action, device, max_steps=args.steps,
        beta=args.beta, iql_tau=args.iql_tau, discount=args.discount, tau=args.tau,
        actor_dropout=args.actor_dropout, neg_mode=args.neg_mode,
        beta_neg=args.beta_neg, iql_tau_neg=args.iql_tau_neg,
        margin=args.margin,
        normalize_weights=not args.no_weight_norm)

    t0 = time.time()
    history = []
    for step in range(args.steps):
        log = trainer.train(buffer.sample(args.batch_size))
        if (step + 1) % args.eval_freq == 0 or step + 1 == args.steps:
            ev = eval_actor(env, trainer.actor, device, args.n_episodes,
                            args.seed, state_mean, state_std)
            s = summarise(ev)
            # keep a few training-side diagnostics next to the eval metrics:
            # adv_neg tells us the danger critic is learning something, and
            # neg_dist says whether the hinge margin is in a sane range
            diag = {k: log[k] for k in
                    ("adv_mean", "adv_neg_mean", "neg_dist", "neg_penalty",
                     "awr_weight_max") if k in log}
            history.append({"step": step + 1, **s, **diag})
            print(f"  step {step+1:>7,}  mean {s['mean']:7.1f}  median {s['median']:7.1f}  "
                  f"worst {s['worst']:7.1f}  best {s['best']:7.1f}  "
                  f"CVaR10 {s['cvar10']:7.1f}  falls {100*s['fall_rate']:4.0f}%  "
                  f"[{time.time()-t0:5.0f}s]")

    result = {
        "run": run_name,
        "config": vars(args),
        "env_id": env_id,
        "device": device,
        "reward_info": reward_info,
        "wallclock_s": time.time() - t0,
        "history": history,
        "final": summarise(ev),
        "final_returns": ev["returns"].tolist(),
        "final_violations": ev["violations"].tolist(),
        "final_fell": ev["fell"].tolist(),
    }
    path = os.path.join(args.out, run_name + ".json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
