import copy

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import nn
from torchrl.record.loggers import Logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
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


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        only_q1: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sa = torch.cat([state, action], 1)

        q1 = self.q1(sa)

        if only_q1:
            return q1, None

        q2 = self.q2(sa)

        return q1, q2


class TD3_BC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        lr: float = 3e-4,
        logger: Logger | None = None,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

        self.logger = logger

    def select_action(self, tensor_dict: TensorDictBase) -> TensorDictBase:
        tensor_dict["action"] = self.actor(tensor_dict["observation"])
        return tensor_dict

    def train(self, batch: TensorDictBase) -> dict:
        self.total_it += 1

        batch = batch.to(DEVICE)

        state = batch["observation"]
        action = batch["action"]
        next_state = batch["next", "observation"]
        reward = batch["next", "reward"]
        not_done = 1.0 - batch["next", "terminated"].float()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Clipped double-Q target
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        metrics = {
            "train/critic_loss": critic_loss.item(),
            "train/q1_mean": current_q1.mean().item(),
        }

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)

            q1, _ = self.critic(state, pi, only_q1=True)

            lmbda = self.alpha / q1.abs().mean().detach()

            bc_loss = F.mse_loss(pi, action)
            actor_loss = -lmbda * q1.mean() + bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

            metrics.update(
                {
                    "train/actor_loss": actor_loss.item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/lambda": lmbda.item(),
                }
            )

        if self.logger is not None:
            for key, val in metrics.items():
                self.logger.log_scalar(key, val, step=self.total_it)

        return metrics
