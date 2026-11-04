"""
Proximal Policy Optimization (PPO) Implementation for Reference-Conditioned H1 Locomotion.
Actor-Critic MLP: 178 -> 512 -> 256 -> 128 -> 19 (Actor) / 1 (Critic).
Hyperparameters: gamma=0.99, gae_lambda=0.95, clip_eps=0.2, lr=3e-4, grad_clip=1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim: int = 178, action_dim: int = 19):
        super().__init__()
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        val = self.critic(obs).squeeze(-1)
        return mean, val

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, val = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, val


class PPOAgent:
    def __init__(
        self, 
        obs_dim: int = 178, 
        action_dim: int = 19, 
        lr: float = 3e-4, 
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ac = ActorCriticMLP(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.ac.parameters(), lr=lr, weight_decay=1e-4)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.grad_clip = 1.0

    def update(self, obs_b: torch.Tensor, act_b: torch.Tensor, logp_b: torch.Tensor, returns_b: torch.Tensor, adv_b: torch.Tensor) -> Dict[str, float]:
        self.ac.train()
        
        # Normalize advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        mean, val = self.ac(obs_b)
        std = torch.exp(self.ac.log_std).expand_as(mean)
        dist = Normal(mean, std)

        new_logp = dist.log_prob(act_b).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_logp - logp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(val, returns_b)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        gnorm = float(nn.utils.clip_grad_norm_(self.ac.parameters(), self.grad_clip).item())
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": gnorm
        }
