"""Policy network for a minimal reinforcement learning example."""

from __future__ import annotations

import torch
from torch import nn


class BanditPolicy(nn.Module):
    """Learnable categorical policy over bandit arms."""

    def __init__(self, n_arms: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_arms))

    def action_distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)

    def act(self) -> tuple[int, torch.Tensor]:
        dist = self.action_distribution()
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    @torch.no_grad()
    def greedy_action(self) -> int:
        return int(torch.argmax(self.logits).item())
