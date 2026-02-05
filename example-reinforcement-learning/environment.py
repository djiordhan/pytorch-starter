"""Simple multi-armed bandit environment for reinforcement learning demos."""

from __future__ import annotations

import torch


class BernoulliBandit:
    """N-armed bandit with Bernoulli rewards."""

    def __init__(self, arm_probabilities: list[float], seed: int = 42) -> None:
        if not arm_probabilities:
            raise ValueError("arm_probabilities must contain at least one arm")

        self.probs = torch.tensor(arm_probabilities, dtype=torch.float32)
        if torch.any((self.probs < 0.0) | (self.probs > 1.0)):
            raise ValueError("Each arm probability must be in [0, 1]")

        self.generator = torch.Generator().manual_seed(seed)

    @property
    def n_arms(self) -> int:
        return int(self.probs.numel())

    def step(self, action: int) -> float:
        """Sample reward (0 or 1) for the selected action."""
        if action < 0 or action >= self.n_arms:
            raise IndexError(f"action {action} out of range for {self.n_arms} arms")

        reward = torch.bernoulli(self.probs[action], generator=self.generator)
        return float(reward.item())

    def best_arm(self) -> int:
        return int(torch.argmax(self.probs).item())

    def best_expected_reward(self) -> float:
        return float(torch.max(self.probs).item())
