"""Train a policy with REINFORCE on a multi-armed bandit."""

from __future__ import annotations

import argparse

import torch
from torch import optim

from environment import BernoulliBandit
from model import BanditPolicy


def moving_average(values: list[float], window: int = 100) -> float:
    if not values:
        return 0.0
    chunk = values[-window:]
    return sum(chunk) / len(chunk)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    env = BernoulliBandit(arm_probabilities=args.arm_probs, seed=args.seed)
    policy = BanditPolicy(n_arms=env.n_arms)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    rewards: list[float] = []
    baseline = 0.0

    for episode in range(1, args.episodes + 1):
        action, log_prob = policy.act()
        reward = env.step(action)
        rewards.append(reward)

        baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * reward
        advantage = reward - baseline

        loss = -log_prob * advantage
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % args.log_every == 0 or episode == 1:
            avg_reward = moving_average(rewards, window=args.log_every)
            print(
                f"Episode {episode:5d} | "
                f"avg_reward({args.log_every})={avg_reward:.3f} | "
                f"greedy_arm={policy.greedy_action()}"
            )

    learned_arm = policy.greedy_action()
    optimal_arm = env.best_arm()

    print("\nTraining finished")
    print(f"Learned best arm: {learned_arm}")
    print(f"True best arm:    {optimal_arm}")
    print(f"Final moving average reward ({args.log_every} eps): {moving_average(rewards, args.log_every):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE on an N-armed Bernoulli bandit")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arm-probs",
        type=float,
        nargs="+",
        default=[0.15, 0.4, 0.6, 0.8],
        help="Reward probabilities for each arm",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
