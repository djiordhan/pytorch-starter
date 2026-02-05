# Example Project: Reinforcement Learning (Multi-Armed Bandit)

This example introduces reinforcement learning with a compact, dependency-light setup.
It trains a policy with **REINFORCE** to solve a Bernoulli multi-armed bandit.

## Project Structure

- `environment.py`: Bandit environment with configurable reward probabilities.
- `model.py`: Learnable categorical policy over actions.
- `train.py`: Training loop with policy gradient and moving-average baseline.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --episodes 3000 --arm-probs 0.15 0.4 0.6 0.8
```

Expected behavior:
- The moving average reward increases over time.
- The learned greedy arm converges to the arm with highest reward probability.

## Things to Try

- Change `--arm-probs` to a harder setup (closer probabilities).
- Increase `--episodes` and reduce `--learning-rate` for more stable convergence.
- Remove the baseline (set `--baseline-momentum 0`) to compare variance.
