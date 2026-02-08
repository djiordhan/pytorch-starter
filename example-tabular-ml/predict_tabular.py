"""Run inference with the trained tabular classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tabular_model import TabularMLP


def _parse_features(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _prepare_features(
    values: Sequence[float],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    feature_array = np.array(values, dtype=np.float32)
    normalized = (feature_array - mean.numpy()) / std.numpy()
    return torch.from_numpy(normalized).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular classifier inference")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of feature values.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("tabular_classifier.pth"),
        help="Path to the saved model checkpoint.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Train the model first with train_tabular.py"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    n_features = checkpoint["n_features"]

    if args.features:
        values = _parse_features(args.features)
        if len(values) != n_features:
            raise ValueError(
                f"Expected {n_features} features, but got {len(values)} values."
            )
    else:
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 1.0, size=n_features).tolist()
        print("No features provided. Using a random sample:")
        print(values)

    model = TabularMLP(
        input_dim=n_features,
        hidden_sizes=checkpoint["hidden_sizes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features = _prepare_features(values, checkpoint["feature_mean"], checkpoint["feature_std"])
    logits = model(features)
    prob = torch.sigmoid(logits).item()
    prediction = "Positive" if prob > 0.5 else "Negative"

    print("=" * 60)
    print("Tabular Classifier Prediction")
    print("=" * 60)
    print(f"Probability of Positive: {prob:.3f}")
    print(f"Predicted Class: {prediction}")


if __name__ == "__main__":
    main()
