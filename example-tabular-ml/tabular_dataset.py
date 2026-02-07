"""Utilities for generating a synthetic tabular classification dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def _generate_synthetic_data(
    n_samples: int,
    n_features: int,
    seed: int,
    noise_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    weights = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    bias = rng.normal(0.0, 0.5)
    logits = features @ weights + bias + rng.normal(0.0, noise_scale, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (probs > 0.5).astype(np.float32)
    return features, labels


def _split_indices(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    test_idx = indices[:test_size]
    val_idx = indices[test_size : test_size + val_size]
    train_idx = indices[test_size + val_size :]
    return train_idx, val_idx, test_idx


def _standardize(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    train_scaled = (train_features - mean) / std
    val_scaled = (val_features - mean) / std
    test_scaled = (test_features - mean) / std
    return train_scaled, val_scaled, test_scaled, mean, std


def create_dataloaders(
    batch_size: int = 128,
    n_samples: int = 10000,
    n_features: int = 12,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    noise_scale: float = 0.6,
) -> DatasetSplits:
    features, labels = _generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
        noise_scale=noise_scale,
    )
    train_idx, val_idx, test_idx = _split_indices(
        n_samples=n_samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_features = features[train_idx]
    val_features = features[val_idx]
    test_features = features[test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_scaled, val_scaled, test_scaled, mean, std = _standardize(
        train_features,
        val_features,
        test_features,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_scaled),
        torch.from_numpy(train_labels),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_scaled),
        torch.from_numpy(val_labels),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_scaled),
        torch.from_numpy(test_labels),
    )

    return DatasetSplits(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        feature_mean=torch.from_numpy(mean),
        feature_std=torch.from_numpy(std),
    )
