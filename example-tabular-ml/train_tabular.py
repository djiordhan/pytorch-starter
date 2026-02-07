"""Train a tabular classifier on a synthetic dataset."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from tabular_dataset import create_dataloaders
from tabular_model import TabularMLP


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == labels).float().mean().item()


def evaluate(model: nn.Module, loader, loss_fn, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, labels)
            total_batches += 1
    return total_loss / total_batches, total_acc / total_batches


def main() -> None:
    batch_size = 128
    epochs = 20
    learning_rate = 3e-4
    weight_decay = 1e-2
    hidden_sizes = (64, 32)
    dropout = 0.2
    n_samples = 12000
    n_features = 12
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    splits = create_dataloaders(
        batch_size=batch_size,
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
    )

    model = TabularMLP(
        input_dim=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = -math.inf
    checkpoint_path = Path("tabular_classifier.pth")

    print("=" * 60)
    print("STEP 1: Training Tabular Classifier")
    print("=" * 60)
    print(f"Train batches: {len(splits.train_loader)} | Val batches: {len(splits.val_loader)}")
    print(f"Device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(splits.train_loader, desc=f"Epoch {epoch}/{epochs}")
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_acc = accuracy_from_logits(logits, labels)
            running_loss += loss.item()
            running_acc += batch_acc
            progress.set_postfix({"loss": loss.item(), "acc": f"{batch_acc:.3f}"})

        train_loss = running_loss / len(splits.train_loader)
        train_acc = running_acc / len(splits.train_loader)
        val_loss, val_acc = evaluate(model, splits.val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": splits.feature_mean,
                    "feature_std": splits.feature_std,
                    "n_features": n_features,
                    "hidden_sizes": hidden_sizes,
                    "dropout": dropout,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved new best model to {checkpoint_path}")

    test_loss, test_acc = evaluate(model, splits.test_loader, loss_fn, device)
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
