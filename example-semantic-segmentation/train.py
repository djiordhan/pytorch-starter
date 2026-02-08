import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--data-root", type=Path, default=Path("toy_data"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=Path, default=Path("segmenter.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
