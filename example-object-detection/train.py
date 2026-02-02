import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ObjectDetectionDataset, collate_fn
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an object detection model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ObjectDetectionDataset(args.data_root)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = build_model(args.num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    model.train()
    for _ in range(args.epochs):
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training complete")


if __name__ == "__main__":
    main()
