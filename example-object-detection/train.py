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
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--save-path", type=Path, default=Path("detector.pth"))
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
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0 or step == len(data_loader):
                avg = running_loss / step
                print(f"epoch={epoch + 1} step={step}/{len(data_loader)} avg_loss={avg:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
