import argparse
from pathlib import Path
import random

from PIL import Image, ImageDraw
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a toy object-detection dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("toy_data"))
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_color() -> tuple[int, int, int]:
    return tuple(random.randint(30, 230) for _ in range(3))


def clamp_box(x1: int, y1: int, x2: int, y2: int, size: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, size - 2))
    y1 = max(0, min(y1, size - 2))
    x2 = max(x1 + 1, min(x2, size - 1))
    y2 = max(y1 + 1, min(y2, size - 1))
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    images_dir = args.output_dir / "images"
    ann_dir = args.output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_images):
        image = Image.new("RGB", (args.image_size, args.image_size), (245, 245, 245))
        draw = ImageDraw.Draw(image)

        num_objects = random.randint(1, 3)
        boxes = []
        labels = []

        for _ in range(num_objects):
            w = random.randint(args.image_size // 8, args.image_size // 3)
            h = random.randint(args.image_size // 8, args.image_size // 3)
            x1 = random.randint(0, args.image_size - w - 1)
            y1 = random.randint(0, args.image_size - h - 1)
            x2 = x1 + w
            y2 = y1 + h
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, args.image_size)

            draw.rectangle((x1, y1, x2, y2), fill=random_color(), outline=(0, 0, 0), width=2)
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # single foreground class

        image_path = images_dir / f"img_{i:04d}.jpg"
        ann_path = ann_dir / f"img_{i:04d}.pt"

        image.save(image_path)
        torch.save(
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            },
            ann_path,
        )

    print(f"Created {args.num_images} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
