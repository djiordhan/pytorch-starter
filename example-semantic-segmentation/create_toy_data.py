import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate_sample(image_size, num_shapes):
    width, height = image_size
    image = Image.new("RGB", image_size, (25, 25, 25))
    mask = Image.new("L", image_size, 0)

    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(num_shapes):
        x0 = random.randint(0, width // 2)
        y0 = random.randint(0, height // 2)
        x1 = random.randint(width // 2, width - 1)
        y1 = random.randint(height // 2, height - 1)

        shape_type = random.choice(["rectangle", "ellipse"])
        color = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )

        if shape_type == "rectangle":
            image_draw.rectangle([x0, y0, x1, y1], fill=color)
            mask_draw.rectangle([x0, y0, x1, y1], fill=1)
        else:
            image_draw.ellipse([x0, y0, x1, y1], fill=color)
            mask_draw.ellipse([x0, y0, x1, y1], fill=1)

    return image, mask


def main():
    parser = argparse.ArgumentParser(description="Create a toy segmentation dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("toy_data"))
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--shapes-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = args.output_dir / "images"
    masks_dir = args.output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_images):
        image, mask = generate_sample(tuple(args.image_size), args.shapes_per_image)
        image.save(images_dir / f"img_{idx:04d}.png")
        mask.save(masks_dir / f"img_{idx:04d}.png")

    print(f"Saved {args.num_images} images to {args.output_dir}")


if __name__ == "__main__":
    main()
