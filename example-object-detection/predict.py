import argparse
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F

from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object-detection inference on one image")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("prediction.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.num_classes)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    output = image.copy()
    draw = ImageDraw.Draw(output)

    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score.item() < args.score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        draw.text((x1, y1), f"{score.item():.2f}", fill=(255, 0, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved prediction image to {args.output}")


if __name__ == "__main__":
    main()
