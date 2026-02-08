import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from model import build_model


def overlay_mask(image, mask, alpha=0.4):
    image_np = np.array(image)
    overlay = image_np.copy()
    overlay[mask == 1] = [255, 0, 0]
    blended = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def main():
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)["out"]
        pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

    blended = overlay_mask(image, pred_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blended.save(args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
