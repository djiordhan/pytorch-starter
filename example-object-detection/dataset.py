from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


@dataclass
class Annotation:
    boxes: torch.Tensor
    labels: torch.Tensor


class ObjectDetectionDataset(Dataset):
    """Placeholder dataset. Replace with logic to load your images and labels."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.jpg"))
        self.annotations = sorted((self.root / "annotations").glob("*.pt"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.images[idx]
        annotation_path = self.annotations[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)
        annotation = torch.load(annotation_path)

        target = {
            "boxes": annotation["boxes"],
            "labels": annotation["labels"],
        }
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)
