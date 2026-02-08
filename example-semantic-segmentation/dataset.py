from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class SegmentationDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.image_paths = sorted((self.root / "images").glob("*.png"))
        self.mask_paths = sorted((self.root / "masks").glob("*.png"))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks do not match.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        image_tensor = F.to_tensor(image)
        mask_array = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)

        return image_tensor, mask_tensor
