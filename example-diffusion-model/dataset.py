"""Dataset utilities for the diffusion example."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloader(batch_size: int = 128, data_dir: str = "data") -> DataLoader:
    """Return a DataLoader for MNIST normalized to [-1, 1]."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
