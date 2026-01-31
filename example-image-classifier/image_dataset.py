import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_loaders(batch_size=128, val_split=0.1, data_dir='./data'):
    """
    Download and prepare CIFAR-10 dataset with train/val/test splits.
    
    CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
    - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 training images
    - 10,000 test images
    
    Args:
        batch_size: Number of images per batch
        val_split: Fraction of training data to use for validation
        data_dir: Directory to store the dataset
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        classes: List of class names
    """
    
    # -- Data Augmentation for Training --
    # These transformations help the model generalize better by creating variations
    train_transform = transforms.Compose([
        # Randomly flip images horizontally (50% chance)
        transforms.RandomHorizontalFlip(),
        
        # Randomly crop the image and resize back to 32x32
        # This simulates different viewpoints
        transforms.RandomCrop(32, padding=4),
        
        # Convert PIL Image to PyTorch tensor (values in [0, 1])
        transforms.ToTensor(),
        
        # Normalize using CIFAR-10 mean and std for each channel (R, G, B)
        # This helps the model train faster and more stably
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # Mean of CIFAR-10 dataset
            std=[0.2470, 0.2435, 0.2616]    # Std of CIFAR-10 dataset
        ),
    ])
    
    # -- Transformations for Validation/Test --
    # No augmentation here, just normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    # Download and load the training dataset
    print("Downloading CIFAR-10 dataset (this may take a few minutes on first run)...")
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load the test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training data into train and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    # DataLoader handles batching, shuffling, and parallel data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle training data each epoch
        num_workers=2,     # Use 2 subprocesses for data loading
        pin_memory=True    # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # No need to shuffle validation data
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Class names for CIFAR-10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {classes}")
    
    return train_loader, val_loader, test_loader, classes


def denormalize_image(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor of shape (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Reverse normalization: x_original = x_normalized * std + mean
    tensor = tensor * std + mean
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def show_sample_images(loader, classes, num_images=8):
    """
    Display a few sample images from the dataset (requires matplotlib).
    
    Args:
        loader: DataLoader to sample from
        classes: List of class names
        num_images: Number of images to display
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    # Get a batch of images
    images, labels = next(iter(loader))
    
    # Plot the images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        # Denormalize and convert to numpy
        img = denormalize_image(images[i])
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {classes[labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved to 'sample_images.png'")
    plt.close()


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing CIFAR-10 dataset loading...")
    train_loader, val_loader, test_loader, classes = get_cifar10_loaders(batch_size=32)
    
    # Show some sample images
    show_sample_images(train_loader, classes)
    
    # Print a sample batch shape
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape:")
    print(f"Images: {images.shape}")  # Should be (batch_size, 3, 32, 32)
    print(f"Labels: {labels.shape}")  # Should be (batch_size,)
