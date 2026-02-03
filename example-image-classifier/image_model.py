import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    
    This CNN follows a classic architecture pattern:
    - Convolutional layers to extract features from images
    - Pooling layers to reduce spatial dimensions
    - Fully connected layers to make final predictions
    
    Perfect for CIFAR-10 (32x32 RGB images, 10 classes)
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # -- Convolutional Block 1 --
        # Input: 3 channels (RGB), Output: 32 feature maps
        # Kernel size 3x3 with padding=1 keeps spatial dimensions the same
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization helps stabilize training
        
        # -- Convolutional Block 2 --
        # Input: 32 channels, Output: 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # -- Convolutional Block 3 --
        # Input: 64 channels, Output: 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # -- Max Pooling --
        # Reduces spatial dimensions by half (e.g., 32x32 -> 16x16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # -- Dropout for regularization --
        # Randomly drops 50% of neurons during training to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # -- Fully Connected Layers --
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        # So we have 128 feature maps of size 4x4 = 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Block 1: Conv -> BatchNorm -> ReLU -> Pool
        # Shape: (B, 3, 32, 32) -> (B, 32, 32, 32) -> (B, 32, 16, 16)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2: Conv -> BatchNorm -> ReLU -> Pool
        # Shape: (B, 32, 16, 16) -> (B, 64, 16, 16) -> (B, 64, 8, 8)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3: Conv -> BatchNorm -> ReLU -> Pool
        # Shape: (B, 64, 8, 8) -> (B, 128, 8, 8) -> (B, 128, 4, 4)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten the feature maps for fully connected layers
        # Shape: (B, 128, 4, 4) -> (B, 2048)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer 1 with dropout
        # Shape: (B, 2048) -> (B, 512)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fully connected layer 2 (output layer)
        # Shape: (B, 512) -> (B, num_classes)
        x = self.fc2(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    A Residual Block with skip connections (inspired by ResNet).
    
    Skip connections help gradients flow better during training,
    allowing us to train deeper networks.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        # If dimensions change, we need to adjust the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    A small ResNet-style network for CIFAR-10.
    
    This is more advanced than SimpleCNN and typically achieves better accuracy.
    Uses residual connections to enable training of deeper networks.
    """
    
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Global average pooling and final classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper function to create a sequence of residual blocks"""
        layers = []
        # First block may have stride > 1 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final classifier
        x = self.fc(x)
        
        return x
