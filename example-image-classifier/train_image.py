import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from image_model import SimpleCNN, ResNetCIFAR
from image_dataset import get_cifar10_loaders
import time

# -- Configuration --
# Hyperparameters for training
BATCH_SIZE = 128          # Number of images per batch
EPOCHS = 20               # Number of complete passes through the dataset
LEARNING_RATE = 0.001     # Initial learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'simple'     # Options: 'simple' or 'resnet'
MODEL_SAVE_PATH = 'image_classifier.pth'

print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Step 1: Load the dataset
print("=" * 60)
print("STEP 1: Loading CIFAR-10 Dataset")
print("=" * 60)
train_loader, val_loader, test_loader, classes = get_cifar10_loaders(
    batch_size=BATCH_SIZE,
    val_split=0.1
)
print()

# Step 2: Initialize the model
print("=" * 60)
print("STEP 2: Initializing Model")
print("=" * 60)
if MODEL_TYPE == 'simple':
    model = SimpleCNN(num_classes=10)
    print("Using SimpleCNN architecture")
else:
    model = ResNetCIFAR(num_classes=10)
    print("Using ResNetCIFAR architecture")

model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()

# Step 3: Setup loss function and optimizer
print("=" * 60)
print("STEP 3: Setting up Training Components")
print("=" * 60)

# CrossEntropyLoss is standard for multi-class classification
# It combines LogSoftmax and NLLLoss in one single class
criterion = nn.CrossEntropyLoss()

# Adam optimizer is a good default choice
# It adapts the learning rate for each parameter
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: gradually decrease learning rate
# CosineAnnealing smoothly reduces LR following a cosine curve
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print(f"Scheduler: CosineAnnealingLR")
print()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test set.
    
    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass only (no gradient computation)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# Step 4: Training Loop
print("=" * 60)
print("STEP 4: Training the Model")
print("=" * 60)

best_val_acc = 0.0
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(EPOCHS):
    start_time = time.time()
    
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print("-" * 60)
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Evaluate on validation set
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
    
    # Update learning rate
    scheduler.step()
    
    # Save training history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)
    
    # Print epoch summary
    epoch_time = time.time() - start_time
    print(f"\nEpoch Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': classes
        }, MODEL_SAVE_PATH)
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

print("\n" + "=" * 60)
print("STEP 5: Final Evaluation on Test Set")
print("=" * 60)

# Load best model
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# Per-class accuracy
print("\n" + "=" * 60)
print("Per-Class Accuracy")
print("=" * 60)

class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:12s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Final test accuracy: {test_acc:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
