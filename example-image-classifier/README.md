# Image Classifier (CIFAR-10)

A beginner-friendly PyTorch implementation of Convolutional Neural Networks (CNNs) for image classification. This example includes two architectures (SimpleCNN and ResNet-style) trained on the CIFAR-10 dataset.

---

## 🖼️ What You'll Learn

- **Convolutional Neural Networks**: How CNNs extract features from images
- **Data Augmentation**: Techniques to improve model generalization
- **Training Pipeline**: Complete workflow from data loading to evaluation
- **Model Evaluation**: Accuracy metrics, per-class analysis, and inference
- **PyTorch Vision**: Working with `torchvision` for computer vision tasks

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_image.py
```

**What to expect:**
- CIFAR-10 dataset downloads automatically (~170MB)
- Training takes ~20-30 minutes on CPU, ~5-10 minutes on GPU
- Progress shown every 50 batches
- Best model saved to `image_classifier.pth`
- Final test accuracy: **65-70%** (SimpleCNN) or **75-80%** (ResNet)

### 3. Test Predictions

```bash
# Test on random samples from CIFAR-10
python predict_image.py

# Classify your own image
python predict_image.py path/to/your/image.jpg
```

---

## 📁 Files

| File | Purpose | Lines |
|------|---------|-------|
| `image_model.py` | CNN architectures (SimpleCNN, ResNetCIFAR) | 200 |
| `image_dataset.py` | CIFAR-10 data loading and augmentation | 180 |
| `train_image.py` | Training script with validation | 220 |
| `predict_image.py` | Inference script for classification | 200 |

---

## 🎯 CIFAR-10 Dataset

**10 Classes:**
- 🛩️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐕 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚚 truck

**Dataset Size:**
- 50,000 training images (32×32 RGB)
- 10,000 test images
- Automatically split: 90% train, 10% validation

---

## 🏗️ Architectures

### SimpleCNN (Beginner-Friendly)

```
Input (32×32×3)
    ↓
Conv2d(3→32) + BatchNorm + ReLU + MaxPool
    ↓ (16×16×32)
Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓ (8×8×64)
Conv2d(64→128) + BatchNorm + ReLU + MaxPool
    ↓ (4×4×128)
Flatten → FC(2048→512) + ReLU + Dropout
    ↓
FC(512→10)
    ↓
Class Probabilities
```

**Parameters:** ~1.2M  
**Expected Accuracy:** 65-70%

### ResNetCIFAR (Advanced)

```
Input (32×32×3)
    ↓
Conv2d(3→64) + BatchNorm + ReLU
    ↓
ResidualBlock(64→64) × 2
    ↓
ResidualBlock(64→128) × 2 (stride=2)
    ↓
ResidualBlock(128→256) × 2 (stride=2)
    ↓
Global Average Pooling
    ↓
FC(256→10)
    ↓
Class Probabilities
```

**Parameters:** ~2.5M  
**Expected Accuracy:** 75-80%  
**Key Feature:** Skip connections enable deeper networks

---

## ⚙️ Configuration

Edit `train_image.py`:

```python
BATCH_SIZE = 128          # Images per batch
EPOCHS = 20               # Training epochs
LEARNING_RATE = 0.001     # Initial learning rate
MODEL_TYPE = 'simple'     # 'simple' or 'resnet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## 📊 Expected Results

### Training Output

```
============================================================
STEP 1: Loading CIFAR-10 Dataset
============================================================
Downloading CIFAR-10 dataset...
Dataset loaded successfully!
Training samples: 45000
Validation samples: 5000
Test samples: 10000

============================================================
STEP 2: Initializing Model
============================================================
Using SimpleCNN architecture
Total parameters: 1,234,567
Trainable parameters: 1,234,567

Epoch [1/20]
------------------------------------------------------------
  Batch [50/352] Loss: 1.8234 Acc: 32.45%
  Batch [100/352] Loss: 1.6543 Acc: 38.92%
  ...

Epoch Summary:
  Train Loss: 1.6543 | Train Acc: 38.92%
  Val Loss: 1.4321 | Val Acc: 45.67%
  Time: 45.23s | LR: 0.001000
  ✓ New best model saved! (Val Acc: 45.67%)

...

Epoch [20/20]
------------------------------------------------------------
Epoch Summary:
  Train Loss: 0.8234 | Train Acc: 71.23%
  Val Loss: 0.9876 | Val Acc: 68.45%
  Time: 42.15s | LR: 0.000012

============================================================
STEP 5: Final Evaluation on Test Set
============================================================
Test Loss: 0.9654
Test Accuracy: 67.34%

============================================================
Per-Class Accuracy
============================================================
airplane    : 78.50%
automobile  : 82.30%
bird        : 65.40%
cat         : 54.20%
deer        : 62.80%
dog         : 58.90%
frog        : 73.60%
horse       : 71.20%
ship        : 79.40%
truck       : 77.10%
```

### Prediction Output

```bash
python predict_image.py cat.jpg
```

```
============================================================
CIFAR-10 Image Classifier - Inference
============================================================

Loading model...
Model loaded from image_classifier.pth
Validation accuracy: 68.45%

Loading image: cat.jpg
Image shape: torch.Size([1, 3, 32, 32])

Making prediction...

============================================================
Predictions (Top 5)
============================================================
1. cat          87.34% ████████████████████████████████████████████
2. dog          8.12%  ████
3. deer         2.45%  █
4. horse        1.23%  
5. bird         0.86%  

✓ Predicted class: CAT (87.34% confidence)
```

---

## 🔧 Customization

### Switch to ResNet

Edit `train_image.py`:
```python
MODEL_TYPE = 'resnet'  # instead of 'simple'
```

ResNet typically achieves 5-10% higher accuracy but trains slower.

### Adjust Hyperparameters

```python
BATCH_SIZE = 64        # Reduce if out of memory
EPOCHS = 30            # Train longer for better results
LEARNING_RATE = 0.0005 # Lower for more stable training
```

### Modify Data Augmentation

Edit `image_dataset.py`:
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add this
    transforms.RandomRotation(15),  # Add this
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2470, 0.2435, 0.2616]),
])
```

### Add More Layers to SimpleCNN

Edit `image_model.py`:
```python
# In SimpleCNN.__init__
self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
self.bn4 = nn.BatchNorm2d(256)

# In SimpleCNN.forward
x = self.conv4(x)
x = self.bn4(x)
x = F.relu(x)
x = self.pool(x)
```

---

## 🎓 Learning Guide

### Step 1: Understand Data Loading (image_dataset.py)

Key concepts:
- **Transforms**: Preprocessing pipeline for images
- **Data Augmentation**: Random flips and crops to improve generalization
- **Normalization**: Standardizing pixel values for faster training
- **DataLoader**: Efficient batching and parallel loading

**Try this:**
```bash
python image_dataset.py
```
This will show sample images and batch shapes.

### Step 2: Study the Architecture (image_model.py)

**SimpleCNN** - Start here:
1. **Convolutional layers**: Extract features (edges, textures, patterns)
2. **Batch normalization**: Stabilize training
3. **Max pooling**: Reduce spatial dimensions
4. **Fully connected layers**: Make final classification

**ResNetCIFAR** - Advanced:
1. **Residual blocks**: Skip connections help gradients flow
2. **Deeper networks**: More layers without vanishing gradients
3. **Global average pooling**: Reduce parameters

### Step 3: Understand Training (train_image.py)

Follow the pipeline:
1. **Load data**: CIFAR-10 with augmentation
2. **Initialize model**: SimpleCNN or ResNetCIFAR
3. **Training loop**:
   - Forward pass (predict classes)
   - Compute loss (cross-entropy)
   - Backward pass (compute gradients)
   - Update weights (Adam optimizer)
4. **Validation**: Check performance on unseen data
5. **Save best model**: Checkpoint with highest validation accuracy

### Step 4: Make Predictions (predict_image.py)

Learn about:
- **Model loading**: Restore saved weights
- **Preprocessing**: Same transforms as training
- **Inference**: Forward pass without gradients
- **Softmax**: Convert logits to probabilities

---

## 🧪 Experiments to Try

### Beginner
1. ✅ Train with default settings (SimpleCNN)
2. ✅ Test predictions on CIFAR-10 samples
3. ✅ Try classifying your own images
4. ✅ Change `EPOCHS` to 30

### Intermediate
1. 📈 Switch to ResNet architecture
2. 📈 Modify data augmentation
3. 📈 Adjust learning rate and batch size
4. 📈 Add more convolutional layers to SimpleCNN

### Advanced
1. 🚀 Implement transfer learning with pre-trained ResNet
2. 🚀 Add TensorBoard logging for visualization
3. 🚀 Implement mixed precision training
4. 🚀 Try different optimizers (SGD with momentum, AdamW)
5. 🚀 Implement learning rate warmup

---

## 🐛 Troubleshooting

### CUDA out of memory
```python
BATCH_SIZE = 64  # or 32
```

### Low accuracy (< 50%)
- Train longer (`EPOCHS = 30`)
- Check data augmentation is working
- Try ResNet architecture
- Verify normalization values are correct

### Training too slow
- Reduce `BATCH_SIZE` if using CPU
- Use GPU if available
- Reduce `EPOCHS` for quick experiments

### Model not improving after epoch 10
- Learning rate might be too high or too low
- Try different optimizer
- Add more data augmentation
- Use learning rate scheduling (already included)

---

## 📖 Key Concepts Explained

### Convolutional Layer
```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```
- Slides a 3×3 filter across the image
- Detects local patterns (edges, textures)
- Shares weights across spatial positions

### Batch Normalization
```python
nn.BatchNorm2d(32)
```
- Normalizes activations across the batch
- Stabilizes training and speeds up convergence
- Acts as regularization

### Max Pooling
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```
- Reduces spatial dimensions by half
- Keeps the maximum value in each 2×2 region
- Provides translation invariance

### Dropout
```python
nn.Dropout(0.5)
```
- Randomly drops 50% of neurons during training
- Prevents overfitting
- Disabled during evaluation

### Cross-Entropy Loss
```python
nn.CrossEntropyLoss()
```
- Measures how wrong the predictions are
- Combines LogSoftmax and NLLLoss
- Standard for classification tasks

---

## 🎯 Next Steps

After mastering this example:

1. **Transfer Learning**: Use pre-trained models from `torchvision.models`
   ```python
   from torchvision import models
   model = models.resnet18(pretrained=True)
   ```

2. **Custom Dataset**: Train on your own images
   ```python
   from torchvision.datasets import ImageFolder
   dataset = ImageFolder('path/to/data', transform=transform)
   ```

3. **Object Detection**: Try YOLO or Faster R-CNN

4. **Semantic Segmentation**: Implement U-Net

5. **Deploy Model**: Create a web API with Flask or FastAPI

---

## 📚 Resources

- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) - Stanford course
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) - ResNet paper
- [PyTorch Vision Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Papers with Code - CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10)

---

## 💡 Tips

- **Visualize**: Look at sample images to understand the data
- **Monitor**: Watch training/validation accuracy gap for overfitting
- **Experiment**: Try one change at a time
- **Be patient**: Good results take 15-20 epochs
- **Use GPU**: Training is 5-10× faster on GPU

---

Happy Learning! 🚀
