# Project Structure

This document provides an overview of all files in the PyTorch Starter project.

---

## 📁 Directory Structure

```
pytorch-starter/
├── README.md                 # Main project documentation
├── QUICKSTART.md            # Step-by-step getting started guide
├── COMPARISON.md            # Detailed comparison of LLM + image classifier
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
│
├── Example 1: Character-Level LLM
│   ├── model.py            # Transformer architecture
│   ├── dataset.py          # Text data loading and tokenization
│   ├── train.py            # Training script
│   ├── prepare_data.py     # Dataset download script
│   └── model.pth           # Trained model (generated)
│
├── Example 2: Image Classifier
│   ├── image_model.py      # CNN architectures (SimpleCNN, ResNetCIFAR)
│   ├── image_dataset.py    # CIFAR-10 data loading
│   ├── train_image.py      # Training script
│   ├── predict_image.py    # Inference script
│   └── image_classifier.pth # Trained model (generated)
│
├── Example 3: Object Detection
│   ├── dataset.py           # Dataset helpers
│   ├── model.py             # Model builder
│   ├── train.py             # Training scaffold
│   └── requirements.txt     # Example-specific dependencies
│
├── Example 4: Diffusion Model
│   ├── dataset.py           # MNIST data loading
│   ├── model.py             # Noise predictor model
│   └── train.py             # Training script
│
├── Example 5: Reinforcement Learning
│   ├── environment.py       # Bernoulli multi-armed bandit environment
│   ├── model.py             # Learnable categorical policy
│   ├── train.py             # REINFORCE training loop
│   └── README.md            # Example-specific guide
│
└── data/                    # Data directory (auto-created)
    ├── tinyshakespeare.txt  # Shakespeare text (LLM)
    └── cifar-10-batches-py/ # CIFAR-10 dataset (Image)
```

---

## 📄 File Descriptions

### Documentation Files

#### `README.md`
- **Purpose**: Main project documentation
- **Contents**: 
  - Overview of all example projects (LLM, CV, diffusion, RL)
  - Quick start instructions
  - Learning guides
  - Suggested experiments
- **Read this**: First!

#### `QUICKSTART.md`
- **Purpose**: Step-by-step getting started guide
- **Contents**:
  - Installation instructions
  - Detailed walkthrough for each example
  - Troubleshooting tips
  - Expected results
- **Read this**: When you want to run the code

#### `COMPARISON.md`
- **Purpose**: Deep dive into differences and similarities
- **Contents**:
  - Side-by-side comparison table
  - Architecture explanations
  - Use case recommendations
  - Hyperparameter tuning guide
- **Read this**: After running the LLM and image classifier examples

---

### Example 1: Character-Level LLM

#### `model.py` (155 lines)
**Purpose**: Transformer-based language model architecture

**Key Classes**:
- `Head`: Single attention head
- `MultiHeadAttention`: Multiple attention heads in parallel
- `FeedForward`: MLP layer
- `Block`: Complete transformer block
- `SimpleLanguageModel`: Full model with embedding and output layers

**Key Concepts**:
- Self-attention mechanism
- Positional embeddings
- Residual connections
- Layer normalization

**Hyperparameters**:
```python
n_embd = 128      # Embedding dimension
n_head = 4        # Number of attention heads
n_layer = 4       # Number of transformer blocks
dropout = 0.1     # Dropout rate
block_size = 64   # Maximum context length
```

---

#### `dataset.py` (60 lines)
**Purpose**: Text data loading and character-level tokenization

**Key Functions**:
- `load_data(path)`: Load text file and create character mappings
- `get_batch(data, block_size, batch_size, device)`: Generate training batches

**What it does**:
1. Reads text file
2. Creates character-to-integer mapping
3. Converts text to tensor of integers
4. Generates random batches for training

**Returns**:
- `data`: Encoded text as tensor
- `vocab_size`: Number of unique characters
- `encode`: Function to convert text → integers
- `decode`: Function to convert integers → text

---

#### `train.py` (85 lines)
**Purpose**: Training script for the language model

**Configuration**:
```python
batch_size = 32
max_iters = 3000
learning_rate = 1e-3
eval_interval = 300
```

**Training Process**:
1. Load and split data (90% train, 10% validation)
2. Initialize model and optimizer
3. Training loop:
   - Sample batch
   - Forward pass
   - Compute loss
   - Backward pass
   - Update weights
4. Generate sample text
5. Save model

**Output**: 
- Training progress every 300 steps
- Generated Shakespeare-like text
- Saved model (`model.pth`)

---

#### `prepare_data.py` (25 lines)
**Purpose**: Download Tiny Shakespeare dataset

**What it does**:
- Downloads text from Andrej Karpathy's GitHub
- Saves to `data/tinyshakespeare.txt`
- ~1MB file, ~1 million characters

**Usage**:
```bash
python prepare_data.py
```

---

### Example 2: Image Classifier

#### `image_model.py` (200 lines)
**Purpose**: CNN architectures for image classification

**Key Classes**:

1. **`SimpleCNN`** (Beginner-friendly)
   - 3 convolutional blocks
   - Batch normalization
   - Max pooling
   - 2 fully connected layers
   - ~1.2M parameters

2. **`ResidualBlock`** (Building block)
   - Convolutional layers with skip connections
   - Enables deeper networks

3. **`ResNetCIFAR`** (Advanced)
   - Residual connections
   - 6 residual blocks
   - Global average pooling
   - ~2.5M parameters

**Architecture Comparison**:
```
SimpleCNN:
Input (32×32×3) → Conv(32) → Pool → Conv(64) → Pool 
→ Conv(128) → Pool → FC(512) → FC(10)

ResNetCIFAR:
Input (32×32×3) → Conv(64) → ResBlock×2 → ResBlock×2 
→ ResBlock×2 → GlobalAvgPool → FC(10)
```

---

#### `image_dataset.py` (180 lines)
**Purpose**: CIFAR-10 data loading and preprocessing

**Key Functions**:

1. **`get_cifar10_loaders(batch_size, val_split, data_dir)`**
   - Downloads CIFAR-10 automatically
   - Applies data augmentation
   - Creates train/val/test loaders
   - Returns: `train_loader, val_loader, test_loader, classes`

2. **`denormalize_image(tensor)`**
   - Reverses normalization for visualization

3. **`show_sample_images(loader, classes)`**
   - Displays sample images (requires matplotlib)

**Data Augmentation** (Training only):
- Random horizontal flip
- Random crop with padding
- Normalization (mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26])

**CIFAR-10 Classes**:
```python
['airplane', 'automobile', 'bird', 'cat', 'deer',
 'dog', 'frog', 'horse', 'ship', 'truck']
```

---

#### `train_image.py` (220 lines)
**Purpose**: Complete training pipeline for image classifier

**Configuration**:
```python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = 'cuda' if available else 'cpu'
MODEL_TYPE = 'simple'  # or 'resnet'
```

**Key Functions**:

1. **`train_one_epoch(model, loader, criterion, optimizer, device)`**
   - Trains for one epoch
   - Returns: average loss and accuracy

2. **`evaluate(model, loader, criterion, device)`**
   - Evaluates on validation/test set
   - Returns: average loss and accuracy

**Training Process**:
1. Load CIFAR-10 dataset
2. Initialize model (SimpleCNN or ResNetCIFAR)
3. Setup optimizer (Adam) and scheduler (CosineAnnealing)
4. Training loop (20 epochs):
   - Train one epoch
   - Evaluate on validation set
   - Save best model
   - Update learning rate
5. Final evaluation on test set
6. Per-class accuracy analysis

**Output**:
- Progress every 50 batches
- Epoch summary (train/val loss and accuracy)
- Best model saved to `image_classifier.pth`
- Final test accuracy
- Per-class accuracy breakdown

---

#### `predict_image.py` (200 lines)
**Purpose**: Inference script for classifying images

**Key Functions**:

1. **`load_model(model_path, model_type, device)`**
   - Loads trained model from checkpoint
   - Returns model in evaluation mode

2. **`preprocess_image(image_path)`**
   - Resizes image to 32×32
   - Applies normalization
   - Returns preprocessed tensor

3. **`predict_image(model, image_tensor, device, top_k)`**
   - Makes prediction
   - Returns top-k predictions with probabilities

4. **`predict_from_file(image_path, model_path, model_type, device)`**
   - Complete pipeline for single image
   - Displays top-5 predictions

5. **`predict_from_dataset(model_path, model_type, num_samples, device)`**
   - Tests on random samples from CIFAR-10
   - Shows accuracy on samples

**Usage**:
```bash
# Test on CIFAR-10 samples
python predict_image.py

# Classify your own image
python predict_image.py path/to/image.jpg

# Use ResNet model
python predict_image.py path/to/image.jpg resnet
```

**Output Example**:
```
Predictions (Top 5)
1. cat          87.34% ████████████████████████████████████████████
2. dog          8.12%  ████
3. deer         2.45%  █
4. horse        1.23%  
5. bird         0.86%  

✓ Predicted class: CAT (87.34% confidence)
```

---

### Example 3: Object Detection

#### `dataset.py`
**Purpose**: Dataset and data loading helpers for detection tasks.

**What it does**:
- Defines a dataset interface returning images and target dictionaries.
- Serves as a scaffold for COCO/VOC/custom datasets.

#### `model.py`
**Purpose**: Model builder for `torchvision` detection architectures.

**What it does**:
- Initializes a detection model and configures the number of classes.

#### `train.py`
**Purpose**: Training loop scaffold for detection.

**What it does**:
- Iterates over data loaders and updates model weights.
- Provides a template for adding evaluation and metrics.

---

### Example 4: Diffusion Model

#### `dataset.py`
**Purpose**: Load MNIST digits with normalization for diffusion training.

**What it does**:
- Downloads MNIST automatically.
- Normalizes images to [-1, 1].

#### `model.py`
**Purpose**: Lightweight noise prediction network with timestep embeddings.

**What it does**:
- Embeds diffusion timesteps with sinusoidal features.
- Predicts noise for DDPM-style training.

#### `train.py`
**Purpose**: Diffusion training loop on MNIST.

**What it does**:
- Samples random timesteps and noise.
- Trains the model to predict the injected noise.
- Saves a trained checkpoint for later sampling.

---

### Configuration Files

#### `requirements.txt`
**Purpose**: Python package dependencies

**Contents**:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio
numpy
tqdm
requests
matplotlib
Pillow
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

#### `.gitignore`
**Purpose**: Specify files to exclude from version control

**Excluded**:
- `/data` - Dataset files (large)
- `__pycache__/` - Python cache
- `*.pth` - Trained models (large)
- `*.pyc`, `*.pyo` - Compiled Python
- `sample_images.png` - Generated visualizations

---

## 🎯 File Usage Guide

### First Time Setup
1. Read `README.md`
2. Install dependencies: `pip install -r requirements.txt`
3. Follow `QUICKSTART.md`

### Running LLM Example
```bash
python prepare_data.py  # Download data
python train.py         # Train model
```

### Running Image Classifier Example
```bash
python train_image.py    # Train model (downloads CIFAR-10 automatically)
python predict_image.py  # Test predictions
```

### Learning the Code
1. **Beginners**: Start with `image_model.py` (SimpleCNN)
2. **Intermediate**: Read `model.py` (Transformer)
3. **Advanced**: Compare the LLM and image classifier, read `COMPARISON.md`

### Modifying the Code
1. **Change hyperparameters**: Edit configuration section in training scripts
2. **Modify architecture**: Edit model files (`model.py`, `image_model.py`)
3. **Add features**: Extend training scripts with new metrics/logging

---

## 📊 Generated Files

These files are created when you run the examples:

### During Training
- `model.pth` - Trained LLM model (~3.5 MB)
- `image_classifier.pth` - Trained image classifier (~5-10 MB)
- `data/tinyshakespeare.txt` - Shakespeare text (~1 MB)
- `data/cifar-10-batches-py/` - CIFAR-10 dataset (~170 MB)

### During Inference
- `sample_images.png` - Visualization of CIFAR-10 samples

**Note**: These files are excluded from Git (see `.gitignore`)

---

## 🔧 Customization Guide

### To Add a New Model Architecture
1. Create new class in `image_model.py` or `model.py`
2. Update training script to support new model type
3. Test with small dataset first

### To Use Custom Dataset
1. Create new dataset file (follow `image_dataset.py` pattern)
2. Implement `__len__` and `__getitem__` methods
3. Update training script to use new dataset

### To Add Logging
1. Install TensorBoard: `pip install tensorboard`
2. Add `SummaryWriter` to training script
3. Log metrics: `writer.add_scalar('loss', loss, step)`

---

## 📚 Learning Path by File

### Week 1: Basics
- [ ] `README.md` - Understand project overview
- [ ] `QUICKSTART.md` - Run the LLM and image classifier examples
- [ ] `requirements.txt` - Understand dependencies

### Week 2: Image Classifier
- [ ] `image_model.py` - Study SimpleCNN architecture
- [ ] `image_dataset.py` - Learn data loading
- [ ] `train_image.py` - Understand training loop
- [ ] `predict_image.py` - Learn inference

### Week 3: LLM
- [ ] `dataset.py` - Character-level tokenization
- [ ] `model.py` - Transformer architecture
- [ ] `train.py` - Autoregressive training

### Week 4: Advanced
- [ ] `COMPARISON.md` - Compare the LLM and image classifier approaches
- [ ] Modify architectures
- [ ] Implement custom features

---

## 🎓 Code Complexity Rating

| File | Lines | Complexity | Best For |
|------|-------|-----------|----------|
| `prepare_data.py` | 25 | ⭐ Easy | Beginners |
| `dataset.py` | 60 | ⭐⭐ Easy-Medium | Beginners |
| `image_dataset.py` | 180 | ⭐⭐ Medium | Intermediate |
| `image_model.py` (SimpleCNN) | 70 | ⭐⭐ Medium | Intermediate |
| `train_image.py` | 220 | ⭐⭐⭐ Medium | Intermediate |
| `predict_image.py` | 200 | ⭐⭐ Medium | Intermediate |
| `model.py` | 155 | ⭐⭐⭐⭐ Hard | Advanced |
| `train.py` | 85 | ⭐⭐⭐ Medium-Hard | Intermediate |
| `image_model.py` (ResNet) | 130 | ⭐⭐⭐⭐ Hard | Advanced |

---

## 💡 Tips for Navigating the Code

1. **Start Simple**: Begin with `image_model.py` (SimpleCNN)
2. **Read Comments**: All files are heavily commented
3. **Run First**: Execute code before deep reading
4. **Experiment**: Change one thing at a time
5. **Compare**: Use `COMPARISON.md` to understand differences

---

## 🚀 Next Steps

After understanding all files:
1. Implement your own architecture
2. Try different datasets
3. Add advanced features (TensorBoard, mixed precision)
4. Deploy as web API
5. Contribute improvements!

---

Happy Coding! 🎉


---

### Example 5: Reinforcement Learning

#### `environment.py`
**Purpose**: Defines a Bernoulli multi-armed bandit with configurable reward probabilities.

#### `model.py`
**Purpose**: Implements a categorical policy parameterized by trainable logits.

#### `train.py`
**Purpose**: Trains the policy using the REINFORCE algorithm with a moving-average baseline.

**Core Concepts**:
- Exploration vs exploitation
- Policy gradients
- Reward optimization without supervised labels
