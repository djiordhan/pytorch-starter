# PyTorch Starter: Two Complete Examples

This project provides beginner-friendly introductions to PyTorch with **two complete examples**:
1. **Character-level Language Model (LLM)** - Text generation using Transformers
2. **Image Classifier** - Computer vision with CNNs on CIFAR-10

Both examples include well-commented code to help you understand PyTorch fundamentals.

---

## 📚 Example 1: Character-Level Language Model (LLM)

A Transformer-based language model inspired by GPT, trained on Shakespeare's works to generate text.

### Files
- `model.py`: SimpleLanguageModel architecture (Transformer blocks, multi-head attention)
- `dataset.py`: Character-level tokenization and data loading
- `train.py`: Training script with text generation
- `prepare_data.py`: Downloads the Tiny Shakespeare dataset

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
python prepare_data.py

# 3. Train the model (3000 iterations)
python train.py
```

The model will train and then generate Shakespeare-like text!

### Learning Guide
- **`model.py`**: Learn how Transformers work - embeddings, self-attention, and residual connections
- **`train.py`**: Understand the PyTorch training loop (forward pass → loss → backward pass → optimizer step)
- **`dataset.py`**: See how text is converted to tensors

### Experiments to Try
- Change `n_layer`, `n_head`, or `n_embd` in `model.py` to see how model size affects results
- Replace `tinyshakespeare.txt` with your own text file
- Implement sub-word tokenization (BPE) instead of character-level

---

## 🖼️ Example 2: Image Classifier (CIFAR-10)

A Convolutional Neural Network (CNN) that classifies images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### Files
- `image_model.py`: Two CNN architectures (SimpleCNN and ResNetCIFAR)
- `image_dataset.py`: CIFAR-10 data loading with augmentation
- `train_image.py`: Training script with validation and metrics
- `predict_image.py`: Inference script for classifying images

### Quick Start

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Train the model (downloads CIFAR-10 automatically)
python train_image.py

# 3. Test on random samples from test set
python predict_image.py

# 4. Classify your own image
python predict_image.py path/to/your/image.jpg
```

### What You'll Learn

**`image_model.py`**: Two architectures to compare
- **SimpleCNN**: Basic CNN with conv layers, pooling, and fully connected layers
- **ResNetCIFAR**: Advanced architecture with residual connections (skip connections)

**`train_image.py`**: Complete training pipeline
- Data augmentation (random flips, crops)
- Learning rate scheduling (CosineAnnealing)
- Validation and checkpointing
- Per-class accuracy analysis

**`image_dataset.py`**: Computer vision data handling
- Image normalization and preprocessing
- Train/validation/test splits
- Data augmentation techniques

### Training Output

You'll see detailed progress during training:
```
Epoch [1/20]
  Batch [50/352] Loss: 1.8234 Acc: 32.45%
  ...
Epoch Summary:
  Train Loss: 1.6543 | Train Acc: 38.92%
  Val Loss: 1.4321 | Val Acc: 45.67%
  ✓ New best model saved!
```

After training, you'll get per-class accuracy:
```
Per-Class Accuracy
airplane    : 78.50%
automobile  : 82.30%
bird        : 65.40%
...
```

### Experiments to Try
- Switch between `SimpleCNN` and `ResNetCIFAR` by changing `MODEL_TYPE` in `train_image.py`
- Adjust hyperparameters: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`
- Modify data augmentation in `image_dataset.py`
- Add more convolutional layers to `SimpleCNN`
- Try different optimizers (SGD with momentum, AdamW)

---

## 🎯 Which Example to Start With?

**Start with Image Classifier if you:**
- Want to see faster, more tangible results
- Are interested in computer vision
- Prefer working with visual data

**Start with LLM if you:**
- Are interested in NLP and text generation
- Want to understand Transformers (foundation of GPT, BERT, etc.)
- Enjoy seeing creative text output

**Do both!** They teach complementary concepts:
- LLM: Sequence modeling, attention mechanisms, autoregressive generation
- Image Classifier: Convolutional networks, image preprocessing, classification metrics

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
```

Install with: `pip install -r requirements.txt`

---

## 🚀 Next Steps

After completing both examples:
1. **Combine concepts**: Try building an image captioning model (CNN encoder + Transformer decoder)
2. **Use pre-trained models**: Explore transfer learning with models from `torchvision.models`
3. **Advanced datasets**: Try ImageNet, COCO, or custom datasets
4. **Deploy your model**: Learn about model serving with Flask or FastAPI
5. **Optimize**: Explore mixed precision training, model quantization, and pruning

---

## 📖 Additional Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

Happy Learning! 🎉
