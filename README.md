# PyTorch Starter: Two Complete Examples

This project provides beginner-friendly introductions to PyTorch with **two complete, self-contained examples**:

1. **Character-Level Language Model (LLM)** - Text generation using Transformers
2. **Image Classifier (CIFAR-10)** - Computer vision with CNNs

Each example is in its own directory with dedicated documentation and data storage.

---

## 📁 Project Structure

```
pytorch-starter/
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
├── COMPARISON.md                  # Detailed comparison of both examples
├── PROJECT_STRUCTURE.md           # Complete documentation
├── requirements.txt               # Shared dependencies
│
├── example-llm/                   # Example 1: Language Model
│   ├── README.md                  # LLM-specific documentation
│   ├── model.py                   # Transformer architecture
│   ├── dataset.py                 # Text data loading
│   ├── train.py                   # Training script
│   ├── prepare_data.py            # Dataset download
│   └── data/                      # LLM data directory
│       └── tinyshakespeare.txt
│
└── example-image-classifier/      # Example 2: Image Classifier
    ├── README.md                  # Image classifier documentation
    ├── image_model.py             # CNN architectures
    ├── image_dataset.py           # CIFAR-10 data loading
    ├── train_image.py             # Training script
    ├── predict_image.py           # Inference script
    └── data/                      # Image data directory
        └── cifar-10-batches-py/
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (shared by both examples)
pip install -r requirements.txt
```

### Example 1: Character-Level LLM

```bash
cd example-llm

# Download dataset
python prepare_data.py

# Train the model
python train.py
```

**Result:** Generates Shakespeare-like text after ~5-10 minutes of training.

📖 **[Read the LLM README](example-llm/README.md)** for detailed instructions.

### Example 2: Image Classifier

```bash
cd example-image-classifier

# Train the model (downloads CIFAR-10 automatically)
python train_image.py

# Test predictions
python predict_image.py

# Classify your own image
python predict_image.py path/to/image.jpg
```

**Result:** Achieves 65-70% accuracy on CIFAR-10 after ~20-30 minutes of training.

📖 **[Read the Image Classifier README](example-image-classifier/README.md)** for detailed instructions.

---

## 📚 What You'll Learn

### Example 1: Character-Level LLM

**Domain:** Natural Language Processing (NLP)  
**Architecture:** Transformer (Self-Attention)  
**Task:** Text Generation

**Key Concepts:**
- ✅ Self-attention mechanism
- ✅ Positional embeddings
- ✅ Autoregressive generation
- ✅ Transformer blocks
- ✅ Character-level tokenization

**Files:** 4 Python files, ~325 lines total  
**Training Time:** ~5-10 min (CPU), ~2-3 min (GPU)  
**Difficulty:** Intermediate

---

### Example 2: Image Classifier (CIFAR-10)

**Domain:** Computer Vision (CV)  
**Architecture:** CNN (Convolutional Neural Network)  
**Task:** Multi-class Classification

**Key Concepts:**
- ✅ Convolutional layers
- ✅ Data augmentation
- ✅ Batch normalization
- ✅ Residual connections (ResNet)
- ✅ Transfer learning concepts

**Files:** 4 Python files, ~800 lines total  
**Training Time:** ~20-30 min (CPU), ~5-10 min (GPU)  
**Difficulty:** Beginner-Intermediate

---

## 🎯 Which Example to Start With?

### Start with Image Classifier if you:
- ✅ Want to see faster, visual results
- ✅ Are new to deep learning
- ✅ Prefer working with images
- ✅ Want to learn CNNs (foundation of computer vision)

### Start with LLM if you:
- ✅ Are interested in NLP and text generation
- ✅ Want to understand Transformers (GPT, BERT, etc.)
- ✅ Have some ML background
- ✅ Enjoy creative, generative tasks

### Do Both! 🎓
They teach complementary concepts:
- **LLM**: Sequence modeling, attention, autoregressive generation
- **Image Classifier**: Spatial features, data augmentation, classification metrics

---

## 📊 Comparison at a Glance

| Feature | LLM | Image Classifier |
|---------|-----|------------------|
| **Dataset** | Shakespeare (~1MB) | CIFAR-10 (~170MB) |
| **Input** | Text sequences | 32×32 RGB images |
| **Output** | Next character | Class (1 of 10) |
| **Architecture** | Transformer | CNN |
| **Parameters** | ~3.5M | ~1.2M (Simple) / ~2.5M (ResNet) |
| **Training Time (CPU)** | 5-10 min | 20-30 min |
| **Evaluation** | Generated text quality | Accuracy (65-70%) |

📖 **[Read the full comparison](COMPARISON.md)** for detailed analysis.

---

## 📖 Documentation

Each example has its own comprehensive README:

- **[example-llm/README.md](example-llm/README.md)** - Complete guide for the language model
- **[example-image-classifier/README.md](example-image-classifier/README.md)** - Complete guide for the image classifier

Additional documentation:

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step getting started guide
- **[COMPARISON.md](COMPARISON.md)** - Detailed comparison of both examples
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - File-by-file documentation

---

## 📦 Requirements

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

**Install:**
```bash
pip install -r requirements.txt
```

**GPU Support:**
- CUDA-enabled GPU recommended but not required
- Both examples work on CPU (just slower)
- Training is 5-10× faster on GPU

---

## 🎓 Learning Path

### Week 1: Get Started
1. Install dependencies
2. Run both examples with default settings
3. Read through the code comments
4. Understand the basic training loop

### Week 2: Deep Dive
1. Study the architectures (CNN vs Transformer)
2. Read the example-specific READMEs
3. Experiment with hyperparameters
4. Try the suggested modifications

### Week 3: Advanced
1. Read the comparison guide
2. Modify the architectures
3. Implement custom features
4. Try your own datasets

### Week 4: Projects
1. Combine concepts (e.g., image captioning)
2. Implement transfer learning
3. Add TensorBoard logging
4. Deploy as a web API

---

## 🧪 Suggested Experiments

### For Both Examples
- Change hyperparameters (learning rate, batch size)
- Modify architectures (add/remove layers)
- Train for longer
- Visualize training progress

### LLM-Specific
- Train on different text (code, poetry, your own writing)
- Implement temperature sampling
- Try word-level tokenization
- Increase model size

### Image Classifier-Specific
- Switch between SimpleCNN and ResNetCIFAR
- Add more data augmentation
- Try transfer learning with pre-trained models
- Classify your own images

---

## 🚀 Next Steps

After completing both examples:

1. **Combine Concepts**: Build an image captioning model (CNN encoder + Transformer decoder)
2. **Transfer Learning**: Use pre-trained models from `torchvision.models`
3. **Advanced Datasets**: Try ImageNet, COCO, or custom datasets
4. **Deploy**: Create a web API with Flask or FastAPI
5. **Optimize**: Explore mixed precision training, quantization, and pruning

---

## 🐛 Troubleshooting

### Installation Issues
```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch separately if needed
pip install torch torchvision torchaudio

# Then install remaining dependencies
pip install -r requirements.txt
```

### CUDA/GPU Issues
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, PyTorch will use CPU (slower but works)
```

### Out of Memory
- Reduce batch size in training scripts
- Use smaller model architectures
- Close other applications
- Consider using Google Colab for free GPU

### Import Errors
- Make sure you're in the correct directory
- Check that all files were moved correctly
- Reinstall dependencies

---

## 📚 Additional Resources

### PyTorch
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Forums](https://discuss.pytorch.org/)

### Deep Learning
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Fast.ai Course](https://course.fast.ai/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

### Transformers & NLP
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Computer Vision
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Papers with Code](https://paperswithcode.com/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)

---

## 💡 Tips for Success

1. **Start Simple**: Run examples with default settings first
2. **Read Comments**: Every line of code is explained
3. **Experiment**: Change one thing at a time
4. **Be Patient**: Training takes time, especially on CPU
5. **Ask Questions**: Use PyTorch forums and communities
6. **Have Fun**: Deep learning is exciting - enjoy the journey!

---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
- Open an issue
- Submit a pull request
- Share your experiments and results

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🎉 Acknowledgments

- **Andrej Karpathy** for inspiration from nanoGPT and minGPT
- **PyTorch Team** for the excellent framework
- **CIFAR-10** dataset creators
- **Tiny Shakespeare** dataset

---

**Happy Learning! 🚀**

Start with either example, experiment, and most importantly - have fun building with PyTorch!
