# PyTorch Starter: Multiple Complete Examples

This project provides beginner-friendly introductions to PyTorch with **multiple complete, self-contained examples**:

1. **Character-Level Language Model (LLM)** - Text generation using Transformers
2. **Image Classifier (CIFAR-10)** - Computer vision with CNNs
3. **Object Detection** - Lightweight scaffolding for detection models
4. **Semantic Segmentation** - Pixel-wise labeling with segmentation models
5. **Diffusion Model (MNIST)** - Generative modeling with noise prediction
6. **Reinforcement Learning (Bandit)** - Policy gradients with REINFORCE
7. **Tabular ML (Binary Classification)** - MLP on synthetic tabular data

Each example is in its own directory with dedicated documentation and data storage.

---

## 📁 Project Structure

```
pytorch-starter/
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
├── COMPARISON.md                  # Detailed comparison of LLM + image classifier
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
├── example-image-classifier/      # Example 2: Image Classifier
│   ├── README.md                  # Image classifier documentation
│   ├── image_model.py             # CNN architectures
│   ├── image_dataset.py           # CIFAR-10 data loading
│   ├── train_image.py             # Training script
│   ├── predict_image.py           # Inference script
│   └── data/                      # Image data directory
│       └── cifar-10-batches-py/
│
├── example-object-detection/      # Example 3: Object Detection
│   ├── README.md                  # Object detection documentation
│   ├── dataset.py                 # Dataset helpers
│   ├── model.py                   # Model builder
│   ├── train.py                   # Training scaffold
│   └── requirements.txt           # Example-specific dependencies
│
├── example-semantic-segmentation/ # Example 4: Semantic Segmentation
│   ├── README.md                  # Segmentation documentation
│   ├── create_toy_data.py         # Toy data generator
│   ├── dataset.py                 # Dataset helpers
│   ├── model.py                   # Model builder
│   ├── train.py                   # Training scaffold
│   ├── predict.py                 # Inference script
│   └── requirements.txt           # Example-specific dependencies
│
├── example-diffusion-model/       # Example 5: Diffusion Model
│   ├── README.md                  # Diffusion model documentation
│   ├── dataset.py                 # MNIST data loader
│   ├── model.py                   # Noise predictor model
│   └── train.py                   # Training script
│
├── example-reinforcement-learning/ # Example 6: Reinforcement Learning
│   ├── README.md                  # RL example documentation
│   ├── environment.py             # Multi-armed bandit environment
│   ├── model.py                   # Policy network
│   └── train.py                   # REINFORCE training script
│
└── example-tabular-ml/            # Example 7: Tabular ML
    ├── README.md                  # Tabular ML documentation
    ├── tabular_dataset.py         # Synthetic data generation
    ├── tabular_model.py           # MLP architecture
    ├── train_tabular.py           # Training script
    └── predict_tabular.py         # Inference script
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (shared by all examples)
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

### Example 3: Semantic Segmentation

```bash
cd example-semantic-segmentation

# Create toy dataset
python create_toy_data.py --output-dir toy_data --num-images 40

# Train the model
python train.py --data-root toy_data --epochs 1

# Run inference
python predict.py --weights segmenter.pth --image toy_data/images/img_0000.png
```

**Result:** Produces a pixel-wise mask overlay for synthetic shapes.

📖 **[Read the Semantic Segmentation README](example-semantic-segmentation/README.md)** for detailed instructions.

### Example 4: Diffusion Model (MNIST)

```bash
cd example-diffusion-model

# Train the model
python train.py --epochs 5 --batch-size 128
```

**Result:** Learns to predict noise on MNIST digits for a DDPM-style setup.

📖 **[Read the Diffusion README](example-diffusion-model/README.md)** for detailed instructions.

### Example 5: Reinforcement Learning (Bandit)

```bash
cd example-reinforcement-learning

# Train policy with REINFORCE
python train.py --episodes 3000
```

**Result:** Learns to favor the highest-reward arm through trial-and-error.

📖 **[Read the RL README](example-reinforcement-learning/README.md)** for detailed instructions.

### Example 6: Tabular ML (Binary Classification)

```bash
cd example-tabular-ml

# Train the MLP classifier
python train_tabular.py

# Run inference
python predict_tabular.py
```

**Result:** Learns a decision boundary on synthetic numerical features.

📖 **[Read the Tabular ML README](example-tabular-ml/README.md)** for detailed instructions.

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

### Try Multiple! 🎓
They teach complementary concepts:
- **LLM**: Sequence modeling, attention, autoregressive generation
- **Image Classifier**: Spatial features, data augmentation, classification metrics
- **Diffusion**: Generative modeling with noise schedules
- **Reinforcement Learning**: Exploration, policy gradients, reward optimization
- **Tabular ML**: Feature standardization, MLPs for structured data

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
- **[example-object-detection/README.md](example-object-detection/README.md)** - Guide for the object detection scaffold
- **[example-semantic-segmentation/README.md](example-semantic-segmentation/README.md)** - Guide for the semantic segmentation scaffold
- **[example-diffusion-model/README.md](example-diffusion-model/README.md)** - Complete guide for the diffusion model
- **[example-reinforcement-learning/README.md](example-reinforcement-learning/README.md)** - Complete guide for the RL bandit example
- **[example-tabular-ml/README.md](example-tabular-ml/README.md)** - Complete guide for the tabular ML example

Additional documentation:

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step getting started guide
- **[COMPARISON.md](COMPARISON.md)** - Detailed comparison of the LLM and image classifier
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
- All examples work on CPU (just slower)
- Training is 5-10× faster on GPU

---

## 🎓 Learning Path

### Week 1: Get Started
1. Install dependencies
2. Run the LLM and image classifier examples with default settings
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

After completing the LLM and image classifier examples:

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


Start with any example, experiment, and most importantly - have fun building with PyTorch!

## Next Steps to Explore
- **Hyperparameters**: Try changing `n_layer`, `n_head`, or `n_embd` in `model.py` and see how it affects training speed and generated text quality.
- **Dataset**: Swap out `tinyshakespeare.txt` for your own text file to train the model on different styles of writing.
- **Tokenization**: Instead of character-level, try implementing a sub-word tokenizer like BPE (Byte Pair Encoding).

## Example: Load a Model and Dataset from Hugging Face
If you want to experiment with pretrained models and hosted datasets, the Hugging Face Hub makes it easy to pull both:

```bash
pip install transformers datasets
```

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sample = dataset[0]["text"]
inputs = tokenizer(sample, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This example downloads the Wikitext-2 dataset and GPT-2 model, then generates text from the first dataset sample.
