# Quick Start Guide

## 🚀 Getting Started

### Installation

```bash
# Navigate to the project directory
cd pytorch-starter

# Install all dependencies (shared by all examples)
pip install -r requirements.txt
```

---

## 📝 Example 1: Text Generation (LLM)

### Navigate to the example

```bash
cd example-llm
```

### Step-by-step

1. **Download the dataset:**
   ```bash
   python prepare_data.py
   ```
   This downloads the Tiny Shakespeare dataset (~1MB) to `data/tinyshakespeare.txt`.

2. **Train the model:**
   ```bash
   python train.py
   ```
   - Training takes ~5-10 minutes on CPU
   - You'll see loss decreasing every 300 steps
   - At the end, it generates Shakespeare-like text
   - Model saved to `model.pth`

3. **What to expect:**
   ```
   Step 0: Train loss 4.2345, Val loss 4.2567
   Step 300: Train loss 2.1234, Val loss 2.3456
   ...
   --- Training Complete! ---
   Generating sample text:
   ROMEO: What say you, my lord?
   ...
   ```

### Files to explore
- `model.py` - See how a Transformer is built
- `train.py` - Understand the training loop
- `dataset.py` - Learn about tokenization
- `README.md` - Detailed documentation

---

## 🖼️ Example 2: Image Classification (CIFAR-10)

### Navigate to the example

```bash
cd example-image-classifier
```

### Step-by-step

1. **Train the classifier:**
   ```bash
   python train_image.py
   ```
   - Downloads CIFAR-10 automatically (~170MB) to `data/`
   - Training takes ~20-30 minutes on CPU, ~5 minutes on GPU
   - Saves the best model to `image_classifier.pth`

2. **Test the model:**
   ```bash
   # Test on random samples from CIFAR-10
   python predict_image.py
   ```

3. **Classify your own image:**
   ```bash
   python predict_image.py path/to/your/image.jpg
   ```
   Note: Works best with images containing objects from CIFAR-10 classes:
   - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Expected Results

**SimpleCNN (default):**
- Training accuracy: ~70-75%
- Validation accuracy: ~65-70%
- Test accuracy: ~65-70%

**ResNetCIFAR (advanced):**
- Training accuracy: ~85-90%
- Validation accuracy: ~75-80%
- Test accuracy: ~75-80%

To use ResNet, edit `train_image.py` and change:
```python
MODEL_TYPE = 'resnet'  # instead of 'simple'
```

### Files to explore
- `image_model.py` - Two CNN architectures
- `train_image.py` - Complete training pipeline
- `image_dataset.py` - Data loading and augmentation
- `predict_image.py` - Inference and prediction
- `README.md` - Detailed documentation

---

## 🧩 Example 3: Semantic Segmentation

```bash
cd example-semantic-segmentation

# Create toy data
python create_toy_data.py --output-dir toy_data --num-images 40

# Train the model
python train.py --data-root toy_data --epochs 1

# Run inference
python predict.py --weights segmenter.pth --image toy_data/images/img_0000.png
```

**Result:** Produces a pixel-wise mask overlay for synthetic shapes.

---

## ✨ Example 4: Diffusion Model (MNIST)

### Navigate to the example

```bash
cd example-diffusion-model
```

### Step-by-step

1. **Train the diffusion model:**
   ```bash
   python train.py --epochs 5 --batch-size 128
   ```
   - Downloads MNIST automatically (~60MB)
   - Trains a noise-prediction network on diffusion timesteps
   - Saves `diffusion_mnist.pth` when finished

2. **What to expect:**
   ```
   Epoch 1: average loss 0.9732
   Epoch 2: average loss 0.8127
   ...
   Saved model to diffusion_mnist.pth
   ```

### Files to explore
- `model.py` - Timestep-conditioned noise predictor
- `dataset.py` - MNIST data loader
- `train.py` - Diffusion training loop
- `README.md` - Detailed documentation

---

## 🎮 Example 5: Reinforcement Learning (Bandit)

### Navigate to the example

```bash
cd example-reinforcement-learning
```

### Step-by-step

1. **Train the RL policy:**
   ```bash
   python train.py --episodes 3000 --arm-probs 0.15 0.4 0.6 0.8
   ```
   - Trains with REINFORCE on a Bernoulli multi-armed bandit
   - Prints moving-average reward and current greedy arm
   - Converges toward the best arm in most runs

2. **What to expect:**
   ```
   Episode   200 | avg_reward(200)=0.515 | greedy_arm=2
   ...
   Learned best arm: 3
   True best arm:    3
   ```

### Files to explore
- `environment.py` - Minimal multi-armed bandit environment
- `model.py` - Categorical policy
- `train.py` - Policy-gradient training loop
- `README.md` - Detailed documentation

---

## 📈 Example 6: Tabular ML (Binary Classification)

### Navigate to the example

```bash
cd example-tabular-ml
```

### Step-by-step

1. **Train the MLP classifier:**
   ```bash
   python train_tabular.py
   ```
   - Builds a synthetic dataset with 12 numerical features
   - Standardizes the features using training statistics
   - Saves the best checkpoint to `tabular_classifier.pth`

2. **Run inference:**
   ```bash
   python predict_tabular.py
   ```
   - Uses a random synthetic feature vector

### Files to explore
- `tabular_dataset.py` - Synthetic data generation + dataloaders
- `tabular_model.py` - MLP architecture
- `train_tabular.py` - Training loop + checkpointing
- `predict_tabular.py` - Inference script
- `README.md` - Detailed documentation

---

## 🎯 Tips for Success

### For LLM Example:
- Start with the default hyperparameters
- Watch how the loss decreases - it should go from ~4.0 to ~1.5
- The generated text will be gibberish at first, but improves with training
- Try training for more iterations (change `max_iters` in `train.py`)

### For Image Classifier:
- GPU is recommended but not required
- First epoch is slower (downloading data)
- Validation accuracy should improve each epoch
- If accuracy plateaus, try:
  - Increasing epochs
  - Adjusting learning rate
  - Switching to ResNet architecture

---

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
# From project root
pip install -r requirements.txt
```

### "Dataset not found" (LLM)
```bash
cd example-llm
python prepare_data.py
```

### "CUDA out of memory" (Image Classifier)
Reduce batch size in `train_image.py`:
```python
BATCH_SIZE = 64  # or 32
```

### Slow training
- LLM: Reduce `max_iters` or `n_layer` in respective files
- Image: Reduce `EPOCHS` or `BATCH_SIZE`
- Consider using Google Colab for free GPU access

### For Diffusion Model:
- Increase `--timesteps` for higher-quality samples
- Add a sampling script to visualize generated digits
- Try Fashion-MNIST for a slightly harder dataset

### For Reinforcement Learning:
- Use more episodes if the policy is unstable
- Lower `--learning-rate` for smoother convergence
- Try harder settings by making arm probabilities close

### Import errors after reorganization
Make sure you're in the correct directory:
```bash
# For LLM
cd example-llm
python train.py

# For Image Classifier
cd example-image-classifier
python train_image.py
```

---

## 📊 Monitoring Training

### LLM
- **Good sign**: Loss decreases from ~4.0 to ~1.5
- **Bad sign**: Loss stays above 3.0 or increases
- **Generated text quality**: Should improve from random characters to word-like patterns

### Image Classifier
- **Good sign**: 
  - Training accuracy increases each epoch
  - Validation accuracy follows training (with a small gap)
- **Bad sign**:
  - Validation accuracy much lower than training (overfitting)
  - Both accuracies stuck below 40%
- **Target**: 65-70% test accuracy with SimpleCNN

---

## 🎓 Learning Path

### Beginner (Week 1)
1. Run the LLM and image classifier examples with default settings
2. Read through the comments in each file
3. Experiment with one hyperparameter at a time
4. Read the example-specific READMEs

### Intermediate (Week 2)
1. Modify the SimpleCNN architecture (add layers)
2. Try different optimizers (SGD, AdamW)
3. Implement a custom dataset for the image classifier
4. Train the LLM on your own text

### Advanced (Week 3+)
1. Combine both: Build an image captioning model
2. Implement transfer learning with pre-trained models
3. Add TensorBoard logging for visualization
4. Deploy your model as a web API

---

## 📚 Next Projects

After mastering these examples:
1. **Object Detection**: YOLO or Faster R-CNN
2. **Semantic Segmentation**: U-Net or DeepLab
3. **Generative Models**: VAE, GAN, or diffusion
4. **Advanced NLP**: BERT fine-tuning or GPT-2
5. **Reinforcement Learning**: DQN or PPO

---

## 💡 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CS231n Course](http://cs231n.stanford.edu/)
- [Fast.ai Course](https://course.fast.ai/)

---

## 📁 Project Structure

```
pytorch-starter/
├── requirements.txt           # Shared dependencies
├── README.md                  # Main documentation
├── QUICKSTART.md             # This file
├── COMPARISON.md             # Detailed comparison
│
├── example-llm/              # Language Model Example
│   ├── README.md
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── prepare_data.py
│   └── data/                 # LLM data directory
│
├── example-image-classifier/ # Image Classifier Example
│   ├── README.md
│   ├── image_model.py
│   ├── image_dataset.py
│   ├── train_image.py
│   ├── predict_image.py
│   └── data/                 # Image data directory
│
├── example-object-detection/ # Object Detection Example
│   ├── README.md
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── requirements.txt
│
├── example-semantic-segmentation/ # Semantic Segmentation Example
│   ├── README.md
│   ├── create_toy_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
│
├── example-diffusion-model/  # Diffusion Model Example
│   ├── README.md
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── example-reinforcement-learning/ # Reinforcement Learning Example
│   ├── README.md
│   ├── environment.py
│   ├── model.py
│   └── train.py
│
└── example-tabular-ml/        # Tabular ML Example
    ├── README.md
    ├── tabular_dataset.py
    ├── tabular_model.py
    ├── train_tabular.py
    └── predict_tabular.py
```

---

Happy Learning! 🚀
