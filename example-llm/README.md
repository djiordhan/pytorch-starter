# Character-Level Language Model (LLM)

A beginner-friendly PyTorch implementation of a Transformer-based language model inspired by GPT. This example trains a character-level model on Shakespeare's works to generate similar text.

---

## 📚 What You'll Learn

- **Transformer Architecture**: Self-attention, positional embeddings, and residual connections
- **Sequence Modeling**: How to handle sequential data in PyTorch
- **Text Generation**: Autoregressive sampling from probability distributions
- **PyTorch Fundamentals**: `nn.Module`, embeddings, training loops, and model saving

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
python prepare_data.py
```

This downloads the Tiny Shakespeare dataset (~1MB) to the `data/` directory.

### 3. Train the Model

```bash
python train.py
```

**What to expect:**
- Training takes ~5-10 minutes on CPU, ~2-3 minutes on GPU
- Loss should decrease from ~4.0 to ~1.5
- At the end, the model generates Shakespeare-like text
- Model saved to `model.pth`

---

## 📁 Files

| File | Purpose | Lines |
|------|---------|-------|
| `model.py` | Transformer architecture | 155 |
| `dataset.py` | Text data loading and tokenization | 60 |
| `train.py` | Training script | 85 |
| `prepare_data.py` | Dataset download script | 25 |

---

## 🏗️ Architecture

```
Input Text (characters)
    ↓
Character Embedding (vocab_size → n_embd)
    +
Positional Embedding (position → n_embd)
    ↓
Transformer Blocks (×4)
├─ Multi-Head Self-Attention (4 heads)
├─ Layer Normalization
├─ Feed-Forward Network (MLP)
└─ Residual Connections
    ↓
Layer Normalization
    ↓
Output Layer (n_embd → vocab_size)
    ↓
Next Character Prediction
```

---

## ⚙️ Hyperparameters

```python
n_embd = 128      # Embedding dimension
n_head = 4        # Number of attention heads
n_layer = 4       # Number of transformer blocks
dropout = 0.1     # Dropout rate
block_size = 64   # Maximum context length
batch_size = 32   # Batch size
learning_rate = 1e-3  # Learning rate
max_iters = 3000  # Training iterations
```

---

## 📊 Expected Results

**Training Progress:**
```
Step 0: Train loss 4.2345, Val loss 4.2567
Step 300: Train loss 2.1234, Val loss 2.3456
Step 600: Train loss 1.8123, Val loss 2.0456
...
Step 2700: Train loss 1.4567, Val loss 1.6789
```

**Generated Text (after training):**
```
ROMEO: What say you, my lord?

DUKE OF YORK:
I'll tell thee what, my lord, I will not be
The king of England, and the world shall see
That I am not a man of such a thing.
```

The text won't be perfect, but you'll see word-like patterns and some grammatical structure!

---

## 🔧 Customization

### Change Model Size

Edit `model.py`:
```python
n_embd = 256      # Larger embedding (better quality, slower)
n_layer = 6       # More layers (deeper model)
n_head = 8        # More attention heads
```

### Train Longer

Edit `train.py`:
```python
max_iters = 5000  # More training steps
```

### Use Your Own Text

Replace `data/tinyshakespeare.txt` with your own text file, then:
```bash
python train.py
```

### Adjust Context Length

Edit `model.py`:
```python
block_size = 128  # Longer context (more memory)
```

---

## 🎓 Learning Guide

### Step 1: Understand the Data (dataset.py)

Start here to see how text is converted to numbers:
- Character-to-integer mapping
- Encoding and decoding functions
- Batch generation

**Key Concept**: Neural networks work with numbers, not text. We map each character to a unique integer.

### Step 2: Study the Model (model.py)

Read through the architecture from bottom to top:

1. **Head**: Single attention head - learns to focus on relevant characters
2. **MultiHeadAttention**: Multiple heads working in parallel
3. **FeedForward**: Simple MLP for processing
4. **Block**: Complete transformer block (attention + feedforward)
5. **SimpleLanguageModel**: Full model with embeddings and output

**Key Concept**: Self-attention allows the model to look at all previous characters when predicting the next one.

### Step 3: Understand Training (train.py)

Follow the training loop:
1. Sample a batch of text
2. Forward pass (predict next characters)
3. Compute loss (how wrong were we?)
4. Backward pass (compute gradients)
5. Update weights (learn from mistakes)

**Key Concept**: The model learns by repeatedly predicting the next character and adjusting its weights based on errors.

---

## 🧪 Experiments to Try

### Beginner
1. ✅ Run with default settings
2. ✅ Change `max_iters` to 5000
3. ✅ Try different learning rates (1e-4, 5e-4, 1e-3)

### Intermediate
1. 📈 Increase `n_embd` to 256
2. 📈 Add more layers (`n_layer = 6`)
3. 📈 Train on your own text (poetry, code, etc.)

### Advanced
1. 🚀 Implement temperature sampling in `generate()`
2. 🚀 Add top-k or nucleus sampling
3. 🚀 Implement BPE tokenization instead of character-level
4. 🚀 Add learning rate scheduling

---

## 🐛 Troubleshooting

### "Dataset not found"
```bash
python prepare_data.py
```

### Loss not decreasing
- Train longer (`max_iters = 5000`)
- Reduce learning rate (`learning_rate = 5e-4`)
- Check that data loaded correctly

### Out of memory
- Reduce `batch_size` to 16
- Reduce `n_embd` to 64
- Reduce `block_size` to 32

### Generated text is gibberish
- Train longer (3000 iterations is minimal)
- Increase model size (`n_embd = 256`, `n_layer = 6`)
- Check that loss is decreasing

---

## 📖 Understanding the Code

### Key PyTorch Concepts

**nn.Module**: Base class for all models
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Define computation
        return output
```

**nn.Embedding**: Converts integers to vectors
```python
self.token_embedding = nn.Embedding(vocab_size, n_embd)
# Input: [batch, seq_len] of integers
# Output: [batch, seq_len, n_embd] of floats
```

**Self-Attention**: Learns relationships between positions
```python
# Query: "What am I looking for?"
# Key: "What do I contain?"
# Value: "What do I actually communicate?"
scores = query @ key.T  # Compute affinities
weights = softmax(scores)  # Normalize
output = weights @ value  # Weighted sum
```

---

## 🎯 Next Steps

After mastering this example:

1. **Implement word-level tokenization** instead of character-level
2. **Add temperature parameter** to control randomness in generation
3. **Implement beam search** for better text generation
4. **Try a larger dataset** (e.g., all of Wikipedia)
5. **Fine-tune a pre-trained model** using Hugging Face Transformers

---

## 📚 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) - Minimal GPT implementation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Even simpler GPT

---

## 💡 Tips

- **Start small**: Run with default settings first
- **Read comments**: Every line is explained
- **Experiment**: Change one thing at a time
- **Visualize**: Print tensor shapes to understand data flow
- **Be patient**: Training takes time, but results are rewarding!

---

Happy Learning! 🚀
