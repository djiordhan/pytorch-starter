# Tabular ML (Binary Classification)

A beginner-friendly PyTorch example that trains a multilayer perceptron (MLP) on a **synthetic tabular dataset**. This example shows how to work with numerical features, standardize them, and run inference on new samples.

---

## 📊 What You'll Learn

- **Tabular data workflows**: Generating and standardizing numerical features
- **MLP models**: Building a simple feed-forward network in PyTorch
- **Binary classification**: Using `BCEWithLogitsLoss`
- **Model checkpointing**: Saving feature statistics with your model
- **Inference pipeline**: Running predictions on new feature vectors

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_tabular.py
```

**What to expect:**
- A synthetic dataset with 12 numerical features
- Standardized features (mean 0, std 1)
- Progress bars for each epoch
- Best model saved to `tabular_classifier.pth`

### 3. Run Predictions

```bash
# Predict with a random synthetic sample
python predict_tabular.py

# Provide your own 12 features
python predict_tabular.py --features "0.2, -1.1, 0.7, 0.3, -0.5, 0.9, 1.2, -0.4, 0.1, -0.8, 0.6, -0.2"
```

---

## 📁 Files

| File | Purpose | Lines |
|------|---------|-------|
| `tabular_dataset.py` | Synthetic data generation + dataloaders | ~120 |
| `tabular_model.py` | MLP architecture for tabular data | ~30 |
| `train_tabular.py` | Training loop + checkpointing | ~150 |
| `predict_tabular.py` | Inference on new samples | ~90 |

---

## 🧠 Model Architecture

```
Input (12 features)
    ↓
Linear(12 → 64) + ReLU + Dropout
    ↓
Linear(64 → 32) + ReLU + Dropout
    ↓
Linear(32 → 1)
    ↓
Sigmoid → Probability
```

---

## ⚙️ Configuration

Edit `train_tabular.py` to change:

```python
batch_size = 128
epochs = 20
learning_rate = 3e-4
hidden_sizes = (64, 32)
n_samples = 12000
n_features = 12
```

---

## ✅ Expected Output

```
Epoch 10 | Train Loss: 0.3621 Acc: 0.842 | Val Loss: 0.3584 Acc: 0.846
  ✓ Saved new best model to tabular_classifier.pth
...
Test Loss: 0.3512 | Test Acc: 0.851
```

---

## 🧪 Experiments to Try

1. **Change dataset size**: `n_samples = 20000`
2. **Add more layers**: Update `hidden_sizes = (128, 64, 32)`
3. **Increase noise**: Set `noise_scale = 1.0` in `create_dataloaders`
4. **Try a smaller model**: `hidden_sizes = (32, 16)`

---

Happy experimenting! 🚀
