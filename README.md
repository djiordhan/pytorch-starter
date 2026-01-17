# PyTorch Starter: Simple LLM

This project is a beginner-friendly introduction to PyTorch and Transformers. It implements a character-level Language Model (LLM) inspired by the GPT architecture, trained on the works of Shakespeare.

## Project Structure

- `model.py`: Contains the `SimpleLanguageModel` architecture (Transformer blocks, multi-head attention, etc.).
- `dataset.py`: Handles data loading, tokenization (char-level), and batching.
- `train.py`: The main script to train the model and generate text.
- `prepare_data.py`: A helper script to download the Tiny Shakespeare dataset.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data
Download the Tiny Shakespeare dataset by running:
```bash
python prepare_data.py
```

### 3. Train the Model
Run the training script. It will train for 3000 iterations and then generate a sample of Shakespeare-like text:
```bash
python train.py
```

## Learning Guide

The code is heavily commented to explain what each part does:
- **`model.py`**: Start here to understand how a Transformer is built from scratch using PyTorch `nn.Module`. You'll learn about embeddings, self-attention, and residual connections.
- **`train.py`**: Look here to see the standard PyTorch training loop: Forward pass -> Loss calculation -> Backward pass -> Optimizer step.
- **`dataset.py`**: Learn how raw text is converted into numbers (tensors) that the model can understand.

## Next Steps to Explore
- **Hyperparameters**: Try changing `n_layer`, `n_head`, or `n_embd` in `model.py` and see how it affects training speed and generated text quality.
- **Dataset**: Swap out `tinyshakespeare.txt` for your own text file to train the model on different styles of writing.
- **Tokenization**: Instead of character-level, try implementing a sub-word tokenizer like BPE (Byte Pair Encoding).
