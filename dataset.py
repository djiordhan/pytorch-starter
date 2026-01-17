import torch
import os

def load_data(file_path):
    """
    Loads text from a file and creates a simple character-level tokenizer.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings from characters to integers and vice versa
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    encode = lambda s: [stoi[c] for c in s]          # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Encode the entire dataset into a torch tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    
    return data, vocab_size, encode, decode

def get_batch(data, block_size, batch_size, device):
    """
    Generate a small batch of data of inputs x and targets y.
    x is a sequence of characters, y is the same sequence shifted by one character.
    """
    # Pick random starting points for our sequences
    idx = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract sequences (x) and their corresponding targets (y)
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    
    # Move batch to the specified device (CPU or GPU)
    x, y = x.to(device), y.to(device)
    return x, y
