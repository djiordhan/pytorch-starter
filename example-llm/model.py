import torch
import torch.nn as nn
from torch.nn import functional as F

# -- Hyperparameters --
# These control the size and complexity of our model
# Small values here so it runs quickly on a CPU
n_embd = 128      # Embedding dimension (size of the vector representation for each character)
n_head = 4        # Number of attention heads
n_layer = 4       # Number of transformer blocks
dropout = 0.1     # Dropout rate to prevent overfitting
block_size = 64   # Maximum context length (how many characters the model looks back at)

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # Key, Query, and Value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 'tril' is used for masking future tokens (causal self-attention)
        # It ensures that position i only attends to positions <= i
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # Batch size, Time (sequence length), Channels (embedding dim)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # Dot product of query and key, scaled by the square root of dimension
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        
        # Mask out future tokens: prevents the model from "cheating" by looking ahead
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        # Softmax turns scores into probabilities that sum to 1
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # Apply the attention weights to the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Linear projection back to embedding dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to original dimension and apply dropout
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity (ReLU) """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Self-attention
        self.ffwd = FeedForward(n_embd)                # Feed-forward
        self.ln1 = nn.LayerNorm(n_embd)                 # Normalization 1
        self.ln2 = nn.LayerNorm(n_embd)                 # Normalization 2

    def forward(self, x):
        # We use residual connections (x + ...) to help gradients flow during training
        # LayerNorm is applied before the transformation (Pre-LN architecture)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLanguageModel(nn.Module):
    """ A character-level Transformer LLM (GPT-style) """

    def __init__(self, vocab_size):
        super().__init__()
        # Each character gets a learned vector representation (Token Embedding)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each position in the sequence gets a learned vector (Positional Embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Sequence of Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # Final layer normalization and linear layer to predict next token
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape tensors for PyTorch's cross_entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generate new characters given a starting context 'idx' """
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
