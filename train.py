import torch
from model import SimpleLanguageModel
from dataset import load_data, get_batch
import os
from tqdm import tqdm

# -- Configuration --
# These parameters define the training process
batch_size = 32      # How many independent sequences will we process in parallel?
max_iters = 3000     # Total training steps
eval_interval = 300  # How often to check loss on validation data
learning_rate = 1e-3 # Step size for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
eval_iters = 200     # Number of batches to use when estimating loss
data_path = 'data/tinyshakespeare.txt'

# Step 1: Check if data exists
if not os.path.exists(data_path):
    print("Dataset not found! Please run 'python prepare_data.py' first.")
    exit()

# Step 2: Load and prepare data
print(f"Loading data from {data_path}...")
data, vocab_size, encode, decode = load_data(data_path)

# Let's split it into training and validation sets
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# Step 3: Initialize the model
print(f"Initializing model on {device}...")
model = SimpleLanguageModel(vocab_size)
model = model.to(device)

# Step 4: Setup the optimizer (AdamW is a common choice for Transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    """ Runs a quick evaluation on train and val sets to track progress """
    out = {}
    model.eval() # Set model to evaluation mode (e.g., disables dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data_split = train_data if split == 'train' else val_data
            X, Y = get_batch(data_split, model.blocks[0].sa.heads[0].tril.shape[0], batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

# Step 5: The Training Loop
print("Starting training...")
for iter in range(max_iters):

    # Every so often, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch(train_data, 64, batch_size, device)

    # Forward pass: evaluate the loss
    logits, loss = model(xb, yb)
    
    # Backward pass: compute gradients and update weights
    optimizer.zero_grad(set_to_none=True) # Reset gradients from previous step
    loss.backward()                       # Compute gradients of the loss w.r.t weights
    optimizer.step()                      # Update weights based on gradients

# Step 6: Generate Sample Text
print("\n--- Training Complete! ---")
print("Generating sample text from the trained model:")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with character 0
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_indices))

# Step 7: Save the model
torch.save(model.state_dict(), 'model.pth')
print("\nModel saved to model.pth")
