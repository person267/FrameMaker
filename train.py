import torch
from model import FrameMakerModel, BLOCK_SIZE, BATCH_SIZE

# 1. Dummy training data (Replace this with a text file of your own writing!)
text = "FrameMaker is a chaotic AI. FrameMaker makes images and text. Let's build a custom model."
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Simple character-level tokenizer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert text data to a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]

def get_batch():
    """Generates a small batch of inputs (x) and targets (y) for training"""
    ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([train_data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([train_data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

# 2. Initialize model and optimizer
model = FrameMakerModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training FrameMaker...")
# 3. Quick training loop
model.train()
for steps in range(300): # Train for 300 quick steps
    xb, yb = get_batch()
    
    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training complete! Final Loss: {loss.item():.4f}")

# 4. Test generation
context = torch.zeros((1, 1), dtype=torch.long)
generated_indices = model.generate(context, max_new_tokens=50)[0].tolist()
print("\nGenerated Output from FrameMaker:")
print(decode(generated_indices))
