import torch
import torch.nn as nn
from model import Transformer
from model_utils import get_device, load_checkpoint_payload
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load("data/tokenizer.json")

# Load the tensors we got from prepare_finetune_data
inputs_tensor = torch.load("data/finetune_inputs.pt", weights_only=True)
targets_tensor = torch.load("data/finetune_targets.pt", weights_only=True)

vocab_size = len(tokenizer.vocab)
batch_size = 16
learning_rate = .000007
max_steps = 1200

device = get_device()

checkpoint_payload, checkpoint_state_dict, checkpoint_config = load_checkpoint_payload("data/model.pth", device)

if checkpoint_config["vocab_size"] != vocab_size:
    raise ValueError(
        f"Tokenizer vocab size ({vocab_size}) does not match the base model checkpoint ({checkpoint_config['vocab_size']})."
    )

model = Transformer(
    checkpoint_config["vocab_size"],
    checkpoint_config["embedding_dimension"],
    checkpoint_config["context_length"],
    checkpoint_config["num_blocks"],
    checkpoint_config["num_heads"],
)

# Load in the pre trained weights to our created model
model.load_state_dict(checkpoint_state_dict)

model = model.to(device)

# Freeze embeddings and first 4 blocks to prevent catastrophic forgetting
for param in model.token_embedding.parameters():
    param.requires_grad = False
for param in model.position_embedding.parameters():
    param.requires_grad = False
for i, block in enumerate(model.blocks):
    if i < 4:
        for param in block.parameters():
            param.requires_grad = False

# Only pass trainable parameters to the optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

for step in range(max_steps):
    
    # Randomly sample a batch of 16 from our examples
    indices = torch.randint(0, len(inputs_tensor), (batch_size,))
    chunks = inputs_tensor[indices].to(device)
    targets = targets_tensor[indices].to(device)
    
    # Get the models predictions
    predictions = model(chunks)
    
    # Calculate loss on the answer tokens
    loss = torch.nn.functional.cross_entropy(
        predictions.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100
    )
    
    # Remove the old gradients
    optimizer.zero_grad()
    
    # Get the new ones and update the weights
    loss.backward()
    
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    if step % 250 == 0:
        print(f"Step {step} Loss: {loss.item():.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "config": checkpoint_config,
}, "data/model_finetuned.pth")
print("Fine-tuned model saved!")
