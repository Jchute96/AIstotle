import os
import torch 
import torch.nn as nn
from model import Transformer
from tokenizer import BPETokenizer

def get_batch(token_ids, batch_size, context_length):
    
    
    target_list = []
    chunk_list = []
    
    # Get random starting positions out of the token_ids so model learns patterns more efficiently
    starting_positions = torch.randint(0, len(token_ids) - context_length - 1, (batch_size,))
    
    # Get the chunk of tokens based off the starting positions and the target tokens for training that are one position after
    for starting_index in starting_positions:
        chunk_list.append(token_ids[starting_index:starting_index+context_length])
        target_list.append(token_ids[starting_index+1:starting_index+context_length+1])
        
    # Combine the chunks of tokens into one tensor  
    chunks = torch.stack(chunk_list)
    
    # Combine the targets of tokens into one tensor  
    targets = torch.stack(target_list)
    
    return (chunks, targets)


tokenizer = BPETokenizer()

# Load the vocab and merge rules for our dataset into our tokenizer
tokenizer.load("data/tokenizer.json")

# Get the dataset from the file
with open("data/dataset.txt", 'r') as file:
    dataset = file.read()

# Encode dataset or load if already encoded
if os.path.exists("data/token_ids.pt"):
    
    # Load pre encoded token ids
    token_ids = torch.load("data/token_ids.pt")
    print("Token ids loaded!")
    
else:
   
    # Convert the ids from the dataset to int ids
    token_ids = tokenizer.encode(dataset)
    
    # Convert the ids to a tensor
    token_ids = torch.tensor(token_ids)
    
    # Save for next use
    torch.save(token_ids, "data/token_ids.pt")
    print("Token ids encoded and saved!")

vocab_size = 5000
embedding_dimension = 256
context_length = 256
num_blocks = 4
num_heads = 4
batch_size = 16
# Controls the adjustment size in model numbers during training
learning_rate = .0003
# Controls how many times the training loop runs
max_steps = 20000
device = ""

# Try to use GPU otherwise use CPU 
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = Transformer(vocab_size, embedding_dimension, context_length, num_blocks, num_heads)

# Move to the device
model = model.to(device)

# Handles adjusting the numbers when the model makes a prediciton and calculates its loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

token_ids = token_ids.to(device)

# Perform the training loop
for step in range(max_steps):
    
    # Get a batch of chunks and targets
    chunks, targets = get_batch(token_ids, batch_size, context_length)
    
    # Pass the chunks to our model which then uses forward chaining to pass the data through the layers in the model to get how likely each token will follow the preceding token
    predictions = model(chunks)
    
    # Calculate the loss of our predictions to see how accurate they were when compared to our targets
    loss = torch.nn.functional.cross_entropy(predictions.view(-1, vocab_size), targets.view(-1))
    
    # Clear previous optimization gradients
    optimizer.zero_grad()
    
    # Calculate the new gradient changes that need to be made for this iteration
    loss.backward()
    
    # Actually make those changes to the numbers
    optimizer.step()
    
    # Print average loss every 500 steps to make sure number is decreasing over time
    if step % 500 == 0:
        print(f"Step {step} Loss: {loss.item():.4f}")
    
# Save the model
torch.save(model.state_dict(), "data/model.pth")
print("Model has been saved!")
    