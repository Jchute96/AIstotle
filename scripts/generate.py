import torch
from model import Transformer
from tokenizer import BPETokenizer
    
tokenizer = BPETokenizer()

# Load the vocab and merge rules for our dataset into our tokenizer
tokenizer.load("data/tokenizer.json")

vocab_size = 5000
embedding_dimension = 256
context_length = 256
num_blocks = 4
num_heads = 4
device = ""

# Try to use GPU otherwise use CPU 
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = Transformer(vocab_size, embedding_dimension, context_length, num_blocks, num_heads)

# Move to the device
model = model.to(device) 

# Load the model weights
model.load_state_dict(torch.load("data/model.pth"))

# Tell pytorch we are doing inference instead of training
model.eval()


def generate(prompt, max_new_tokens=100, temperature=0.1):
    
    # Encode the prompt into int ids
    token_ids = tokenizer.encode(prompt)
    
    # Convert it to a tensor and use unsqueeze to add a dimension for the batch at position 0
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    
    # Generate new tokens for the response
    for token in range(max_new_tokens):
        
        # Make sure we only give the model our correct context length
        context = token_ids[:,-context_length:]
        
        # Get the predictions
        predictions = model(context)
        
        # Get all the scores for the last token and then apply the temperature to them to control randomness
        last_token_predictions = predictions[:, -1, :] / temperature
        
        # Convert to probabilities
        probabilities = torch.softmax(last_token_predictions, dim=-1)
        
        # Get the next token based off of its probabilities
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        # Add the token to the sequence
        token_ids = torch.cat([token_ids, next_token], dim=1)
        
    # Decode and return the newly created response
    return tokenizer.decode(token_ids[0].tolist())
