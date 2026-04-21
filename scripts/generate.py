import os
import torch
from model import Transformer
from model_utils import get_device, load_checkpoint_payload
from tokenizer import BPETokenizer
    
tokenizer = BPETokenizer()

# Load the vocab and merge rules for our dataset into our tokenizer
tokenizer.load("data/tokenizer.json")

vocab_size = len(tokenizer.vocab)
device = get_device()
checkpoint_path = "data/model_finetuned.pth" if os.path.exists("data/model_finetuned.pth") else "data/model.pth"
_, checkpoint_state_dict, checkpoint_config = load_checkpoint_payload(checkpoint_path, device)

if checkpoint_config["vocab_size"] != vocab_size:
    raise ValueError(
        f"Tokenizer vocab size ({vocab_size}) does not match the checkpoint ({checkpoint_config['vocab_size']})."
    )

model = Transformer(
    checkpoint_config["vocab_size"],
    checkpoint_config["embedding_dimension"],
    checkpoint_config["context_length"],
    checkpoint_config["num_blocks"],
    checkpoint_config["num_heads"],
)

# Move to the device
model = model.to(device) 

# Load the model weights
model.load_state_dict(checkpoint_state_dict)

context_length = checkpoint_config["context_length"]

# Tell pytorch we are doing inference instead of training
model.eval()


def generate(prompt, max_new_tokens=100, temperature=0.1):
    
    # Encode the prompt into int ids
    token_ids = tokenizer.encode(prompt)
    
    # Convert it to a tensor and use unsqueeze to add a dimension for the batch at position 0
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    
    # Make it so that gradients are not tracked during generation since they are not needed
    with torch.no_grad():
        # Generate new tokens for the response
        for token in range(max_new_tokens):
            
            # Make sure we only give the model our correct context length
            context = token_ids[:,-context_length:]
        
            # Get the predictions
            predictions = model(context)
        
            # Get all the scores for the last token and then apply the temperature to them to control randomness
            last_token_predictions = predictions[:, -1, :] / temperature

            # Keep only the top 50 tokens to prevent repetition loops
            top_values, _ = torch.topk(last_token_predictions, 50)
            last_token_predictions[last_token_predictions < top_values[:, -1:]] = float('-inf')

            # Convert to probabilities
            probabilities = torch.softmax(last_token_predictions, dim=-1)

            # Get the next token based off of its probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)
        
            # Add the token to the sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)
        
    # Decode and return the newly created response
    return tokenizer.decode(token_ids[0].tolist())

print(generate("[QUESTION] What is the secret to life? [ANSWER]", temperature=0.7))

