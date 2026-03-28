import torch
import torch.nn as nn

# Inherits from Pytorch's nn module that handles neural networks
class Transformer(nn.Module):

    def __init__(self, vocab_size, embedding_dimension, context_length):
        
        # Run nn.Modules initialization
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        
        # Create matrices filled with random numbers for additional columns that are vocab_size x embedding_dimension and context_length x embedding_dimension
        # These will contain our token and positon embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)
        
    
    # Takes token ids for a batch and translates them into a grid of meaning vectors and position vectors
    def forward(self, x):
        
        # Get the token embeddings for every id in x
        token_embeddings = self.token_embedding(x)
    
        # Create a tensor of positions and create it on the same device x is on
        # Tensors can handle math operations, can be moved to GPU, and know their own shape
        positions = torch.arange(x.shape[1], device=x.device)
    
        # Get the position embeddings for every position
        position_embeddings = self.position_embedding(positions)
    
        # Return the token and position embeddings added together
        return token_embeddings + position_embeddings
        