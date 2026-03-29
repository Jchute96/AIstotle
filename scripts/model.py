import torch
import torch.nn as nn


# Inherits from Pytorch's nn module that handles neural networks
class AttentionHead(nn.Module):
    
    def __init__(self, embedding_dimension, head_size):
        
        super().__init__()
        
        # Create three matrices that are responsible for the query, key, and value
        self.query = nn.Linear(embedding_dimension, head_size)
        self.key = nn.Linear(embedding_dimension, head_size)
        self.value = nn.Linear(embedding_dimension, head_size)
        
        self.head_size = head_size
        
    
    def forward(self, x):
        
        # Create unique query, key, and value vectors for each token embedding in x
        query_vector = self.query(x)
        key_vector = self.key(x)
        value_vector = self.value(x)
        
        # Calculate the relevance scores by using the dot product to compare each tokens query and key vectors
        relevance_scores = torch.matmul(query_vector, key_vector.transpose(-2, -1))
        
        # Scale the scores down by dividing by the square root of head_size
        relevance_scores = relevance_scores / (self.head_size ** 0.5)
        
        # Convert relevance scores into probabilities ranging from 0 to 1 that represent how related each token is to the others
        relevance_weights = torch.softmax(relevance_scores, dim=-1)
        
        # Multiply each tokens relevance weights with the value vector of each other token to get the specific context between a token and other tokens
        context_vector = torch.matmul(relevance_weights, value_vector)
        
        return context_vector


class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_dimension, num_heads):
        
        super().__init__()
        
        # Get the head size 
        head_size = embedding_dimension // num_heads
        
        # Create all of the attention heads
        self.heads = nn.ModuleList([ AttentionHead(embedding_dimension, head_size) for head in range(num_heads)])
        
    
    def forward(self, x):
        
        # Pass x to each heads forward method and concatenate the results together
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class FeedForward(nn.Module):
    
    def __init__(self, embedding_dimension):
        
        super().__init__()
        
        # Create a set of layers for input to pass sequentially through
        self.layers = nn.Sequential(
            # Expand the vectors dimension to 4 x embedding dimension size
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            # Set any negative numbers to 0
            nn.ReLU(),
            # Reduce the vectors dimension to original dimension size
            nn.Linear(4 * embedding_dimension, embedding_dimension)
        )
        
    def forward(self, x):
        return self.layers(x)
        

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