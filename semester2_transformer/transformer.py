"""
Simplified Transformer implementation from scratch
Based on the PDF: Extracting LTL formulas from Transformers

Architecture:
- Leftmost hard-max attention
- Single-head attention
- No positional encoding
- No masking
- Boolean word classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedTransformer(nn.Module):
    """
    Simplified Transformer for Boolean word classification as described in the PDF
    
    The transformer takes an admissible word (ending with omega symbol) and outputs
    a Boolean value True or False based on the first element of the feature vector
    at the last position after processing through L layers.
    """
    
    def __init__(self, vocab_size, d_model, d_prime, num_layers, activation=F.relu, threshold=0.0):
        """
        Args:
            vocab_size: Size of vocabulary (including omega symbol)
            d_model: Dimension of feature vectors
            d_prime: Dimension for query/key/value matrices
            num_layers: Number of transformer layers L
            activation: Activation function σ (default: ReLU)
            threshold: Classification threshold θ
        """
        super().__init__()
        self.d_model = d_model
        self.d_prime = d_prime
        self.num_layers = num_layers
        self.activation = activation
        self.threshold = threshold
        
        # Encoding matrix W_enc: vocab_size x d_model
        # No two rows should be equal
        self.W_enc = nn.Parameter(torch.randn(vocab_size, d_model))
        # Ensure no two rows are equal (add small random noise)
        with torch.no_grad():
            noise = torch.randn_like(self.W_enc) * 0.01
            self.W_enc += noise
        
        # Layer parameters for each layer ℓ ∈ {1, ..., L}
        # Query, Key, Value matrices: d_model x d_prime
        self.Q_layers = nn.ModuleList([
            nn.Linear(d_model, d_prime, bias=False) 
            for _ in range(num_layers)
        ])
        self.K_layers = nn.ModuleList([
            nn.Linear(d_model, d_prime, bias=False) 
            for _ in range(num_layers)
        ])
        self.V_layers = nn.ModuleList([
            nn.Linear(d_model, d_prime, bias=False) 
            for _ in range(num_layers)
        ])
        
        # Output matrix O^ℓ: d_prime x d_model
        self.O_layers = nn.ModuleList([
            nn.Linear(d_prime, d_model, bias=False) 
            for _ in range(num_layers)
        ])
        
    def leftmost_argmax(self, scores):
        """
        Leftmost hard-max attention: LArgMax function
        Returns one-hot vector where 1 is at the leftmost maximum position
        
        Args:
            scores: Attention scores [batch_size, seq_length, seq_length]
            
        Returns:
            Attention weights [batch_size, seq_length, seq_length] with one-hot rows
        """
        batch_size, seq_length, _ = scores.shape
        output = torch.zeros_like(scores)
        
        for b in range(batch_size):
            for i in range(seq_length):
                row = scores[b, i, :]
                max_val = row.max()
                # Find all positions with maximum value
                max_indices = (row == max_val).nonzero(as_tuple=True)[0]
                if len(max_indices) > 0:
                    # Select the leftmost maximum
                    leftmost_max_idx = max_indices[0].item()
                    output[b, i, leftmost_max_idx] = 1.0
                    
        return output
    
    def forward_raw(self, x):
        """
        Forward pass returning raw logits (before threshold)
        
        Args:
            x: Token indices [batch_size, seq_length]
            
        Returns:
            Raw logits [batch_size]
        """
        batch_size, seq_length = x.shape
        
        # Initial encoding: X_0
        X = self.W_enc[x]  # [batch_size, seq_length, d_model]
        
        # Apply layers ℓ = 1 to L
        for layer_idx in range(self.num_layers):
            Q = self.Q_layers[layer_idx](X)
            K = self.K_layers[layer_idx](X)
            V = self.V_layers[layer_idx](X)
            scores = torch.bmm(Q, K.transpose(1, 2))
            A = self.leftmost_argmax(scores)
            attended = torch.bmm(A, V)
            output = self.O_layers[layer_idx](attended)
            X = self.activation(output) + X
        
        # Return raw value from last position, first element
        return X[:, -1, 0]
    
    def forward(self, x):
        """
        Forward pass of the simplified transformer
        
        Args:
            x: Token indices [batch_size, seq_length]
               Should be an admissible word ending with omega symbol
            
        Returns:
            Boolean classification [batch_size] - True if above threshold, False otherwise
        """
        raw_output = self.forward_raw(x)
        return raw_output > self.threshold
