# range_predictor/range_network.py
"""
Range Network for Neural Range Prediction

PyTorch neural network that predicts opponent hand properties from game features.
Outputs probabilities for different hand categories to enable accurate range construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class RangeNetwork(nn.Module):
    """
    Neural network for predicting a hand embedding vector.
    """
    
    def __init__(self, input_dim: int = 184, hidden_dims: list = None, dropout: float = 0.3, embedding_dim: int = 8):
        """
        Initialize the Range Network.
        
        Args:
            input_dim: Size of input feature vector (default 184 from schema)
            hidden_dims: List of hidden layer sizes [256, 128, 64]
            dropout: Dropout probability for regularization
            embedding_dim: Size of output embedding vector (8 for new properties)
        """
        super(RangeNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # --- START: ARCHITECTURE CHANGE ---
        # Replace the multiple output heads with a single head for the embedding vector.
        self.embedding_head = nn.Linear(prev_dim, embedding_dim)
        # --- END: ARCHITECTURE CHANGE ---
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            
        Returns:
            The predicted hand embedding vector [batch_size, embedding_dim]
        """
        x = self.feature_layers(features)
        
        # --- START: FORWARD PASS CHANGE ---
        # Output the embedding vector. Use tanh to constrain outputs to the [-1, 1] range.
        embedding = torch.tanh(self.embedding_head(x))
        return embedding
        # --- END: FORWARD PASS CHANGE ---
    
    def predict_hand_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict hand embedding for a single opponent state.
        
        Args:
            features: Feature vector for one opponent [input_dim]
            
        Returns:
            Hand embedding vector [embedding_dim]
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Add batch dimension
            
            embedding = self.forward(features)
            return embedding.squeeze()  # Remove batch dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'embedding_dim': self.embedding_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_range_network(input_dim: int = 184, config: Dict[str, Any] = None) -> RangeNetwork:
    """
    Factory function to create a RangeNetwork with optional configuration.
    
    Args:
        input_dim: Size of input feature vector
        config: Optional configuration dictionary
        
    Returns:
        Initialized RangeNetwork
    """
    if config is None:
        config = {}
    
    return RangeNetwork(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        dropout=config.get('dropout', 0.3),
        embedding_dim=config.get('embedding_dim', 8)
    )

