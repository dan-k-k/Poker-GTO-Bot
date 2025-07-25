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
    Neural network for predicting opponent range properties.
    
    Takes opponent's complete feature vector and outputs probabilities
    for different hand categories/properties.
    """
    
    def __init__(self, input_dim: int = 184, hidden_dims: list = None, dropout: float = 0.3):
        """
        Initialize the Range Network.
        
        Args:
            input_dim: Size of input feature vector (default 184 from schema)
            hidden_dims: List of hidden layer sizes [256, 128, 64]
            dropout: Dropout probability for regularization
        """
        super(RangeNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
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
        
        # Output heads for different hand properties
        # Each outputs a probability [0, 1] for that hand category
        self.output_heads = nn.ModuleDict({
            'premium_pair': nn.Linear(prev_dim, 1),      # AA-TT
            'mid_pair': nn.Linear(prev_dim, 1),          # 99-66  
            'small_pair': nn.Linear(prev_dim, 1),        # 55-22
            'suited_broadway': nn.Linear(prev_dim, 1),   # AKs-QJs
            'offsuit_broadway': nn.Linear(prev_dim, 1),  # AKo-QJo
            'suited_connector': nn.Linear(prev_dim, 1),  # JTs-54s
            'suited_ace': nn.Linear(prev_dim, 1),        # A5s-A2s
            'bluff_candidate': nn.Linear(prev_dim, 1),   # Low cards, suited gaps
            'strong_draw': nn.Linear(prev_dim, 1),       # Strong flush/straight draws
            'weak_draw': nn.Linear(prev_dim, 1),         # Weak draws, gutshots
        })
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            
        Returns:
            Dictionary of hand property probabilities
        """
        # Extract features through hidden layers
        x = self.feature_layers(features)
        
        # Apply sigmoid to each output head for probability
        outputs = {}
        for property_name, head in self.output_heads.items():
            outputs[property_name] = torch.sigmoid(head(x))
        
        return outputs
    
    def predict_range_properties(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Predict range properties for a single opponent state.
        
        Args:
            features: Feature vector for one opponent [input_dim]
            
        Returns:
            Dictionary of property probabilities as floats
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Add batch dimension
            
            outputs = self.forward(features)
            
            # Convert to float probabilities
            properties = {}
            for prop_name, prob_tensor in outputs.items():
                properties[prop_name] = prob_tensor.squeeze().item()
        
        return properties
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'output_properties': list(self.output_heads.keys()),
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
        dropout=config.get('dropout', 0.3)
    )

