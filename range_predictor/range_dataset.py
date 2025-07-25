# range_predictor/range_dataset.py
"""
Range Dataset for Neural Range Prediction

PyTorch Dataset class for loading and processing range prediction training data.
Handles feature vectors and hand property labels collected during self-play.
"""

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

class RangeDataset(Dataset):
    """
    PyTorch Dataset for range prediction training.
    
    Loads (feature_vector, hand_properties) pairs collected during self-play
    and provides them in format suitable for PyTorch training.
    """
    
    def __init__(self, 
                 data_file: str, 
                 feature_dim: int = 184,
                 normalize_features: bool = True,
                 hand_properties: List[str] = None):
        """
        Initialize the Range Dataset.
        
        Args:
            data_file: Path to the training data file (JSON lines format)
            feature_dim: Expected dimension of feature vectors
            normalize_features: Whether to normalize input features
            hand_properties: List of hand property names to predict
        """
        self.data_file = data_file
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features
        
        if hand_properties is None:
            self.hand_properties = [
                'premium_pair', 'mid_pair', 'small_pair',
                'suited_broadway', 'offsuit_broadway', 'suited_connector',
                'suited_ace', 'bluff_candidate', 'strong_draw', 'weak_draw'
            ]
        else:
            self.hand_properties = hand_properties
        
        # Load and process the data
        self.data = self._load_data()
        
        # Compute normalization statistics if needed
        self.feature_mean = None
        self.feature_std = None
        if self.normalize_features and len(self.data) > 0:
            self._compute_normalization_stats()
    
    def _load_data(self) -> List[Dict]:
        """Load training data from file."""
        if not os.path.exists(self.data_file):
            print(f"Warning: Data file {self.data_file} does not exist. Creating empty dataset.")
            return []
        
        data = []
        try:
            with open(self.data_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        
                        # Validate required fields
                        if 'features' not in item or 'hand_properties' not in item:
                            print(f"Warning: Skipping line {line_num} - missing required fields")
                            continue
                        
                        # Validate feature dimension
                        if len(item['features']) != self.feature_dim:
                            print(f"Warning: Skipping line {line_num} - feature dim {len(item['features'])} != {self.feature_dim}")
                            continue
                        
                        data.append(item)
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} - JSON decode error: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Data file {self.data_file} not found. Starting with empty dataset.")
        
        print(f"Loaded {len(data)} training samples from {self.data_file}")
        return data
    
    def _compute_normalization_stats(self):
        """Compute mean and std for feature normalization."""
        if len(self.data) == 0:
            return
        
        features = np.array([item['features'] for item in self.data])
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0)
        
        # Prevent division by zero
        self.feature_std = np.maximum(self.feature_std, 1e-8)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using computed statistics."""
        if self.feature_mean is None or self.feature_std is None:
            return features
        
        return (features - self.feature_mean) / self.feature_std
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single training sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features_tensor, hand_properties_dict)
        """
        item = self.data[idx]
        
        # Convert features to tensor
        features = np.array(item['features'], dtype=np.float32)
        if self.normalize_features:
            features = self._normalize_features(features)
        features_tensor = torch.from_numpy(features)
        
        # Convert hand properties to tensors
        hand_props = {}
        for prop_name in self.hand_properties:
            # Default to 0.0 if property not found
            prop_value = item['hand_properties'].get(prop_name, 0.0)
            hand_props[prop_name] = torch.tensor(prop_value, dtype=torch.float32)
        
        return features_tensor, hand_props
    
    def get_sample_info(self) -> Dict:
        """Get information about the dataset."""
        if len(self.data) == 0:
            return {
                'num_samples': 0,
                'feature_dim': self.feature_dim,
                'hand_properties': self.hand_properties
            }
        
        # Compute property statistics
        prop_stats = {}
        for prop in self.hand_properties:
            values = [item['hand_properties'].get(prop, 0.0) for item in self.data]
            prop_stats[prop] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'positive_rate': np.mean([v > 0.5 for v in values])
            }
        
        return {
            'num_samples': len(self.data),
            'feature_dim': self.feature_dim,
            'hand_properties': self.hand_properties,
            'property_statistics': prop_stats,
            'normalized': self.normalize_features
        }


def create_data_loaders(data_file: str, 
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       feature_dim: int = 184,
                       shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_file: Path to the training data file
        batch_size: Batch size for training
        train_split: Proportion of data to use for training
        feature_dim: Feature vector dimension
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = RangeDataset(data_file, feature_dim=feature_dim)
    
    if len(full_dataset) == 0:
        print("Warning: No data loaded. Returning empty loaders.")
        return None, None
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


# Utility function for hand property classification
def classify_hand_properties(rank1: int, rank2: int, suited: bool) -> Dict[str, float]:
    """
    Classify a two-card hand into property categories.
    
    Args:
        rank1: Rank of first card (0=2, 12=A)
        rank2: Rank of second card (0=2, 12=A) 
        suited: Whether the hand is suited
        
    Returns:
        Dictionary of hand property probabilities (0.0 or 1.0)
    """
    # Ensure rank1 >= rank2 for consistency
    high_rank, low_rank = max(rank1, rank2), min(rank1, rank2)
    
    properties = {
        'premium_pair': 0.0,
        'mid_pair': 0.0, 
        'small_pair': 0.0,
        'suited_broadway': 0.0,
        'offsuit_broadway': 0.0,
        'suited_connector': 0.0,
        'suited_ace': 0.0,
        'bluff_candidate': 0.0,
        'strong_draw': 0.0,
        'weak_draw': 0.0
    }
    
    is_pair = (high_rank == low_rank)
    is_broadway = high_rank >= 9  # T, J, Q, K, A
    is_ace = high_rank == 12  # Ace
    gap = high_rank - low_rank
    
    if is_pair:
        if high_rank >= 8:  # TT+
            properties['premium_pair'] = 1.0
        elif high_rank >= 4:  # 66-99
            properties['mid_pair'] = 1.0
        else:  # 22-55
            properties['small_pair'] = 1.0
    else:
        if suited:
            if is_broadway and low_rank >= 9:  # AKs-QJs
                properties['suited_broadway'] = 1.0
            elif is_ace and low_rank <= 3:  # A5s-A2s
                properties['suited_ace'] = 1.0
            elif gap <= 1 and high_rank >= 8:  # JTs, T9s, etc.
                properties['suited_connector'] = 1.0
            elif low_rank <= 6:  # Low suited cards
                properties['bluff_candidate'] = 1.0
        else:
            if is_broadway and low_rank >= 9:  # AKo-QJo
                properties['offsuit_broadway'] = 1.0
            elif low_rank <= 4:  # Low offsuit cards
                properties['bluff_candidate'] = 1.0
    
    return properties

