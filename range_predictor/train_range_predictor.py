# range_predictor/train_range_predictor.py
"""
Training Script for Range Prediction Network

Trains the neural network to predict opponent hand properties from game features.
Handles data loading, training loop, validation, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime

from range_network import RangeNetwork, create_range_network
from range_dataset import RangeDataset, create_data_loaders

class RangePredictor:
    """Training and evaluation manager for range prediction."""
    
    def __init__(self, 
                 model: RangeNetwork,
                 device: torch.device = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize the range predictor trainer.
        
        Args:
            model: RangeNetwork to train
            device: Device to train on (CPU/GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss function - Binary Cross Entropy for each property
        self.criterion = nn.BCELoss()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.property_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0.0}
        num_batches = 0
        
        for batch_idx, (features, hand_props) in enumerate(train_loader):
            features = features.to(self.device)
            
            # Move hand properties to device
            target_props = {}
            for prop_name, prop_values in hand_props.items():
                target_props[prop_name] = prop_values.to(self.device).unsqueeze(1)  # Add feature dimension
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Compute loss for each property
            total_loss = 0.0
            property_losses = {}
            
            for prop_name in predictions.keys():
                if prop_name in target_props:
                    prop_loss = self.criterion(predictions[prop_name], target_props[prop_name])
                    property_losses[prop_name] = prop_loss.item()
                    total_loss += prop_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            for prop_name, loss in property_losses.items():
                if prop_name not in epoch_losses:
                    epoch_losses[prop_name] = 0.0
                epoch_losses[prop_name] += loss
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total': 0.0}
        property_predictions = {prop: [] for prop in self.model.output_heads.keys()}
        property_targets = {prop: [] for prop in self.model.output_heads.keys()}
        num_batches = 0
        
        with torch.no_grad():
            for features, hand_props in val_loader:
                features = features.to(self.device)
                
                # Move hand properties to device
                target_props = {}
                for prop_name, prop_values in hand_props.items():
                    target_props[prop_name] = prop_values.to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(features)
                
                # Compute loss for each property
                total_loss = 0.0
                for prop_name in predictions.keys():
                    if prop_name in target_props:
                        prop_loss = self.criterion(predictions[prop_name], target_props[prop_name])
                        total_loss += prop_loss
                        
                        # Collect predictions and targets for accuracy calculation
                        property_predictions[prop_name].extend(predictions[prop_name].cpu().numpy())
                        property_targets[prop_name].extend(target_props[prop_name].cpu().numpy())
                
                epoch_losses['total'] += total_loss.item()
                num_batches += 1
        
        # Average losses
        epoch_losses['total'] /= num_batches
        
        # Compute accuracies (predictions > 0.5 vs targets > 0.5)
        accuracies = {}
        for prop_name in property_predictions.keys():
            if len(property_predictions[prop_name]) > 0:
                pred_binary = np.array(property_predictions[prop_name]) > 0.5
                target_binary = np.array(property_targets[prop_name]) > 0.5
                accuracies[prop_name] = np.mean(pred_binary == target_binary)
        
        return epoch_losses, accuracies
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              num_epochs: int = 100,
              save_path: str = 'range_predictor.pt',
              early_stop_patience: int = 20) -> Dict[str, List[float]]:
        """
        Full training loop with validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum number of epochs
            save_path: Path to save the best model
            early_stop_patience: Epochs to wait before early stopping
            
        Returns:
            Dictionary containing training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model architecture: {self.model.get_model_info()}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses['total'])
            
            # Validation
            val_info = "No validation"
            if val_loader is not None:
                val_losses, val_accuracies = self.validate_epoch(val_loader)
                self.val_losses.append(val_losses['total'])
                self.property_accuracies.append(val_accuracies)
                
                # Learning rate scheduling
                self.scheduler.step(val_losses['total'])
                
                # Early stopping check
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    # Save best model
                    self.save_model(save_path)
                else:
                    patience_counter += 1
                
                val_info = f"Val Loss: {val_losses['total']:.4f}"
                if val_accuracies:
                    avg_acc = np.mean(list(val_accuracies.values()))
                    val_info += f", Avg Acc: {avg_acc:.3f}"
            
            # Print progress
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Train Loss: {train_losses['total']:.4f}, {val_info}")
            
            # Early stopping
            if val_loader is not None and patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience: {early_stop_patience})")
                break
        
        # Load best model if validation was used
        if val_loader is not None and os.path.exists(save_path):
            print(f"Loading best model from {save_path}")
            self.load_model(save_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'property_accuracies': self.property_accuracies
        }
    
    def save_model(self, path: str):
        """Save model state and training info."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_model_info(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Range Prediction Network')
    parser.add_argument('--data_file', type=str, default='trainingL1/range_training_data.jsonl',
                        help='Path to training data file')
    parser.add_argument('--model_save_path', type=str, default='range_predictor/range_predictor.pt',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=184, help='Feature vector dimension')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training data split ratio')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RANGE PREDICTOR TRAINING")
    print("=" * 60)
    print(f"Data file: {args.data_file}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Feature dimension: {args.feature_dim}")
    
    # Create model save directory
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Create data loaders
    print("\\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        args.data_file,
        batch_size=args.batch_size,
        train_split=args.train_split,
        feature_dim=args.feature_dim
    )
    
    if train_loader is None:
        print("Error: No training data found. Please generate training data first.")
        return
    
    # Create model
    print("\\nInitializing model...")
    model = create_range_network(input_dim=args.feature_dim)
    
    # Create trainer
    trainer = RangePredictor(model, learning_rate=args.learning_rate)
    
    # Train model
    print("\\nStarting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_path=args.model_save_path
    )
    
    print("\\nTraining completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"Final val loss: {history['val_losses'][-1]:.4f}")
    
    print(f"Model saved to: {args.model_save_path}")


if __name__ == '__main__':
    main()

