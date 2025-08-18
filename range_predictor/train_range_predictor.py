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

from .range_network import RangeNetwork, create_range_network
from .range_dataset import RangeDataset, create_data_loaders

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
        
        # Use Mean Squared Error for comparing continuous embedding vectors
        self.criterion = nn.MSELoss()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.property_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        # --- TRAIN LOOP CHANGE ---
        for features, target_embedding in train_loader:
            features = features.to(self.device)
            target_embedding = target_embedding.to(self.device)
            
            self.optimizer.zero_grad()
            
            # The model now outputs a single embedding vector
            predicted_embedding = self.model(features)
            
            # Calculate the MSE loss between the two vectors
            loss = self.criterion(predicted_embedding, target_embedding)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            # --- VALIDATION LOOP CHANGE ---
            for features, target_embedding in val_loader:
                features = features.to(self.device)
                target_embedding = target_embedding.to(self.device)
                
                predicted_embedding = self.model(features)
                loss = self.criterion(predicted_embedding, target_embedding)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
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
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_info = "No validation"
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(save_path)
                else:
                    patience_counter += 1
                
                val_info = f"Val Loss: {val_loss:.4f}"
            
            # Print progress
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, {val_info}")
            
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
    parser.add_argument('--data_file', type=str, default='training_output/range_training_data.jsonl',
                        help='Path to training data file')
    parser.add_argument('--model_save_path', type=str, default='range_predictor/range_predictor.pt',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=498, help='Feature vector dimension')
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

