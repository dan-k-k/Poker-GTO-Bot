# app/trainingL1/network_trainer.py
# Neural network training logic for both average strategy and best response networks

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque


class NetworkTrainer:
    """
    UPDATES the chosen network
    Handles training of both average strategy and best response networks.
    """
    
    def __init__(self, avg_pytorch_net, br_pytorch_net):
        self.avg_pytorch_net = avg_pytorch_net # the GTOPokerNet
        self.br_pytorch_net = br_pytorch_net # the GTOPokerNet
        
        # Optimizers with learning rate scheduling
        self.avg_optimizer = optim.Adam(self.avg_pytorch_net.parameters(), lr=0.0001)
        self.br_optimizer = optim.Adam(self.br_pytorch_net.parameters(), lr=0.0001)
        
        # Learning rate schedulers
        self.avg_scheduler = optim.lr_scheduler.StepLR(self.avg_optimizer, step_size=200, gamma=0.8)
        self.br_scheduler = optim.lr_scheduler.StepLR(self.br_optimizer, step_size=200, gamma=0.8)
        
        # 🎯 AS Loss Weights - Tunable hyperparameters
        self.as_action_weight = 1.0             # Action classification (base weight)
        self.as_bet_weight = 10.0                # Bet sizing regression
        self.as_entropy_weight_start = 0.2      # Starting entropy weight
        self.as_entropy_weight_end = 0.1       # Ending entropy weight (decays over time)
        
        # 🎯 BR Loss Weights - Tunable hyperparameters (Updated for stability)
        self.br_policy_weight = 1.0     # Policy loss (base weight) - Slightly reduced from 5.0
        self.br_value_weight = 0.5      # Value function loss (vf_coef) - Increased from 0.5
        self.br_entropy_weight = 0.1   # Entropy bonus (ent_coef)
        self.br_bet_weight = 1.0        # Bet sizing loss
        
        # Training state
        self.current_episode = 0
        
        # Buffers for training data
        self.reservoir_buffer = deque(maxlen=200000)  # Larger buffer for better distribution
        self.br_buffer = deque(maxlen=50000)  # Experience replay for best response
        self.as_validation_buffer = deque(maxlen=5000)  # AS validation data (unseen)
        self.br_validation_buffer = deque(maxlen=5000)  # BR validation data (unseen)
        
        # Mutation tracking
        self.last_mutation_episode = -10
        self.mutation_cooldown = 5
        
        # BR validation tracking
        self.br_validation_losses = deque(maxlen=5)
        self.br_plateau_threshold = 10
        self.br_baseline_loss = None
    
    def set_current_episode(self, episode: int):
        """Set the current episode for training context."""
        self.current_episode = episode
    
    def add_to_reservoir_buffer(self, experience: Dict):
        """
        Add experience to reservoir buffer using true reservoir sampling.
        
        This prevents the buffer from being dominated by recent strategies,
        ensuring diverse historical strategies remain available for AS training.
        """
        if len(self.reservoir_buffer) < self.reservoir_buffer.maxlen:
            self.reservoir_buffer.append(experience)
        else:
            # TRUE RESERVOIR SAMPLING
            # Instead of always removing the oldest (popleft), replace a RANDOM element.
            # This keeps old, diverse strategies in the buffer much longer.
            idx_to_replace = random.randint(0, len(self.reservoir_buffer) - 1)
            
            # Convert deque to list temporarily for indexed access
            buffer_list = list(self.reservoir_buffer)
            buffer_list[idx_to_replace] = experience
            
            # Rebuild deque from modified list
            self.reservoir_buffer.clear()
            self.reservoir_buffer.extend(buffer_list)
    
    def add_to_br_buffer(self, experience: Dict):
        """Add experience to BR buffer (for BR training)."""
        self.br_buffer.append(experience)
    
    def add_to_br_validation_buffer(self, experience: Dict):
        """Add experience to BR validation buffer (for unbiased evaluation)."""
        self.br_validation_buffer.append(experience)
    
    def add_to_as_validation_buffer(self, experience: Dict):
        """Add experience to AS validation buffer (for unbiased AS evaluation)."""
        self.as_validation_buffer.append(experience)
    
    def train_average_strategy(self, show_debug=False) -> float:
        """
        Train the average strategy network using supervised learning (imitation learning).
        
        Key insight: AS learns to imitate the actions that BR network has taken over time.
        This is pure supervised learning, not reinforcement learning.
        
        Head 1: Action Head - Classification (cross-entropy loss)
        Head 2: Bet Sizing Head - Regression (MSE loss)
        """
        if len(self.reservoir_buffer) < 32:  # Reduced threshold for faster startup
            return 0.0
        
        # Sample from reservoir buffer which contains BR network's historical actions
        batch_size = min(512, len(self.reservoir_buffer))
        batch = random.sample(list(self.reservoir_buffer), batch_size)
        
        features = torch.FloatTensor(np.array([exp['features'] for exp in batch]))
        actions = torch.LongTensor(np.array([exp['action'] for exp in batch]))
        
        self.avg_optimizer.zero_grad()
        predictions = self.avg_pytorch_net(features)
        
        # === HEAD 1: ACTION HEAD (SUPERVISED CLASSIFICATION) ===
        # Target: actions that BR network actually took
        action_probs = predictions['action_probs']
        action_loss = torch.nn.functional.cross_entropy(action_probs, actions)
        
        # === HEAD 2: BET SIZING HEAD (SUPERVISED REGRESSION) ===
        # Target: bet sizes that BR network actually used
        bet_predictions = predictions['bet_sizing'].squeeze()
        bet_loss = self._calculate_as_bet_sizing_loss(batch, bet_predictions)
        
        # === OPTIONAL: ENTROPY REGULARIZATION ===
        # Decaying entropy weight over time to reduce exploration as training progresses
        progress = min(self.current_episode / 1000, 1.0)
        entropy_weight = self.as_entropy_weight_start * (1.0 - progress) + self.as_entropy_weight_end * progress
        entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1))
        
        # === COMBINED LOSS WITH CONFIGURABLE WEIGHTS ===
        total_loss = (self.as_action_weight * action_loss) + (self.as_bet_weight * bet_loss) + (entropy_weight * entropy_loss)
        
        # Debug output
        if show_debug:
            weighted_action = self.as_action_weight * action_loss.item()
            weighted_bet = self.as_bet_weight * bet_loss.item()
            weighted_entropy = entropy_weight * entropy_loss.item()
            print(f"   AS DEBUG: action={action_loss.item():.6f}*{self.as_action_weight}={weighted_action:.6f}")
            print(f"   AS DEBUG: bet={bet_loss.item():.6f}*{self.as_bet_weight}={weighted_bet:.6f}")
            print(f"   AS DEBUG: entropy={entropy_loss.item():.6f}*{entropy_weight:.4f}={weighted_entropy:.6f}")
            print(f"   AS DEBUG: total_loss={total_loss.item():.6f}")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_pytorch_net.parameters(), 2.0)
        self.avg_optimizer.step()
        self.avg_scheduler.step()
        
        return total_loss.item()
    
    def train_best_response(self, show_debug=False) -> float:
        """
        Train the best response network using PPO-style RL with dual heads.
        
        Head 1: Action Head - PPO-style value learning (include prediction)
        Head 2: Bet Sizing Head - Reward-supervised learning (bet sizes that led to high rewards)
        """
        if len(self.br_buffer) < 32:  # Reduced threshold for faster startup
            return 0.0
        
        batch = random.sample(list(self.br_buffer), min(128, len(self.br_buffer)))
        
        features = torch.FloatTensor(np.array([exp['features'] for exp in batch]))
        actions = torch.LongTensor(np.array([exp['action'] for exp in batch]))
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch]))
        
        self.br_optimizer.zero_grad()
        predictions = self.br_pytorch_net(features)
        
        # === PPO ACTOR-CRITIC TRAINING ===
        action_probs = predictions['action_probs']
        state_values = predictions['state_values'].squeeze()
        
        # === ADVANTAGE CALCULATION ===
        # Advantage = actual reward - predicted state value
        advantages = rewards - state_values.detach()  # Stop gradients for advantage
        
        # === POLICY LOSS (ACTOR) ===
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        policy_loss = -torch.mean(action_log_probs * advantages)
        
        # === VALUE LOSS (CRITIC) ===
        value_loss = torch.nn.functional.mse_loss(state_values, rewards)
        
        # === ENTROPY BONUS ===
        entropy_bonus = -dist.entropy().mean()
        
        # === BET SIZING LOSS ===
        bet_predictions = predictions['bet_sizing'].squeeze()
        bet_loss = self._calculate_bet_sizing_loss(batch, bet_predictions, advantages)
        
        # === COMBINED LOSS WITH CONFIGURABLE WEIGHTS ===
        total_loss = (self.br_policy_weight * policy_loss) + (self.br_value_weight * value_loss) + (self.br_entropy_weight * entropy_bonus) + (self.br_bet_weight * bet_loss)
        
        # Debug output
        if show_debug:
            # Calculate average reward components from the batch
            avg_reward = rewards.mean().item()
            avg_profit_reward = np.mean([exp.get('reward_profit', 0.0) for exp in batch])
            avg_equity_reward = np.mean([exp.get('reward_equity', 0.0) for exp in batch])
            
            # Additional PPO-specific debug info
            avg_advantage = advantages.mean().item()
            avg_state_value = state_values.mean().item()

            print(f"   BR DEBUG (Losses):")
            print(f"     Policy : {policy_loss.item():.4f} * {self.br_policy_weight:.1f} = {self.br_policy_weight * policy_loss.item():.4f}")
            print(f"     Value  : {value_loss.item():.4f} * {self.br_value_weight:.1f} = {self.br_value_weight * value_loss.item():.4f}")
            print(f"     Entropy: {entropy_bonus.item():.4f} * {self.br_entropy_weight:.2f} = {self.br_entropy_weight * entropy_bonus.item():.4f}")
            print(f"     BetSize: {bet_loss.item():.4f} * {self.br_bet_weight:.1f} = {self.br_bet_weight * bet_loss.item():.4f}")
            print(f"     ------------------------------------")
            print(f"     Total Loss: {total_loss.item():.4f}")
            print(f"   BR DEBUG (Rewards & Values):")
            print(f"     Avg Reward       : {avg_reward: .4f} (Profit: {avg_profit_reward: .4f}, Equity: {avg_equity_reward: .4f})")
            print(f"     Avg State Value  : {avg_state_value: .4f} (Agent's Prediction)")
            print(f"     Avg Advantage    : {avg_advantage: .4f} (Reward vs. Prediction)")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.br_pytorch_net.parameters(), 2.0)
        self.br_optimizer.step()
        self.br_scheduler.step()
        
        return total_loss.item()
    
    def _calculate_as_bet_sizing_loss(self, batch: List[Dict], bet_predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate supervised regression loss for AS bet sizing head.
        
        Target: Learn to predict the exact bet sizes that BR network used.
        """
        bet_targets = []
        bet_mask = []
        
        for exp in batch:
            if exp['action'] == 2 and exp['bet_amount'] > 0:  # Only for bet/raise actions
                # Normalize bet size relative to pot (AS uses more conservative range)
                pot_size = exp.get('pot_size', 100)  # Fallback pot size
                normalized_bet = min(exp['bet_amount'] / max(pot_size, 1), 2.2) / 2.2
                # CONSISTENCY FIX: Train network to predict sqrt of target since ActionSelector squares the output
                sqrt_target = torch.sqrt(torch.tensor(normalized_bet, dtype=torch.float32)).item()
                bet_targets.append(sqrt_target)
                bet_mask.append(1.0)
            else:
                bet_targets.append(0.0)
                bet_mask.append(0.0)  # No loss for non-betting actions
        
        bet_targets = torch.FloatTensor(bet_targets)
        bet_mask = torch.FloatTensor(bet_mask)
        
        # Standard MSE loss for regression
        bet_loss = torch.mean(bet_mask * (bet_predictions - bet_targets)**2)
        
        return bet_loss  # Weight applied in main loss calculation
    
    def _calculate_bet_sizing_loss(self, batch: List[Dict], bet_predictions: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Calculate advantage-weighted bet sizing loss for BR network.
        
        Key insight: Train the bet sizing head to predict bet sizes that led to 
        surprisingly good outcomes (high advantage = better than expected).
        """
        bet_targets = []
        bet_weights = []
        
        for i, exp in enumerate(batch):
            if exp['action'] == 2 and exp['bet_amount'] > 0:  # Only for bet/raise actions
                # Normalize bet size relative to pot (exploiter can bet larger)
                pot_size = exp.get('pot_size', 100)  # Fallback pot size
                normalized_bet = min(exp['bet_amount'] / max(pot_size, 1), 2.2) / 2.2
                # CONSISTENCY FIX: Train network to predict sqrt of target since ActionSelector squares the output
                sqrt_target = torch.sqrt(torch.tensor(normalized_bet, dtype=torch.float32)).item()
                bet_targets.append(sqrt_target)
                
                # Weight by reward: higher rewards = more important to learn from
                # Use advantage (how much better than expected) as weight
                # Sigmoid ensures positive weights in reasonable range
                advantage_weight = torch.sigmoid(advantages[i]).item() + 0.1  # 0.1 to 1.1 range
                bet_weights.append(advantage_weight)
            else:
                bet_targets.append(0.0)
                bet_weights.append(0.0)  # No loss for non-betting actions
        
        bet_targets = torch.FloatTensor(bet_targets)
        bet_weights = torch.FloatTensor(bet_weights)
        
        # Weighted MSE loss: learn bet sizes that led to high rewards
        bet_loss = torch.mean(bet_weights * (bet_predictions - bet_targets)**2)
        
        return bet_loss  # Weight applied in main loss calculation
    
    def calculate_br_validation_loss(self) -> float:
        """Calculate BR validation loss on truly held-out data to detect local minima."""
        if len(self.br_validation_buffer) < 64:
            return float('inf')
        
        # Set BR network to eval mode
        self.br_pytorch_net.eval()
        
        batch_size = min(128, len(self.br_validation_buffer))
        batch = random.sample(list(self.br_validation_buffer), batch_size)  # Fixed: use validation data
        
        features = torch.FloatTensor(np.array([exp['features'] for exp in batch]))
        actions = torch.LongTensor(np.array([exp['action'] for exp in batch]))
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch]))
        
        # Extract bet sizing targets
        bet_targets = []
        bet_mask = []
        for exp in batch:
            if exp['action'] == 2 and exp['bet_amount'] > 0:
                normalized_bet = min(exp['bet_amount'] / max(exp['pot_size'], 1), 2.2) / 2.2
                bet_targets.append(normalized_bet)
                bet_mask.append(1.0)
            else:
                bet_targets.append(0.0)
                bet_mask.append(0.0)
        
        bet_targets = torch.FloatTensor(bet_targets)
        bet_mask = torch.FloatTensor(bet_mask)
        
        with torch.no_grad():
            predictions = self.br_pytorch_net(features)
            
            # PPO validation loss (same components as training)
            action_probs = predictions['action_probs']
            state_values = predictions['state_values'].squeeze()
            
            # Use same weights as training for consistency
            
            # Advantage calculation
            advantages = rewards - state_values
            
            # Policy loss
            dist = torch.distributions.Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            policy_loss = -torch.mean(action_log_probs * advantages)
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(state_values, rewards)
            
            # Entropy bonus
            entropy_bonus = -dist.entropy().mean()
            
            # Bet sizing loss (unweighted - weight applied in total)
            bet_predictions = predictions['bet_sizing'].squeeze()
            bet_loss = torch.mean(bet_mask * (bet_predictions - bet_targets)**2)
            
            total_loss = (self.br_policy_weight * policy_loss) + (self.br_value_weight * value_loss) + (self.br_entropy_weight * entropy_bonus) + (self.br_bet_weight * bet_loss)
        
        # Reset to training mode
        self.br_pytorch_net.train()
        
        return total_loss.item()
    
    def calculate_as_validation_loss(self) -> float:
        """Calculate AS validation loss on held-out data."""
        if len(self.as_validation_buffer) < 64:
            return float('inf')
        
        # Set to eval mode
        self.avg_pytorch_net.eval()
        
        batch_size = min(128, len(self.as_validation_buffer))
        batch = random.sample(list(self.as_validation_buffer), batch_size)
        
        features = torch.FloatTensor(np.array([exp['features'] for exp in batch]))
        actions = torch.LongTensor(np.array([exp['action'] for exp in batch]))
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch]))
        
        # Extract bet sizing targets
        bet_targets = []
        bet_mask = []
        for exp in batch:
            if exp['action'] == 2 and exp['bet_amount'] > 0:
                normalized_bet = min(exp['bet_amount'] / max(exp['pot_size'], 1), 2.2) / 2.2
                bet_targets.append(normalized_bet)
                bet_mask.append(1.0)
            else:
                bet_targets.append(0.0)
                bet_mask.append(0.0)
        
        bet_targets = torch.FloatTensor(bet_targets)
        bet_mask = torch.FloatTensor(bet_mask)
        
        with torch.no_grad():
            predictions = self.avg_pytorch_net(features)
            
            # Action prediction loss
            action_probs = predictions['action_probs']
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            regrets = -rewards  # CFR: convert rewards to regrets
            policy_loss = torch.mean(action_log_probs * regrets * 3.0)
            
            # Bet sizing loss (unweighted - weight applied in total)
            bet_predictions = predictions['bet_sizing'].squeeze()
            bet_loss = torch.mean(bet_mask * (bet_predictions - bet_targets)**2)
            
            # Apply AS weights for consistency with training
            total_loss = (self.as_action_weight * policy_loss) + (self.as_bet_weight * bet_loss)
        
        # Reset to training mode
        self.avg_pytorch_net.train()
        
        return total_loss.item()
    
    def mutate_br_strategy(self):
        """Apply strategic mutation to BR network to escape local minima."""
        mutation_strength = 0.02  # 2% parameter perturbation
        
        print(f"   Applying {mutation_strength*100:.1f}% parameter mutation to BR network...")
        
        with torch.no_grad():
            for param in self.br_pytorch_net.parameters():
                if param.requires_grad:
                    # Add gaussian noise proportional to parameter magnitude
                    noise = torch.randn_like(param.data) * mutation_strength
                    param.data += noise
        
        # Reset optimizer momentum to prevent old gradients from undoing mutation
        self.br_optimizer.state = {}
        print("   BR optimizer state reset to preserve mutation effects")
    
    def should_mutate_br(self) -> bool:
        """Check if BR validation loss has plateaued and mutation should be triggered."""
        # Check cooldown period first
        episodes_since_mutation = self.current_episode - self.last_mutation_episode
        if episodes_since_mutation < self.mutation_cooldown:
            return False  # Still in cooldown period
        
        if len(self.br_validation_losses) < self.br_plateau_threshold:
            return False  # Not enough data yet
        
        recent_losses = list(self.br_validation_losses)[-self.br_plateau_threshold:]
        
        # Use relative improvement from baseline instead of absolute plateau
        if self.br_baseline_loss is None:
            return False  # No baseline set yet
        
        current_loss = recent_losses[-1]
        improvement_from_baseline = self.br_baseline_loss - current_loss  # More positive = better
        
        # Check if BR has stopped improving relative to when it started against this AS
        recent_improvement = max(recent_losses) - min(recent_losses)  # Recent learning progress
        
        # Plateau conditions (adjusted for relative thresholds):
        # 1. Made significant improvement from baseline (BR learned to exploit this AS)
        # 2. Recent progress is minimal (learning has stalled)
        baseline_magnitude = abs(self.br_baseline_loss) + 1e-6  # Avoid division by zero
        significant_baseline_improvement = improvement_from_baseline > 0.05 * baseline_magnitude  # 5% of baseline
        minimal_recent_progress = recent_improvement < 0.02 * baseline_magnitude  # 2% of baseline
        
        if significant_baseline_improvement and minimal_recent_progress:
            print(f"   BR exploitation plateau detected:")
            print(f"     Baseline: {self.br_baseline_loss:.4f} → Current: {current_loss:.4f} (improved by {improvement_from_baseline:.4f})")
            print(f"     Recent progress: {recent_improvement:.4f} (stalled)")
            print(f"     Episodes since last mutation: {episodes_since_mutation}")
            return True
        
        return False
    
    def set_br_baseline_loss(self, loss: float):
        """Set the BR baseline loss for plateau detection."""
        self.br_baseline_loss = loss
    
    def add_br_validation_loss(self, loss: float):
        """Add a BR validation loss for plateau detection."""
        self.br_validation_losses.append(loss)
    
    def set_last_mutation_episode(self, episode: int):
        """Set the last mutation episode."""
        self.last_mutation_episode = episode
    
    def update_loss_weights(self, as_weights: Dict = None, br_weights: Dict = None):
        """Update loss weights during training for dynamic hyperparameter tuning."""
        if as_weights:
            if 'action' in as_weights:
                self.as_action_weight = as_weights['action']
            if 'bet' in as_weights:
                self.as_bet_weight = as_weights['bet']
            if 'entropy_start' in as_weights:
                self.as_entropy_weight_start = as_weights['entropy_start']
            if 'entropy_end' in as_weights:
                self.as_entropy_weight_end = as_weights['entropy_end']
            print(f"📊 Updated AS weights: action={self.as_action_weight}, bet={self.as_bet_weight}, entropy={self.as_entropy_weight_start}→{self.as_entropy_weight_end}")
        
        if br_weights:
            if 'policy' in br_weights:
                self.br_policy_weight = br_weights['policy']
            if 'value' in br_weights:
                self.br_value_weight = br_weights['value']
            if 'entropy' in br_weights:
                self.br_entropy_weight = br_weights['entropy']
            if 'bet' in br_weights:
                self.br_bet_weight = br_weights['bet']
            print(f"📊 Updated BR weights: policy={self.br_policy_weight}, value={self.br_value_weight}, entropy={self.br_entropy_weight}, bet={self.br_bet_weight}")
    
    def get_current_loss_weights(self) -> Dict:
        """Get current loss weights for inspection."""
        return {
            'as': {
                'action': self.as_action_weight,
                'bet': self.as_bet_weight,
                'entropy_start': self.as_entropy_weight_start,
                'entropy_end': self.as_entropy_weight_end
            },
            'br': {
                'policy': self.br_policy_weight,
                'value': self.br_value_weight,
                'entropy': self.br_entropy_weight,
                'bet': self.br_bet_weight
            }
        }
    
    def update_average_network(self):
        """Update average strategy with accumulated experience."""
        print("📊 Updating average strategy network...")
        
        # The average network should represent the average of all strategies played
        # In practice, this is approximated by training on the reservoir buffer
        if len(self.reservoir_buffer) > 512:
            # Multiple training steps for better convergence
            for _ in range(3):  # Multiple updates per call
                self.train_average_strategy()
            print(f"   Reservoir: {len(self.reservoir_buffer)}")

