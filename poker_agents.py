# poker_agents.py
# Consolidated agent architecture - replaces agents.py, agents_NN.py, gto_agent.py
# this is most up-to-date

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from feature_extractor import FeatureExtractor

class GTOPokerNet(nn.Module):
    """
    Standard GTO poker network architecture.
    Used by both training and inference.
    """
    def __init__(self, input_size: int = 736):  # Updated: Moved is_facing features to additional (self-only)
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(32, 3)  # fold, call, raise
        self.bet_sizing_head = nn.Linear(32, 1)  # continuous bet size [0,1]
        self.value_head = nn.Linear(32, 1)  # state value for PPO critic
        
        # Initialize heads
        nn.init.constant_(self.bet_sizing_head.bias, 0.4)
        nn.init.normal_(self.bet_sizing_head.weight, 0.0, 0.1)
        nn.init.constant_(self.value_head.bias, 0.0)
        nn.init.normal_(self.value_head.weight, 0.0, 0.1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        return {
            'action_probs': torch.softmax(self.action_head(shared_out), dim=-1),
            'bet_sizing': torch.sigmoid(self.bet_sizing_head(shared_out)),
            'state_values': self.value_head(shared_out)
        }

class PokerAgent:
    """
    Base poker agent class.
    All agents inherit from this.
    """
    def __init__(self, seat_id: int):
        self.seat_id = seat_id
        
    def compute_action(self, state: Dict, env) -> Tuple[int, Optional[int]]:
        """
        Compute action for the current state.
        Returns (action, amount) where action ∈ {0: fold, 1: call, 2: raise}
        """
        raise NotImplementedError
        
    def new_hand(self):
        """Reset for new hand."""
        pass
        
    def observe(self, player_action, player_id, state_before_action, env):
        """Observe an opponent's action."""
        pass
        
    def observe_showdown(self, showdown_state, env):
        """Observe showdown results."""
        pass


class GTOAgent(PokerAgent):
    """
    GTO poker agent that uses trained PyTorch models.
    This is your Layer 1 agent.
    """
    
    def __init__(self, seat_id: int, model_path: str = "gto_average_strategy.pt"):
        super().__init__(seat_id)
        self.model_path = model_path
        
        # Load the trained model
        self.network = GTOPokerNet()
        try:
            self.network.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.network.eval()
            print(f"✅ Loaded GTO model from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load GTO model: {e}")
            print("   Using random initialization")
            
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
    def compute_action(self, state: Dict, env) -> Tuple[int, Optional[int]]:
        """Compute GTO action using the trained network."""
        # Extract features using new architecture
        features = self.feature_extractor.extract_features(env.state, self.seat_id)
        
        # Get network prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            predictions = self.network(features_tensor)
            action_probs = predictions['action_probs'][0].numpy()
            bet_sizing = predictions['bet_sizing'][0].numpy()
        
        # Filter legal actions
        legal_actions = state['legal_actions']
        if not legal_actions:
            return 1, None  # Default to call
        
        # Sample action from legal actions
        filtered_probs = np.zeros(3)
        for action in legal_actions:
            filtered_probs[action] = action_probs[action]
        
        if np.sum(filtered_probs) > 0:
            filtered_probs /= np.sum(filtered_probs)
        else:
            for action in legal_actions:
                filtered_probs[action] = 1.0 / len(legal_actions)
        
        action = np.random.choice(3, p=filtered_probs)
        
        # Handle bet sizing
        amount = None
        if action == 2 and 2 in legal_actions:
            amount = self._determine_bet_size(state, bet_sizing[0])
        
        return action, amount
    
    def _determine_bet_size(self, state: Dict, sizing_output: float) -> int:
        """Convert network output to actual bet size."""
        pot = state['pot']
        stack = state['stacks'][self.seat_id]
        
        # This maps the [0, 1] output to a [0.05, 2.2] pot multiple
        MIN_BET_FRAC = 0.05
        MAX_BET_FRAC = 2.2
        pot_multiple = MIN_BET_FRAC + (MAX_BET_FRAC - MIN_BET_FRAC) * (sizing_output ** 2)

        target_bet = int(pot * pot_multiple)
        
        # Ensure legal bet size
        min_bet = 2  # Minimum bet (big blind)
        max_bet = stack
        
        final_bet = max(min_bet, min(target_bet, max_bet))
        return final_bet
    
    def new_hand(self):
        """Reset for new hand."""
        super().new_hand()
        self.feature_extractor.new_hand()


# Backward compatibility alias
NeuralPokerAgent = GTOAgent  # For existing code that uses NeuralPokerAgent

