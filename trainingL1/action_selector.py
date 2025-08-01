# trainingL1/action_selector.py
# Neural network action selection for both average strategy and best response

import torch
import numpy as np
import random
from typing import Tuple, Optional


class ActionSelector:
    """
    CHOOSES actions (and bet size) for both players
    Handles action selection for both average strategy and best response networks.
    """
    
    def __init__(self, env, avg_pytorch_net, br_pytorch_net):
        self.env = env
        self.avg_pytorch_net = avg_pytorch_net
        self.br_pytorch_net = br_pytorch_net
        self.current_episode = 0
    
    def set_current_episode(self, episode: int):
        """Set the current episode for exploration noise calculation."""
        self.current_episode = episode
    
    def _get_network_action(self, network, features, state, bet_config) -> Tuple[int, Optional[int]]:
        """Generic method to get action from any network with specified bet sizing config."""
        network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            predictions = network(features_tensor)
            action_probs = predictions['action_probs'][0].numpy()
        network.train()  # Reset to training mode
        
        legal_actions = state['legal_actions']
        
        # Handle edge case of no legal actions
        if not legal_actions:
            return 1, None  # Default to check/call
        
        # Filter and normalize probabilities
        filtered_probs = np.zeros(3)
        for action in legal_actions:
            filtered_probs[action] = action_probs[action]
        
        if np.sum(filtered_probs) > 0:
            filtered_probs /= np.sum(filtered_probs)
        else:
            # Fallback to uniform distribution over legal actions
            filtered_probs = np.zeros(3)
            for action in legal_actions:
                filtered_probs[action] = 1.0 / len(legal_actions)
        
        # Final safety check
        prob_sum = np.sum(filtered_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            filtered_probs = np.zeros(3)
            for action in legal_actions:
                filtered_probs[action] = 1.0 / len(legal_actions)
        
        action = np.random.choice(3, p=filtered_probs)
        
        # Handle bet sizing and call amount
        amount = None
        if action == 2 and 2 in legal_actions:
            sizing_output = predictions['bet_sizing'][0].numpy()
            amount = self._determine_bet_size(state, sizing_output, state['to_move'], bet_config)
        elif action == 1:
            # If action is a call, determine the amount needed to call
            current_max_bet = max(state['current_bets'])
            player_bet = state['current_bets'][state['to_move']]
            amount_to_call = current_max_bet - player_bet
            if amount_to_call > 0:
                amount = amount_to_call
        
        return action, amount
    
    def _determine_bet_size(self, state, sizing_output, player_id, config) -> int:
        """Generic bet size determination using configuration parameters."""
        min_raise = self.env._min_raise_amount(player_id)
        max_bet = state['stacks'][player_id]
        pot = state['pot']
        
        # Handle edge cases
        if min_raise is None or min_raise > max_bet:
            return max_bet  # All-in
        if max_bet <= 0:
            return 0
        
        # Extract and process bet fraction
        bet_fraction = float(sizing_output[0])
        
        # Add exploration noise during training
        if self.current_episode < 500:
            noise_strength = config['noise_strength'] * (1.0 - self.current_episode / 500)
            bet_fraction += random.uniform(-noise_strength, noise_strength)
        
        # Clamp to configuration range
        bet_fraction = max(config['min_clamp'], min(config['max_clamp'], bet_fraction))
        
        # Map to betting range
        overbet_size = min(int(pot * config['overbet_mult']), max_bet)
        bet_range = overbet_size - min_raise
        
        if bet_range > 0:
            # Normalize bet_fraction to [0, 1] for interpolation
            normalized_fraction = (bet_fraction - config['min_clamp']) / (config['max_clamp'] - config['min_clamp'])
            continuous_size = min_raise + (normalized_fraction * bet_range)
            final_size = int(continuous_size)
        else:
            final_size = min_raise
        
        # Final enforcement of legal bet amount
        return max(min_raise, min(final_size, max_bet))
    
    def _get_average_strategy_action(self, features, state) -> Tuple[int, Optional[int]]:
        """Get action from average strategy network (GTO approximation)."""
        bet_config = {
            'noise_strength': 0.1,
            'min_clamp': 0.1,
            'max_clamp': 2.0,
            'overbet_mult': 2.0
        }
        return self._get_network_action(self.avg_pytorch_net, features, state, bet_config)
    
    def _get_best_response_action(self, features, state) -> Tuple[int, Optional[int]]:
        """Get action from best response network (exploiter)."""
        bet_config = {
            'noise_strength': 0.15,
            'min_clamp': 0.05,
            'max_clamp': 2.2,
            'overbet_mult': 2.2
        }
        return self._get_network_action(self.br_pytorch_net, features, state, bet_config)
    
    def _determine_gto_bet_size(self, state, sizing_output, player_id) -> int:
        """Determine GTO-focused bet size - legacy method for compatibility."""
        config = {
            'noise_strength': 0.1,
            'min_clamp': 0.1,
            'max_clamp': 2.0,
            'overbet_mult': 2.0
        }
        return self._determine_bet_size(state, sizing_output, player_id, config)
    
    def _determine_exploitative_bet_size(self, state, sizing_output, player_id) -> int:
        """Determine exploitative bet size - legacy method for compatibility."""
        config = {
            'noise_strength': 0.15,
            'min_clamp': 0.05,
            'max_clamp': 2.2,
            'overbet_mult': 2.2
        }
        return self._determine_bet_size(state, sizing_output, player_id, config)
    
    def _get_tight_aggressive_action(self, features, state) -> Tuple[int, Optional[int]]:
        """Simple tight-aggressive strategy for exploitability testing."""
        legal_actions = state['legal_actions']
        if not legal_actions:
            return 1, None  # Default fallback
            
        hand_strength = features[0]  # First feature is hand strength
        
        # Tight preflop requirements
        if state['stage'] == 0:
            if hand_strength > 0.6:  # Strong hands only
                if 2 in legal_actions:
                    return 2, self._determine_gto_bet_size(state, [0.7], state['to_move'])
                elif 1 in legal_actions:
                    return 1, None
                else:
                    return legal_actions[0], None  # Fallback to first legal action
            else:
                # Try to fold, but fallback to legal action if fold not available
                if 0 in legal_actions:
                    return 0, None
                else:
                    return legal_actions[0], None
        
        # Postflop: aggressive with strong hands, fold weak hands
        if hand_strength > 0.5:
            if 2 in legal_actions:
                return 2, self._determine_gto_bet_size(state, [0.8], state['to_move'])
            elif 1 in legal_actions:
                return 1, None
            else:
                return legal_actions[0], None
        else:
            if 0 in legal_actions:
                return 0, None
            else:
                return legal_actions[0], None
    
    def _get_loose_passive_action(self, features, state) -> Tuple[int, Optional[int]]:
        """Simple loose-passive strategy for exploitability testing."""
        legal_actions = state['legal_actions']
        if not legal_actions:
            return 1, None  # Default fallback
            
        hand_strength = features[0]  # First feature is hand strength
        
        # Loose preflop - call with many hands
        if state['stage'] == 0:
            if hand_strength > 0.2:  # Call with many hands
                if 1 in legal_actions:
                    return 1, None
                elif 2 in legal_actions:
                    return 2, self._determine_gto_bet_size(state, [0.5], state['to_move'])
                else:
                    return legal_actions[0], None
            else:
                if 0 in legal_actions:
                    return 0, None
                else:
                    return legal_actions[0], None
        
        # Postflop: passive - mostly call/check, rarely bet
        if hand_strength > 0.3:
            if 1 in legal_actions:
                return 1, None
            elif 2 in legal_actions and random.random() < 0.2:  # Rarely bet
                return 2, self._determine_gto_bet_size(state, [0.5], state['to_move'])
            else:
                return legal_actions[0], None
        else:
            if 0 in legal_actions:
                return 0, None
            else:
                return legal_actions[0], None
            
