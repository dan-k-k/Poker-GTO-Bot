# app/trainingL1/action_selector.py
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
    
    def _get_network_action(self, network, features, state, bet_config) -> Tuple[int, Optional[int], dict]:
        """Generic method to get action from any network with specified bet sizing config."""
        network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            predictions = network(features_tensor)
            action_probs = predictions['action_probs'][0].numpy()
            
            # Extract debug info
            debug_info = {
                'action_probs': action_probs.copy(),
                'state_value': predictions.get('state_values', torch.tensor([0.0]))[0].numpy() if 'state_values' in predictions else 0.0
            }
        network.train()  # Reset to training mode
        
        legal_actions = state['legal_actions']
        
        # Handle edge case of no legal actions
        if not legal_actions:
            return 1, None, {}  # Default to check/call
        
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
        
        return action, amount, debug_info
    
    def _determine_bet_size(self, state, sizing_output, player_id, config) -> int:
        """
        Generic bet size determination using configuration parameters.
        This version includes both the de-normalization bug fix AND the
        strategic quadratic mapping for finer control over smaller bets.
        """
        # Step 1: Get the absolute legal boundaries for the bet in chips.
        min_legal_raise_chips = self.env._min_raise_amount(player_id)
        max_legal_raise_chips = state['stacks'][player_id]
        pot = state['pot']

        if min_legal_raise_chips is None or min_legal_raise_chips > max_legal_raise_chips:
            return max_legal_raise_chips

        # Step 2: Get and apply strategic mapping to the network's [0, 1] output.
        normalized_value = float(sizing_output[0])
        normalized_value = max(0.0, min(1.0, normalized_value))

        # --- STRATEGIC ENHANCEMENT ---
        # Apply quadratic scaling to give the network finer control over smaller bet sizes.
        normalized_value = normalized_value ** 2
        # --- END ENHANCEMENT ---

        # Step 3: De-normalize the value to the desired pot-relative size.
        min_pot_relative = config['min_clamp']  # e.g., 0.05 (5%)
        max_pot_relative = config['max_clamp']  # e.g., 2.2 (220%)
        
        desired_pot_relative_size = min_pot_relative + normalized_value * (max_pot_relative - min_pot_relative)

        # Step 4: Convert the desired pot-relative size into a raw chip amount.
        desired_chips = int(pot * desired_pot_relative_size)

        # Step 5: Enforce the absolute legal boundaries.
        final_chips = max(min_legal_raise_chips, min(desired_chips, max_legal_raise_chips))
        
        return int(final_chips)
    
    def _get_average_strategy_action(self, features, state) -> Tuple[int, Optional[int], dict]:
        """Get action from average strategy network (GTO approximation)."""
        bet_config = {
            'noise_strength': 0.1,
            'min_clamp': 0.05,
            'max_clamp': 2.2,
            'overbet_mult': 2.2
        }
        return self._get_network_action(self.avg_pytorch_net, features, state, bet_config)
    
    def _get_best_response_action(self, features, state) -> Tuple[int, Optional[int], dict]:
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
            'min_clamp': 0.05,
            'max_clamp': 2.2,
            'overbet_mult': 2.2
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
    
    def _get_tight_aggressive_action(self, features, state) -> Tuple[int, Optional[int], dict]:
        """Simple tight-aggressive strategy for exploitability testing."""
        legal_actions = state['legal_actions']
        if not legal_actions:
            return 1, None, {}  # Default fallback
            
        hand_strength = features[0]  # First feature is hand strength
        
        # Tight preflop requirements
        if state['stage'] == 0:
            if hand_strength > 0.6:  # Strong hands only
                if 2 in legal_actions:
                    return 2, self._determine_gto_bet_size(state, [0.7], state['to_move']), {}
                elif 1 in legal_actions:
                    return 1, None, {}
                else:
                    return legal_actions[0], None, {}  # Fallback to first legal action
            else:
                # Try to fold, but fallback to legal action if fold not available
                if 0 in legal_actions:
                    return 0, None, {}
                else:
                    return legal_actions[0], None, {}
        
        # Postflop: aggressive with strong hands, fold weak hands
        if hand_strength > 0.5:
            if 2 in legal_actions:
                return 2, self._determine_gto_bet_size(state, [0.8], state['to_move']), {}
            elif 1 in legal_actions:
                return 1, None, {}
            else:
                return legal_actions[0], None, {}
        else:
            if 0 in legal_actions:
                return 0, None, {}
            else:
                return legal_actions[0], None, {}
    
    def _get_loose_passive_action(self, features, state) -> Tuple[int, Optional[int], dict]:
        """Simple loose-passive strategy for exploitability testing."""
        legal_actions = state['legal_actions']
        if not legal_actions:
            return 1, None, {}  # Default fallback
            
        hand_strength = features[0]  # First feature is hand strength
        
        # Loose preflop - call with many hands
        if state['stage'] == 0:
            if hand_strength > 0.2:  # Call with many hands
                if 1 in legal_actions:
                    return 1, None, {}
                elif 2 in legal_actions:
                    return 2, self._determine_gto_bet_size(state, [0.5], state['to_move']), {}
                else:
                    return legal_actions[0], None, {}
            else:
                if 0 in legal_actions:
                    return 0, None, {}
                else:
                    return legal_actions[0], None, {}
        
        # Postflop: passive - mostly call/check, rarely bet
        if hand_strength > 0.3:
            if 1 in legal_actions:
                return 1, None, {}
            elif 2 in legal_actions and random.random() < 0.2:  # Rarely bet
                return 2, self._determine_gto_bet_size(state, [0.5], state['to_move']), {}
            else:
                return legal_actions[0], None, {}
        else:
            if 0 in legal_actions:
                return 0, None, {}
            else:
                return legal_actions[0], None, {}
            
