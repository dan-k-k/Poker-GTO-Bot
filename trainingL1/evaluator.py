# trainingL1/evaluator.py
# Exploitability evaluation and GTO testing

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque


class Evaluator:
    """
    1. Handles exploitability measurement and GTO evaluation.
    2. DECIDES whether to train AS (avg strat.) or BR (best response)
    """
    
    def __init__(self, env, feature_extractor, action_selector):
        self.env = env
        self.feature_extractor = feature_extractor
        self.action_selector = action_selector
    
    def measure_gto_exploitability(self) -> float:
        """Measure how exploitable the average strategy is using multiple opponent strategies."""
        # Set networks to eval mode for consistent evaluation
        self.action_selector.avg_pytorch_net.eval()
        self.action_selector.br_pytorch_net.eval()
        
        total_exploitability = 0
        num_strategies = 3  # Test against multiple different strategies
        
        for strategy_idx in range(num_strategies):
            strategy_loss = 0
            num_hands = 100
            
            for _ in range(num_hands):
                state = self.env.reset(preserve_stacks=False)
                self.feature_extractor.new_hand(self.env.state.stacks.copy())
                initial_stack = self.env.state.stacks[0]
                
                done = False
                while not done:
                    current_player = state['to_move']
                    
                    if current_player == 0:  # Average strategy (being tested)
                        features, _ = self.feature_extractor.extract_features(self.env.state, 0)
                        action, amount = self.action_selector._get_average_strategy_action(features, state)
                    else:  # Opponent strategy
                        features, _ = self.feature_extractor.extract_features(self.env.state, 1)
                        
                        if strategy_idx == 0:
                            # Best response network
                            action, amount = self.action_selector._get_best_response_action(features, state)
                        elif strategy_idx == 1:
                            # Tight aggressive strategy
                            action, amount = self.action_selector._get_tight_aggressive_action(features, state)
                        else:
                            # Loose passive strategy
                            action, amount = self.action_selector._get_loose_passive_action(features, state)
                    
                    state, _, done = self.env.step(action, amount)
                
                final_stack = self.env.state.stacks[0]
                strategy_loss += (initial_stack - final_stack)  # How much average strategy loses
            
            total_exploitability += strategy_loss / (num_hands * 200)  # Normalized per strategy
        
        # Reset to training mode
        self.action_selector.avg_pytorch_net.train()
        self.action_selector.br_pytorch_net.train()
        
        return total_exploitability / num_strategies  # Average exploitability across strategies
    
    def should_train_best_response(self, episode: int, return_reason: bool = False) -> Tuple[bool, Optional[str]]:
        """Training schedule: 10 BR bootstrap, then 1 BR : 2 AS alternating."""
        
        # Phase 1: Bootstrap (Episodes 0-9)
        # Train BR first to populate reservoir buffer for AS
        if episode < 10:
            should_train_br = True
            reason = f"Phase 1: Bootstrap BR training (episode {episode})"
        
        # Phase 2: Alternating 1 BR : 2 AS (Episodes 10+)
        # Cycle every 3 episodes: 1 BR, then 2 AS
        else:
            cycle_position = (episode - 10) % 3
            should_train_br = cycle_position == 0  # First episode of each 3-episode cycle = BR
            
            if should_train_br:
                cycle_number = (episode - 10) // 3 + 1
                reason = f"Phase 2: BR training (Cycle {cycle_number}, episode {episode})"
            else:
                cycle_number = (episode - 10) // 3 + 1
                as_episode = cycle_position  # 1 or 2
                reason = f"Phase 2: AS training {as_episode}/2 (Cycle {cycle_number}, episode {episode})"
        
        if return_reason:
            return should_train_br, reason
        return should_train_br
    
