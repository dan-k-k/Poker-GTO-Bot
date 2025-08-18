# trainingL1/evaluator.py
# Exploitability evaluation and GTO testing

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque


class Evaluator:
    """
    1. Handles exploitability measurement and GTO evaluation.
    2. DECIDES whether to train AS (avg strat.) or BR (best response) based on a configurable schedule.
    """
    
    def __init__(self, env, feature_extractor, action_selector,
                 bootstrap_episodes: int = 10,
                 br_frequency: int = 1,
                 as_frequency: int = 2):
        """
        Initializes the Evaluator with a configurable training schedule.

        Args:
            bootstrap_episodes: The number of initial episodes to train only the Best Response agent.
            br_frequency: The number of Best Response training episodes per cycle.
            as_frequency: The number of Average Strategy training episodes per cycle.
        """
        self.env = env
        self.feature_extractor = feature_extractor
        self.action_selector = action_selector
        
        # Store schedule parameters
        self.bootstrap_episodes = bootstrap_episodes
        self.br_frequency = br_frequency
        self.as_frequency = as_frequency
        self.cycle_length = br_frequency + as_frequency
    
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
                # Initialize the hand in the tracker, which also records the blind posts.
                self.feature_extractor.history_tracker.initialize_hand_with_blinds(
                    self.env.state,
                    hand_number=_
                )
                initial_stack = self.env.state.stacks[0]
                
                done = False
                while not done:
                    current_player = state['to_move']
                    
                    if current_player == 0:  # Average strategy (being tested)
                        features, _ = self.feature_extractor.extract_features(self.env.state, 0, role="AS")
                        action, amount, _ = self.action_selector._get_average_strategy_action(features, state)
                    else:  # Opponent strategy
                        features, _ = self.feature_extractor.extract_features(self.env.state, 1, role="BR")
                        
                        if strategy_idx == 0:
                            # Best response network
                            action, amount, _ = self.action_selector._get_best_response_action(features, state)
                        elif strategy_idx == 1:
                            # Tight aggressive strategy
                            action, amount, _ = self.action_selector._get_tight_aggressive_action(features, state)
                        else:
                            # Loose passive strategy
                            action, amount, _ = self.action_selector._get_loose_passive_action(features, state)
                    
                    state, _, done = self.env.step(action, amount)
                
                final_stack = self.env.state.stacks[0]
                strategy_loss += (initial_stack - final_stack)  # How much average strategy loses
            
            total_exploitability += strategy_loss / (num_hands * 200)  # Normalized per strategy
        
        # Reset to training mode
        self.action_selector.avg_pytorch_net.train()
        self.action_selector.br_pytorch_net.train()
        
        return total_exploitability / num_strategies  # Average exploitability across strategies
    
    def should_train_best_response(self, episode: int, return_reason: bool = False) -> Tuple[bool, Optional[str]]:
        """Training schedule logic now uses the configured parameters."""
        
        # Phase 1: Bootstrap
        if episode < self.bootstrap_episodes:
            should_train_br = True
            reason = f"Phase 1: Bootstrap BR training ({episode + 1}/{self.bootstrap_episodes})"
        
        # Phase 2: Alternating Cycle
        else:
            # Determine position in the current cycle
            cycle_position = (episode - self.bootstrap_episodes) % self.cycle_length
            
            # The first 'br_frequency' episodes of the cycle are for BR training
            should_train_br = cycle_position < self.br_frequency
            
            cycle_number = (episode - self.bootstrap_episodes) // self.cycle_length + 1
            if should_train_br:
                br_episode_num = cycle_position + 1
                reason = f"Phase 2: BR training {br_episode_num}/{self.br_frequency} (Cycle {cycle_number})"
            else:
                as_episode_num = cycle_position - self.br_frequency + 1
                reason = f"Phase 2: AS training {as_episode_num}/{self.as_frequency} (Cycle {cycle_number})"
        
        if return_reason:
            return should_train_br, reason
        return should_train_br
    
