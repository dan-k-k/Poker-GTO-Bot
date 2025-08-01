# trainingL1/data_collector.py
# Game simulation and data collection for NFSP training

import numpy as np
import random
import json
from typing import Dict, List, Tuple, Optional
from collections import deque

try:
    from .equity_calculator import EquityCalculator, RangeConstructor
    from .live_feature_debugger import LiveFeatureDebugger, format_features_for_hand_log, dump_all_features_to_log
    from ..range_predictor.range_dataset import classify_hand_properties
    from ..analyzers.event_identifier import HandEventIdentifier, HandHistory, RawAction, ActionType, Street
except ImportError:
    from equity_calculator import EquityCalculator, RangeConstructor
    from trainingL1.live_feature_debugger import LiveFeatureDebugger, format_features_for_hand_log, dump_all_features_to_log
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from range_predictor.range_dataset import classify_hand_properties
    from analyzers.event_identifier import HandEventIdentifier, HandHistory, RawAction, ActionType, Street


class DataCollector:
    """
    PLAYS 200 hands (or whatever length is set)
    Handles game simulation and data collection for NFSP training.
    """
    
    def __init__(self, env, feature_extractor, stack_depth_simulator=None, stats_tracker=None,
                 profit_weight=0.2, equity_weight=0.8):
        self.env = env
        self.feature_extractor = feature_extractor
        self.stack_depth_simulator = stack_depth_simulator
        self.stats_tracker = stats_tracker
        
        # 🎯 Reward Shaping Weights - Configurable hyperparameters
        self.profit_weight = profit_weight
        self.equity_weight = equity_weight
        
        # Episode win tracking (across all sessions in an episode)
        self.episode_session_wins = {'player_0': 0, 'player_1': 0}
        self.current_session_wins = {'player_0': 0, 'player_1': 0}
        self.session_hands = 0
        
        # Equity calculation components
        self.equity_calculator = EquityCalculator()
        self.range_constructor = None  # Will be set when action_selector is available
        
        # Action history tracking for range construction
        self.opponent_action_history = []
        
        # Live feature debugger for hand history logging
        self.live_feature_debugger = LiveFeatureDebugger()
        
        # Track exhaustive dumps to prevent spam
        self.exhaustive_dump_triggered = {}
        
        # Range training data collection
        self.range_data_file = 'training_output/range_training_data.jsonl'
        self.range_data_buffer = []
        self.range_data_batch_size = 100  # Write in batches for efficiency
        
        # Hand-level training buffer for before/after state collection
        self.hand_training_buffer = []
        
        # HandEventIdentifier for clean statistical event detection
        self.event_identifier = HandEventIdentifier()
        
        # Current hand tracking for HandHistory creation
        self.current_hand_actions = []
        self.current_hand_id = ""
        self.current_starting_stacks = {}
        self.current_dealer_position = 0
        self.current_community_cards = {}
    
    def _initialize_range_constructor(self, action_selector):
        """Initialize range constructor with action selector."""
        if self.range_constructor is None:
            self.range_constructor = RangeConstructor(action_selector, self.feature_extractor)
    
    def _collect_range_training_data(self, features: List[float], hole_cards: List[str]):
        """
        Collect training data for range prediction network.
        
        Args:
            features: Complete feature vector for the opponent 
            hole_cards: Opponent's actual hole cards
        """
        try:
            if len(hole_cards) != 2 or '??' in hole_cards:
                return  # Skip if we don't have valid hole cards
            
            # Convert card strings to ranks and determine if suited
            card1_rank = self._card_string_to_rank(hole_cards[0])
            card2_rank = self._card_string_to_rank(hole_cards[1])
            suited = hole_cards[0][1] == hole_cards[1][1]  # Same suit
            
            if card1_rank == -1 or card2_rank == -1:
                return  # Skip invalid cards
            
            # Get hand properties using the classification function
            hand_properties = classify_hand_properties(card1_rank, card2_rank, suited)
            
            # CRITICAL FIX: Sanitize features to avoid equity paradox
            # Remove features that depend on range predictions to prevent circular learning
            sanitized_features = self._sanitize_features_for_range_training(features)
            
            # Create training sample
            training_sample = {
                'features': sanitized_features,
                'hand_properties': hand_properties
            }
            
            # Add to buffer
            self.range_data_buffer.append(training_sample)
            
            # Write batch if buffer is full
            if len(self.range_data_buffer) >= self.range_data_batch_size:
                self._write_range_data_batch()
                
        except Exception as e:
            # Silently skip on error to avoid disrupting training
            pass
    
    def _card_string_to_rank(self, card_str: str) -> int:
        """Convert card string like '2s' to rank (0-12)."""
        if len(card_str) < 1:
            return -1
        rank_char = card_str[0]
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                   '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        return rank_map.get(rank_char, -1)
    
    def _write_range_data_batch(self):
        """Write accumulated range training data to file."""
        try:
            with open(self.range_data_file, 'a') as f:
                for sample in self.range_data_buffer:
                    f.write(json.dumps(sample) + '\n')
            self.range_data_buffer.clear()
        except Exception as e:
            # Silently skip on error to avoid disrupting training
            pass
    
    def flush_range_data(self):
        """Flush any remaining range training data to file."""
        if self.range_data_buffer:
            self._write_range_data_batch()
    
    def _start_new_hand(self, hand_num: int, current_episode: int):
        """Initialize tracking for a new hand using HandEventIdentifier architecture."""
        self.current_hand_id = f"ep{current_episode}_h{hand_num}"
        self.current_hand_actions = []
        self.current_starting_stacks = dict(enumerate(self.env.state.stacks.copy()))
        self.current_dealer_position = getattr(self.env.state, 'dealer', 0)
        self.current_community_cards = {street: [] for street in Street}
    
    def _record_raw_action(self, player_id: int, action: int, amount: int, state: Dict):
        """Record a raw action for later analysis by HandEventIdentifier."""
        # Convert numeric action to ActionType
        action_type_map = {0: ActionType.FOLD, 1: ActionType.CALL, 2: ActionType.RAISE}
        if action == 1 and amount == 0:
            action_type = ActionType.CHECK
        elif action == 2 and max(state.get('current_bets', [0])) == 0:
            action_type = ActionType.BET
        else:
            action_type = action_type_map.get(action, ActionType.FOLD)
        
        # Convert stage to Street enum
        stage = state.get('stage', 0)
        street = Street(min(stage, 3))  # Cap at river
        
        # Determine if player was facing a bet
        current_bets = state.get('current_bets', [0, 0])
        was_facing_bet = max(current_bets) > current_bets[player_id]
        
        # Create RawAction object
        raw_action = RawAction(
            player_id=player_id,
            street=street,
            action_type=action_type,
            amount=amount if amount else 0,
            pot_size_before=state.get('pot', 0),
            was_facing_bet=was_facing_bet,
            stack_size=state.get('stacks', [0, 0])[player_id]
        )
        
        self.current_hand_actions.append(raw_action)
    
    def _update_community_cards(self, state: Dict):
        """Update community cards tracking based on current state."""
        board = self._get_board_cards()
        if len(board) == 0:
            return  # Preflop
        elif len(board) == 3:
            self.current_community_cards[Street.FLOP] = [self._card_string_to_id(card) for card in board]
        elif len(board) == 4:
            self.current_community_cards[Street.TURN] = [self._card_string_to_id(card) for card in board]
        elif len(board) == 5:
            self.current_community_cards[Street.RIVER] = [self._card_string_to_id(card) for card in board]
    
    def _card_string_to_id(self, card_str: str) -> int:
        """Convert card string like '2s' to card ID (0-51)."""
        if len(card_str) < 2:
            return 0
        rank_char = card_str[0]
        suit_char = card_str[1]
        
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                   '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        
        rank = rank_map.get(rank_char, 0)
        suit = suit_map.get(suit_char, 0)
        
        return rank * 4 + suit
    
    def _finish_hand_with_event_identifier(self, final_state: Dict, player_map: Dict = None):
        """Complete hand analysis using HandEventIdentifier and update StatsTracker.""" 
        if not self.stats_tracker:
            return
        
        # Get hole cards for players
        hole_cards = {}
        for player_id in [0, 1]:
            cards = self._get_hand_cards(player_id)
            if cards and len(cards) >= 2:
                hole_cards[player_id] = [self._card_string_to_id(card) for card in cards[:2]]
        
        # Get showdown information
        showdown_hands = {}
        winners = final_state.get('winners', [])
        if final_state.get('stage', 0) >= 3:  # Reached showdown
            for player_id in [0, 1]:
                if player_id in hole_cards:
                    showdown_hands[player_id] = hole_cards[player_id]
        
        # Create HandHistory object
        hand_history = HandHistory(
            hand_id=self.current_hand_id,
            players=[0, 1],
            starting_stacks=self.current_starting_stacks,
            blinds=(self.env.small_blind, self.env.big_blind),
            dealer_position=self.current_dealer_position,
            hole_cards=hole_cards,
            community_cards=self.current_community_cards,
            raw_actions=self.current_hand_actions,
            final_pot=final_state.get('pot', 0),
            winners=winners,
            showdown_hands=showdown_hands
        )
        
        # Use HandEventIdentifier to analyze the complete hand
        try:
            player_events = self.event_identifier.identify_events(hand_history)
            
            # Update StatsTracker with clean events for each player
            for player_id, events in player_events.items():
                # Use player_map to get persistent model ID, fallback to seat ID
                if player_map and player_id in player_map:
                    persistent_id = player_map[player_id]
                else:
                    persistent_id = str(player_id)  # Fallback for backwards compatibility
                
                self.stats_tracker.update_from_events(persistent_id, events)
                
        except Exception as e:
            # Don't crash training if event identification fails
            print(f"Warning: HandEventIdentifier failed: {e}")
            # Fallback to empty events to maintain training stability
            for player_id in [0, 1]:
                persistent_id = player_map.get(player_id, str(player_id)) if player_map else str(player_id)
                self.stats_tracker.update_from_events(persistent_id, {})
    
    def _sanitize_features_for_range_training(self, features: List[float]) -> List[float]:
        """
        Sanitizes features for range training using the universal metadata-driven function.
        Removes both private information (hole cards) and leaky features (range-dependent).
        """
        from analyzers.feature_utils import sanitize_features
        return sanitize_features(features, purpose='training')
    
    def _get_hand_cards(self, player_id: int) -> List[str]:
        """Extract hand cards for a player from environment state."""
        try:
            # Access hand from environment state - correct field names
            if hasattr(self.env.state, 'hole_cards') and self.env.state.hole_cards:
                hand = self.env.state.hole_cards[player_id]
                if hand and len(hand) >= 2:
                    return [self._card_to_string(hand[0]), self._card_to_string(hand[1])]
            
            # Fallback: try the state dict format
            state_dict = self.env.get_state_dict()
            if player_id == state_dict.get('to_move'):
                hole_cards = state_dict.get('hole', [])
                if len(hole_cards) >= 2:
                    return [self._card_to_string(hole_cards[0]), self._card_to_string(hole_cards[1])]
            
            return ['??', '??']  # Changed fallback to show unknown cards
        except Exception as e:
            return ['??', '??']  # Fallback if hand access fails
    
    def _get_board_cards(self) -> List[str]:
        """Extract board cards from environment state."""
        try:
            # Access board from environment state - correct field names
            if hasattr(self.env.state, 'community') and self.env.state.community:
                return [self._card_to_string(card) for card in self.env.state.community]
            
            # Fallback: try the state dict format
            state_dict = self.env.get_state_dict()
            community = state_dict.get('community', [])
            if community:
                return [self._card_to_string(card) for card in community]
            
            return []
        except Exception as e:
            return []
    
    def _card_to_string(self, card_id: int) -> str:
        """Convert card ID to string representation."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _get_street_name(self, stage: int) -> str:
        """Convert stage number to street name."""
        stage_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return stage_map.get(stage, 'preflop')
    
    def _calculate_equity_based_reward(self, player_id: int, action: int, amount: Optional[int],
                                     old_state: Dict, new_state: Dict, opponent_stats: Dict) -> float:
        """
        Calculate immediate reward based on equity change.
        
        This is the core PBRS (Potential-Based Reward Shaping) implementation.
        """
        try:
            # Initialize range constructor if needed
            if self.range_constructor is None:
                return 0.0  # Can't calculate without range constructor
                
            # Get hand and board
            my_hand = self._get_hand_cards(player_id)
            old_board = self._get_board_cards()
            
            # Use comprehensive opponent stats passed from trainer
            old_opponent_range = self.range_constructor.construct_range(
                self.opponent_action_history, old_board, old_state.get('pot', 100), opponent_stats
            )
            old_equity = self.equity_calculator.calculate_equity(
                my_hand, old_board, old_opponent_range, num_simulations=200  # Fast for training
            )
            
            # Calculate new equity after action
            if new_state.get('terminal', False):
                # Hand ended - equity is 0 or 1
                if player_id in new_state.get('winners', []):
                    new_equity = 1.0
                else:
                    new_equity = 0.0
            else:
                # Hand continues - recalculate equity
                new_board = self._get_board_cards()
                
                # Construct new range with comprehensive opponent stats
                new_opponent_range = self.range_constructor.construct_range(
                    self.opponent_action_history, new_board, new_state.get('pot', 100), opponent_stats
                )
                
                new_equity = self.equity_calculator.calculate_equity(
                    my_hand, new_board, new_opponent_range, num_simulations=200
                )
            
            # Calculate equity change
            equity_change = new_equity - old_equity
            
            # Scale by pot size (more important in bigger pots)
            pot_size = old_state.get('pot', 100)
            pot_multiplier = min(pot_size / 200.0, 3.0)  # Cap at 3x
            
            # DEBUG: Reward calculation is now tracked in training debug output
            # Return PBRS reward: Φ(new_state) - Φ(old_state)
            return equity_change * pot_multiplier
            
        except Exception:
            # Fallback to simple heuristic reward if equity calculation fails
            # This ensures training continues even if equity system has issues
            if action == 2:  # Bet/raise
                return 0.02  # Small positive reward for aggression
            elif action == 0:  # Fold
                return -0.01  # Small negative for folding
            else:  # Call
                return 0.01  # Neutral reward
    
    def collect_average_strategy_data(self, num_hands: int, action_selector, current_episode: int, opponent_stats: Dict, player_map: Dict = None) -> Tuple[List[Dict], float]:
        """
        Collect data playing average strategy vs best response with stack depth simulation.
        
        Args:
            num_hands: Number of hands to play
            action_selector: Object with _get_average_strategy_action and _get_best_response_action methods
            current_episode: Current training episode
            opponent_stats: Comprehensive stats for opponent modeling
            
        Returns:
            Tuple of (experiences, win_rate, hand_stats)
        """
        experiences = []
        wins = 0
        
        # Hand statistics tracking
        all_ins = 0
        folds = 0
        showdowns = 0
        total_actions = 0
        bets_and_raises = 0
        
        for hand_num in range(num_hands):
            # 1. DECIDE: Should we start a new session or continue the current one?
            # This decision is based on the results of the *previous* hand.
            if self.stack_depth_simulator:
                if self.stack_depth_simulator.should_reset_session(self.stack_depth_simulator.current_stacks):
                    # ACTION A: Start a fresh session with new asymmetric stacks.
                    new_stacks = self.stack_depth_simulator.reset_session()
                    state = self.env.reset(preserve_stacks=False)
                    self.env.state.stacks = list(new_stacks)
                else:
                    # ACTION B: Continue the current session, preserving stacks.
                    # The env already has the stacks from the last hand, so reset() just starts a new deal.
                    state = self.env.reset(preserve_stacks=True)
                    # DEBUG: Check deck shuffling  
                    # if hasattr(self.env, 'deck') and hasattr(self.env.deck, 'cards'):
                    #     print(f"DEBUG: First 10 cards after reset: {self.env.deck.cards[:10]}")
            else:
                # Fallback for no simulator: always reset completely.
                state = self.env.reset(preserve_stacks=False)
                        
            # By this point, `state` is guaranteed to be a fresh, playable hand.
            # Safety check: skip any terminal states that might slip through
            if state.get('terminal'):
                continue
            
            # ✅ FIX: Initialize the hand in the tracker, which also records the blind posts.
            # This replaces the old self.feature_extractor.new_hand() call.
            self.feature_extractor.history_tracker.initialize_hand_with_blinds(
                self.env.state,
                hand_number=hand_num
            )
            
            # ✅ FIX: Reset ActionSequencer to prevent stale data from previous hand
            self.feature_extractor.action_sequencer.new_street()
            
            # ✅ FIX: Clear any cached hand analyzer state from previous hand
            self.feature_extractor.hand_analyzer.clear_cache()
            
            # Initialize equity-based reward system for this hand
            self._initialize_range_constructor(action_selector)
            self.opponent_action_history = []  # Reset for new hand
            
            # Initialize new HandEventIdentifier architecture
            self._start_new_hand(hand_num, current_episode)
            
            # Clear hand training buffer for new hand
            self.hand_training_buffer.clear()
            
            hand_experiences = []
            initial_stack = self.env.state.stacks[0]
            
            # 💡 --- CORRECTED LOGGING LOGIC ---
            # Decide at the START of the hand if it needs logging for ANY reason.
            # Reason 1: It's a regularly scheduled log (e.g., every 50th hand).
            is_regular_log_hand = (hand_num % 49 == 0)
            
            # Reason 2: It's an episode where we are actively LOOKING for an interesting hand to dump.
            is_dump_watch_episode = (current_episode == 11)  # Change 11 to any target episode
            
            # A hand log will be created if either condition is true.
            should_log_hand = is_regular_log_hand or is_dump_watch_episode
            hand_log = [] if should_log_hand else None
            
            if should_log_hand:
                p0_hand = self._get_hand_cards(0)
                p1_hand = self._get_hand_cards(1)
                hand_log.append(f"[Ep {current_episode}, Hand {hand_num}] P0(AS) dealt {p0_hand}. P1(BR) dealt {p1_hand}.")
            
            done = False
            hand_action_count = 0
            current_street = "PREFLOP"
            pot_at_street_start = state.get('pot', 100)
            
            while not done:
                current_player = state['to_move']
                old_state = state.copy()  # Store state before action
                
                # Track street changes for logging
                if should_log_hand:
                    board = self._get_board_cards()
                    new_street = ""
                    if len(board) == 0:
                        new_street = "PREFLOP"
                    elif len(board) == 3:
                        new_street = "FLOP"
                    elif len(board) == 4:
                        new_street = "TURN"
                    elif len(board) == 5:
                        new_street = "RIVER"
                    
                    if new_street != current_street:
                        current_street = new_street
                        if len(board) > 0:
                            pot_at_street_start = state.get('pot', 100)
                            hand_log.append(f"- {current_street} {board} (pot: {pot_at_street_start}): ")
                        else:
                            pot_at_street_start = state.get('pot', 100)
                
                if current_player == 0:  # Average strategy agent
                    # Extract features for current player (Player 0), opponent is Player 1
                    # 🎯 GTO TRAINING: AS agent trains without opponent stats to learn unexploitable strategy
                    features, schema = self.feature_extractor.extract_features(
                        self.env.state, 0, opponent_stats=None, opponent_action_history=None)
                    
                    # === INTELLIGENT EXHAUSTIVE DUMP (CORRECTED) ===
                    # Only check for dumps on designated watch episodes.
                    dump_key = f"AS_{current_episode}"
                    if (is_dump_watch_episode and 
                        dump_key not in self.exhaustive_dump_triggered and
                        self.live_feature_debugger.is_interesting_scenario(schema, self.env.state)):
                        
                        print(f"\n📸 EXHAUSTIVE DUMP: Interesting scenario detected in Episode {current_episode}")
                        try:
                            # The hand_log is guaranteed to exist because should_log_hand is true.
                            scenario_desc = self.live_feature_debugger.describe_scenario(schema, self.env.state)
                            
                            # Add the dump content to the existing hand log.
                            hand_log.append(f"\n{'='*60}\n🎯 SCENARIO: {scenario_desc}\n{'='*60}\n")
                            
                            # Add exhaustive dump context to include complete hand history
                            context = {
                                'episode': current_episode,
                                'hand': hand_num,
                                'street': current_street,
                                'stage': self.env.state.stage,
                                'pot': state.get('pot', 0),
                                'stacks': state.get('stacks', []),
                                'scenario': scenario_desc,
                                'hand_history': hand_log  # Include complete hand history up to this point
                            }
                            
                            hand_log.append(dump_all_features_to_log(schema, f"P0(AS)-{scenario_desc}", context))
                            
                            print(f"✅ Exhaustive dump queued for scenario: {scenario_desc}")
                            self.exhaustive_dump_triggered[dump_key] = True
                        except Exception as e:
                            print(f"⚠️ Failed to queue scenario dump: {e}")
                    # === END INTELLIGENT DUMP ===
                    
                    action, amount = action_selector._get_average_strategy_action(features, state)
                    
                    # Add feature logging for detailed hand analysis
                    if should_log_hand:
                        player_name = "P0(AS)"
                        # Use detailed logging for interesting decisions
                        detailed = self.live_feature_debugger.should_log_detailed_features(schema, action, amount)
                        feature_log = format_features_for_hand_log(
                            schema, player_name, action, amount, 
                            detailed=detailed
                        )
                        hand_log.append(feature_log)
                    
                    # Store experience data with pot size at action time
                    experience = {
                        'features': features,
                        'action': action,
                        'bet_amount': amount if amount is not None else 0,
                        'pot_size': old_state.get('pot', 100)  # CRITICAL FIX: Use pot size BEFORE action
                    }
                    hand_experiences.append(experience)
                    
                else:  # Best response opponent
                    # ✅ 1. Capture the "ACTIVE" state BEFORE the BR agent acts
                    as_stats = self.stats_tracker.get_player_percentages("average_strategy_v1")
                    features_before, schema_before = self.feature_extractor.extract_features(
                        self.env.state, 1, opponent_stats=as_stats, opponent_action_history=None)
                    
                    # Store the before state in buffer
                    self.hand_training_buffer.append({
                        'features': features_before,
                        'seat_id': 1
                    })
                    
                    # ✅ 2. The BR agent makes its decision
                    action, amount = action_selector._get_best_response_action(features_before, state)
                    
                    # Add feature logging for detailed hand analysis
                    if should_log_hand:
                        player_name = "P1(BR)"
                        # Use detailed logging for interesting decisions
                        detailed = self.live_feature_debugger.should_log_detailed_features(schema_before, action, amount)
                        feature_log = format_features_for_hand_log(
                            schema_before, player_name, action, amount, 
                            detailed=detailed
                        )
                        hand_log.append(feature_log)
                    
                    # CRITICAL FIX: Track opponent actions for equity calculation
                    action_data = {'action': action, 'amount': amount or 0}
                    self.opponent_action_history.append(action_data)
                    
                # Track action statistics
                if action == 0:  # Fold
                    folds += 1
                elif action == 2:  # Raise/Bet
                    bets_and_raises += 1
                elif amount is not None and amount == state['stacks'][current_player]:  # All-in
                    all_ins += 1
                
                # Record raw action for later analysis by HandEventIdentifier  
                if self.stats_tracker:
                    self._record_raw_action(current_player, action, amount or 0, state)
                    # Update community cards tracking
                    self._update_community_cards(state)
                
                hand_action_count += 1
                total_actions += 1
                
                # DEBUG: Log the action taken
                if should_log_hand:
                    player_name = "P0" if current_player == 0 else "P1"
                    action_str = ""
                    if action == 0:
                        action_str = "folds"
                    elif action == 1:
                        if amount and amount > 0:
                            action_str = f"calls {amount}"
                        else:
                            action_str = "checks"
                    elif action == 2:
                        if amount:
                            # Show both the raise amount and total commitment for clarity
                            current_commitment = old_state.get('current_bets', [0])[current_player]
                            new_total = current_commitment + amount
                            
                            # A "bet" is the FIRST aggressive action. A "raise" is a subsequent one.
                            if max(old_state.get('current_bets', [0])) == 0:
                                action_str = f"bets {amount}"
                            else:
                                action_str = f"raises by {amount} (from {current_commitment} to {new_total})"
                        else:
                            action_str = "raises"  # Fallback for no amount
                    
                    hand_log.append(f"  {player_name} {action_str}.")
                
                # Store the stage BEFORE the action
                current_stage = old_state.get('stage', 0)
                
                # ✅ FIX: Record action in ActionSequencer BEFORE taking it
                # Only record meaningful poker actions, not engine-generated intermediate steps
                if action == 0:
                    action_type = 'fold'
                elif action == 1:
                    # ✅ FIX: Only record actual calls, not blind completions or zero-amount actions
                    current_bets = old_state.get('current_bets', [])
                    max_bet = max(current_bets) if current_bets else 0
                    player_bet = current_bets[current_player] if current_player < len(current_bets) else 0
                    to_call = max_bet - player_bet
                    
                    if to_call == 0:
                        action_type = 'check'
                    elif amount and amount > 0:
                        action_type = 'call'
                    else:
                        # Skip recording phantom actions (blind completions, etc.)
                        action_type = None
                elif action == 2:
                    # Action 2 can be bet (first aggressive action) or raise (in response to bet)
                    current_bets = old_state.get('current_bets', [])
                    max_bet = max(current_bets) if current_bets else 0
                    action_type = 'bet' if max_bet == 0 else 'raise'
                else:
                    action_type = None  # Skip unknown/invalid actions
                
                # ✅ FIX: Only record actual poker actions, skip phantom actions
                if action_type is not None:
                    # 🐛 DEBUG: Log action recording to hand history for context
                    if hasattr(self, 'hand_log'):
                        self.hand_log.append(f"    DEBUG: Recording action for P{current_player}: {action_type} {amount or 0}")
                    self.feature_extractor.record_action(current_player, action_type, amount or 0)
                
                # Take the action and get new state
                new_state, _, done = self.env.step(action, amount)
                
                # ✅ 3. Capture the "PASSIVE" state AFTER the BR agent has acted
                if current_player == 1:  # If the BR agent just moved
                    as_stats = self.stats_tracker.get_player_percentages("average_strategy_v1")
                    features_after, schema_after = self.feature_extractor.extract_features(
                        self.env.state, 1, opponent_stats=as_stats, opponent_action_history=None)
                    
                    # Store the after state in buffer
                    self.hand_training_buffer.append({
                        'features': features_after,
                        'seat_id': 1
                    })
                
                # --- CRITICAL FIX: UPDATE INVESTMENT TRACKING ---
                # Update StreetHistoryTracker with player investments for commitment features
                if action in [1, 2] and amount and amount > 0:  # Call or Bet/Raise
                    self.feature_extractor.history_tracker.update_investment(current_player, amount)
                # --- END: INVESTMENT TRACKING FIX ---
                
                # --- CRITICAL FIX: STREET TRANSITION DETECTION ---
                # Check if the street has changed after the action
                if not done and new_state.get('stage', 0) > current_stage:
                    # The street has ended. Save a snapshot of the completed street.
                    street_that_just_ended = self._get_street_name(current_stage)
                    
                    # ✅ Get the final schema which includes the calculated strategic features
                    as_stats = self.stats_tracker.get_player_percentages("average_strategy_v1") if self.stats_tracker else {}
                    _, final_schema = self.feature_extractor.extract_features(
                        self.env.state, 0, opponent_stats=as_stats, full_hand_action_history=self.opponent_action_history
                    )
                    
                    # Convert the strategic features object to a dict to be saved
                    strategic_data_to_save = None
                    if hasattr(final_schema, 'current_strategic') and final_schema.current_strategic:
                        strategic_data_to_save = final_schema.current_strategic.to_dict()
                    
                    # ✅ Pass the strategic data to the snapshot saver
                    self.feature_extractor.save_street_snapshot(
                        self.env.state,
                        strategic_features=strategic_data_to_save,
                        street=street_that_just_ended
                    )
                    
                    # Reset the ActionSequencer for the new street
                    self.feature_extractor.new_street()
                # --- END: CRITICAL FIX ---
                
                # Calculate equity-based reward for Player 0 actions
                if current_player == 0 and len(hand_experiences) > 0:
                    equity_reward = self._calculate_equity_based_reward(
                        0, action, amount, old_state, new_state, opponent_stats
                    )
                    # Add equity reward to the latest experience
                    hand_experiences[-1]['equity_reward'] = equity_reward
                
                # Update state for next iteration
                state = new_state
            
            # Calculate results
            final_stack = self.env.state.stacks[0]
            hand_profit = final_stack - initial_stack
            
            # Complete hand analysis using HandEventIdentifier
            self._finish_hand_with_event_identifier(state, player_map)
            
            # ✅ Process hand training buffer - convert to range training data
            br_hole_cards = self._get_hand_cards(1)  # BR agent is Player 1
            for buffer_entry in self.hand_training_buffer:
                self._collect_range_training_data(buffer_entry['features'], br_hole_cards)
            
            # DEBUG: Log showdown and write hand history
            if should_log_hand:
                # Get pot size from old_state before it gets reset to 0
                pot_size = old_state.get('pot', 0)
                hand_log.append(f"  (Pot: {pot_size})")
                
                if self.env.state.winners:
                    winner_str = "P0" if 0 in self.env.state.winners else "P1"
                    hand_log.append(f"- SHOWDOWN: {winner_str} wins pot of {pot_size}.")
                
                # Write to debug file
                try:
                    with open("training_output/hand_histories.log", "a") as f:
                        f.write("\n".join(hand_log) + "\n\n")
                except:
                    pass  # Don't crash training if logging fails
            
            # Track session wins
            self.session_hands += 1
            if self.env.state.winners:
                if 0 in self.env.state.winners:
                    self.current_session_wins['player_0'] += 1
                elif 1 in self.env.state.winners:
                    self.current_session_wins['player_1'] += 1
            
            # 3. UPDATE: After the hand is over, inform the simulator of the new stack sizes.
            final_stacks = self.env.state.stacks.copy()
            if self.stack_depth_simulator:
                # The simulator tracks final stacks and will signal reset on next iteration if needed
                self.stack_depth_simulator.advance_hand(final_stacks)
                
                # Count session winners silently
                if self.stack_depth_simulator.should_reset_session(final_stacks) and self.session_hands > 0:
                    self._count_session_winner()
            
            # Process experiences with combined rewards
            for exp in hand_experiences:
                # Traditional end-of-hand reward (configurable weight)
                profit_reward = hand_profit / 200.0 * self.profit_weight
                
                # Equity-based immediate reward (configurable weight)
                equity_reward = exp.get('equity_reward', 0.0) * self.equity_weight
                
                # Combined reward: immediate equity feedback + final outcome
                exp['reward'] = profit_reward + equity_reward
                exp['reward_profit'] = profit_reward  # Log profit component
                exp['reward_equity'] = equity_reward  # Log equity component
                exp['episode'] = current_episode
                # pot_size already set at action time - don't override here
                
                # Add stack depth information
                if self.stack_depth_simulator:
                    session_info = self.stack_depth_simulator.get_current_session_info()
                    exp['effective_stack'] = session_info['effective_stack']
                    exp['session_hand'] = session_info['session_hand']
            
            # Check if hand went to showdown
            if self.env.state.stage >= 3:  # Reached river or beyond
                showdowns += 1
            
            experiences.extend(hand_experiences)
            if hand_profit > 0:
                wins += 1
        
        # Flush any remaining range training data
        self.flush_range_data()
        
        return experiences, wins / num_hands
    
    def collect_best_response_data(self, num_hands: int, action_selector, current_episode: int, opponent_stats: Dict, player_map: Dict = None) -> Tuple[List[Dict], float]:
        """
        Collect data for best response training.
        
        Args:
            num_hands: Number of hands to play
            action_selector: Object with _get_average_strategy_action and _get_best_response_action methods
            current_episode: Current training episode
            opponent_stats: Comprehensive stats for opponent modeling
            
        Returns:
            Tuple of (experiences, win_rate, hand_stats)
        """
        experiences = []
        wins = 0
        
        # Hand statistics tracking
        all_ins = 0
        folds = 0
        showdowns = 0
        total_actions = 0
        bets_and_raises = 0
        
        for hand_num in range(num_hands):
            # 1. DECIDE: Should we start a new session or continue the current one?
            # This decision is based on the results of the *previous* hand.
            if self.stack_depth_simulator:
                if self.stack_depth_simulator.should_reset_session(self.stack_depth_simulator.current_stacks):
                    # ACTION A: Start a fresh session with new asymmetric stacks.
                    new_stacks = self.stack_depth_simulator.reset_session()
                    state = self.env.reset(preserve_stacks=False)
                    self.env.state.stacks = list(new_stacks)
                else:
                    # ACTION B: Continue the current session, preserving stacks.
                    # The env already has the stacks from the last hand, so reset() just starts a new deal.
                    state = self.env.reset(preserve_stacks=True)
                    # DEBUG: Check deck shuffling  
                    # if hasattr(self.env, 'deck') and hasattr(self.env.deck, 'cards'):
                    #     print(f"DEBUG: First 10 cards after reset: {self.env.deck.cards[:10]}")  
            else:
                # Fallback for no simulator: always reset completely.
                state = self.env.reset(preserve_stacks=False)
                        
            # By this point, `state` is guaranteed to be a fresh, playable hand.
            # Safety check: skip any terminal states that might slip through
            if state.get('terminal'):
                continue
            
            # ✅ FIX: Initialize the hand in the tracker, which also records the blind posts.
            # This replaces the old self.feature_extractor.new_hand() call.
            self.feature_extractor.history_tracker.initialize_hand_with_blinds(
                self.env.state,
                hand_number=hand_num
            )
            
            # ✅ FIX: Reset ActionSequencer to prevent stale data from previous hand
            self.feature_extractor.action_sequencer.new_street()
            
            # ✅ FIX: Clear any cached hand analyzer state from previous hand
            self.feature_extractor.hand_analyzer.clear_cache()
            
            # Initialize equity-based reward system for this hand
            self._initialize_range_constructor(action_selector)
            self.opponent_action_history = []  # Reset for new hand
            
            hand_experiences = []
            initial_stack = self.env.state.stacks[0]
            
            # Initialize new HandEventIdentifier architecture
            self._start_new_hand(hand_num, current_episode)
            
            done = False
            while not done:
                current_player = state['to_move']
                old_state = state.copy()  # Store state before action
                
                if current_player == 0:  # Best response agent
                    # Extract features for current player (Player 0), opponent is Player 1
                    features, _ = self.feature_extractor.extract_features(
                        self.env.state, 0, opponent_stats, self.opponent_action_history
                    )
                    
                    
                    action, amount = action_selector._get_best_response_action(features, state)
                    
                    # Store experience data with pot size at action time
                    experience = {
                        'features': features,
                        'action': action,
                        'bet_amount': amount if amount is not None else 0,
                        'pot_size': old_state.get('pot', 100)  # CRITICAL FIX: Use pot size BEFORE action
                    }
                    hand_experiences.append(experience)
                    
                else:  # Average strategy opponent
                    # Extract features for current player (Player 1), opponent is Player 0
                    # For opponent, we don't have comprehensive stats, so pass None
                    features, _ = self.feature_extractor.extract_features(
                        self.env.state, 1, opponent_stats=None, opponent_action_history=None)
                    action, amount = action_selector._get_average_strategy_action(features, state)
                    
                    # CRITICAL FIX: Track opponent actions for equity calculation
                    action_data = {'action': action, 'amount': amount or 0}
                    self.opponent_action_history.append(action_data)
                    
                    # ❌ REMOVED: Don't collect range data from AS opponent in BR training
                    # Range data should only come from BR agent (the exploiter)
                
                # Track action statistics
                if action == 0:  # Fold
                    folds += 1
                elif action == 2:  # Raise/Bet
                    bets_and_raises += 1
                elif amount is not None and amount == state['stacks'][current_player]:  # All-in
                    all_ins += 1
                
                # Record raw action for later analysis by HandEventIdentifier
                if self.stats_tracker:
                    self._record_raw_action(current_player, action, amount or 0, state)
                    # Update community cards tracking
                    self._update_community_cards(state)
                
                total_actions += 1
                
                # Store the stage BEFORE the action
                current_stage = old_state.get('stage', 0)
                
                # ✅ FIX: Record action in ActionSequencer BEFORE taking it
                # Only record meaningful poker actions, not engine-generated intermediate steps
                if action == 0:
                    action_type = 'fold'
                elif action == 1:
                    # ✅ FIX: Only record actual calls, not blind completions or zero-amount actions
                    current_bets = old_state.get('current_bets', [])
                    max_bet = max(current_bets) if current_bets else 0
                    player_bet = current_bets[current_player] if current_player < len(current_bets) else 0
                    to_call = max_bet - player_bet
                    
                    if to_call == 0:
                        action_type = 'check'
                    elif amount and amount > 0:
                        action_type = 'call'
                    else:
                        # Skip recording phantom actions (blind completions, etc.)
                        action_type = None
                elif action == 2:
                    # Action 2 can be bet (first aggressive action) or raise (in response to bet)
                    current_bets = old_state.get('current_bets', [])
                    max_bet = max(current_bets) if current_bets else 0
                    action_type = 'bet' if max_bet == 0 else 'raise'
                else:
                    action_type = None  # Skip unknown/invalid actions
                
                # ✅ FIX: Only record actual poker actions, skip phantom actions
                if action_type is not None:
                    # 🐛 DEBUG: Log action recording to hand history for context
                    if hasattr(self, 'hand_log'):
                        self.hand_log.append(f"    DEBUG: Recording action for P{current_player}: {action_type} {amount or 0}")
                    self.feature_extractor.record_action(current_player, action_type, amount or 0)
                
                # Take the action and get new state
                new_state, _, done = self.env.step(action, amount)
                
                # --- CRITICAL FIX: UPDATE INVESTMENT TRACKING ---
                # Update StreetHistoryTracker with player investments for commitment features
                if action in [1, 2] and amount and amount > 0:  # Call or Bet/Raise
                    self.feature_extractor.history_tracker.update_investment(current_player, amount)
                # --- END: INVESTMENT TRACKING FIX ---
                
                # --- CRITICAL FIX: STREET TRANSITION DETECTION ---
                # Check if the street has changed after the action
                if not done and new_state.get('stage', 0) > current_stage:
                    # The street has ended. Save a snapshot of the completed street.
                    street_that_just_ended = self._get_street_name(current_stage)
                    
                    # ✅ Get the final schema which includes the calculated strategic features
                    as_stats = self.stats_tracker.get_player_percentages("average_strategy_v1") if self.stats_tracker else {}
                    _, final_schema = self.feature_extractor.extract_features(
                        self.env.state, 0, opponent_stats=as_stats, full_hand_action_history=self.opponent_action_history
                    )
                    
                    # Convert the strategic features object to a dict to be saved
                    strategic_data_to_save = None
                    if hasattr(final_schema, 'current_strategic') and final_schema.current_strategic:
                        strategic_data_to_save = final_schema.current_strategic.to_dict()
                    
                    # ✅ Pass the strategic data to the snapshot saver
                    self.feature_extractor.save_street_snapshot(
                        self.env.state,
                        strategic_features=strategic_data_to_save,
                        street=street_that_just_ended
                    )
                    
                    # Reset the ActionSequencer for the new street
                    self.feature_extractor.new_street()
                # --- END: CRITICAL FIX ---
                
                # Calculate equity-based reward for Player 0 (BR agent) actions
                if current_player == 0 and len(hand_experiences) > 0:
                    equity_reward = self._calculate_equity_based_reward(
                        0, action, amount, old_state, new_state, opponent_stats
                    )
                    # Add equity reward to the latest experience
                    hand_experiences[-1]['equity_reward'] = equity_reward
                
                # Update state for next iteration
                state = new_state
            
            # Calculate results
            final_stack = self.env.state.stacks[0]
            hand_profit = final_stack - initial_stack
            
            # Complete hand analysis using HandEventIdentifier
            self._finish_hand_with_event_identifier(state, player_map)
            
            # Track session wins
            self.session_hands += 1
            if self.env.state.winners:
                if 0 in self.env.state.winners:
                    self.current_session_wins['player_0'] += 1
                elif 1 in self.env.state.winners:
                    self.current_session_wins['player_1'] += 1
            
            # 3. UPDATE: After the hand is over, inform the simulator of the new stack sizes.
            final_stacks = self.env.state.stacks.copy()
            if self.stack_depth_simulator:
                # The simulator tracks final stacks and will signal reset on next iteration if needed
                self.stack_depth_simulator.advance_hand(final_stacks)
                
                # Count session winners silently
                if self.stack_depth_simulator.should_reset_session(final_stacks) and self.session_hands > 0:
                    self._count_session_winner()
            
            # Check if hand went to showdown
            if self.env.state.stage >= 3:  # Reached river or beyond
                showdowns += 1
            
            # Process experiences with combined rewards
            for exp in hand_experiences:
                # Traditional end-of-hand reward (configurable weight)
                profit_reward = hand_profit / 200.0 * self.profit_weight
                
                # Equity-based immediate reward (configurable weight)
                equity_reward = exp.get('equity_reward', 0.0) * self.equity_weight
                
                # Combined reward: immediate equity feedback + final outcome
                exp['reward'] = profit_reward + equity_reward
                exp['reward_profit'] = profit_reward  # Log profit component
                exp['reward_equity'] = equity_reward  # Log equity component
                exp['episode'] = current_episode
                # pot_size already set at action time - don't override here
                
                # Add stack depth information
                if self.stack_depth_simulator:
                    session_info = self.stack_depth_simulator.get_current_session_info()
                    exp['effective_stack'] = session_info['effective_stack']
                    exp['session_hand'] = session_info['session_hand']
            
            experiences.extend(hand_experiences)
            if hand_profit > 0:
                wins += 1
        
        # Flush any remaining range training data
        self.flush_range_data()
        
        return experiences, wins / num_hands
    
    def _count_session_winner(self):
        """Count the session winner and add to episode totals."""
        # Determine session winner based on who won more hands
        p0_wins = self.current_session_wins['player_0']
        p1_wins = self.current_session_wins['player_1']
        
        if p0_wins > p1_wins:
            self.episode_session_wins['player_0'] += 1
        elif p1_wins > p0_wins:
            self.episode_session_wins['player_1'] += 1
        # Tie sessions don't count for either player
        
        # Reset for next session
        self.current_session_wins = {'player_0': 0, 'player_1': 0}
        self.session_hands = 0
    
    def reset_session_tracking(self):
        """Reset session tracking for new episode."""
        self.episode_session_wins = {'player_0': 0, 'player_1': 0}
        self.current_session_wins = {'player_0': 0, 'player_1': 0}
        self.session_hands = 0
    
    def print_episode_summary(self):
        """Print episode-level session win summary."""
        total_sessions = self.episode_session_wins['player_0'] + self.episode_session_wins['player_1']
        if total_sessions > 0:
            print(f"🏆 Episode Sessions Won: P0={self.episode_session_wins['player_0']}, P1={self.episode_session_wins['player_1']} ({total_sessions} total)")
        else:
            print("🏆 Episode Sessions Won: No completed sessions")
    
