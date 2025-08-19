# app/trainingL1/data_collector.py
# Orchestrates game simulation and data collection for NFSP training

import numpy as np
import json
import copy
from typing import Dict, List, Tuple, Optional, Callable

# 1. Use relative imports (single dot) for files in the same package (trainingL1)
from .game_simulator import HandHistoryManager, RewardShaper, SessionTracker
from .equity_calculator import EquityCalculator
from .range_constructors import RangeConstructor
from .live_feature_debugger import LiveFeatureDebugger, format_features_for_hand_log, dump_all_features_to_log

# 2. Use absolute imports (from the project root) for files in other packages
from app.range_predictor.range_dataset import classify_hand_properties
from app.analyzers.event_identifier import HandEventIdentifier
from app.poker_core import card_to_string, string_to_card_id, get_street_name


class DataCollector:
    """
    Plays hands and collects experiences for training by orchestrating simulation components.
    Refactored to use focused helper classes and eliminate code duplication.
    """
    
    def __init__(self, env, feature_extractor, stack_depth_simulator=None, stats_tracker=None,
                 profit_weight=0.7, equity_weight=0.3, range_constructor=None, equity_calculator=None):
        self.env = env
        self.feature_extractor = feature_extractor
        self.stack_depth_simulator = stack_depth_simulator
        self.stats_tracker = stats_tracker
        
        # 🎯 Reward Shaping Weights - Configurable hyperparameters
        self.profit_weight = profit_weight
        self.equity_weight = equity_weight
        
        # Helper class instances - use shared components when provided
        self.equity_calculator = equity_calculator or EquityCalculator()
        self.event_identifier = HandEventIdentifier()
        self.history_manager = HandHistoryManager(stats_tracker, self.event_identifier, env)
        self.session_tracker = SessionTracker()
        self.live_feature_debugger = LiveFeatureDebugger()
        
        # Use shared range constructor when provided
        self.range_constructor = range_constructor or RangeConstructor()
        self.reward_shaper: Optional[RewardShaper] = None
        
        # Range prediction data collection
        self.range_data_file = 'training_output/range_training_data.jsonl'
        self.range_data_buffer = []
        self.range_data_batch_size = 100
        
        # Track exhaustive dumps to prevent spam
        self.exhaustive_dump_triggered = {}

    def _initialize_helpers(self, action_selector):
        """Initialize components that depend on the action_selector."""
        if self.reward_shaper is None:
            self.reward_shaper = RewardShaper(self.equity_calculator, self.range_constructor)

    def collect_average_strategy_data(self, num_hands: int, action_selector, current_episode: int, 
                                    player_map: Dict = None) -> Tuple[List[Dict], float]:
        """Collects data by playing the Average Strategy against a Best Response opponent."""
        self._initialize_helpers(action_selector)
        self.session_tracker.reset_episode()
        
        # Player 0 is the agent being trained (Average Strategy)
        p0_policy = lambda features, state: action_selector._get_average_strategy_action(features, state)
        p1_policy = lambda features, state: action_selector._get_best_response_action(features, state)
        
        return self._run_simulation(
            num_hands, current_episode, p0_policy, p1_policy, 
            "AS", player_map, collect_range_data=False
        )

    def collect_best_response_data(self, num_hands: int, action_selector, current_episode: int, 
                                 player_map: Dict = None) -> Tuple[List[Dict], float]:
        """Collects data by playing the Best Response against an Average Strategy opponent."""
        self._initialize_helpers(action_selector)
        self.session_tracker.reset_episode()
        
        # Player 0 is the agent being trained (Best Response)
        p0_policy = lambda features, state: action_selector._get_best_response_action(features, state)
        p1_policy = lambda features, state: action_selector._get_average_strategy_action(features, state)
        
        return self._run_simulation(
            num_hands, current_episode, p0_policy, p1_policy, 
            "BR", player_map, collect_range_data=True
        )

    def _run_simulation(self, num_hands: int, current_episode: int, p0_policy: Callable, p1_policy: Callable,
                       p0_role: str, player_map: Optional[Dict], 
                       collect_range_data: bool) -> Tuple[List[Dict], float]:
        """
        Generic simulation loop that runs hands and collects data.
        Eliminates all code duplication between AS and BR training.
        """
        experiences = []
        wins = 0

        for hand_num in range(num_hands):
            state = self._start_hand(hand_num, current_episode)
            if state.get('terminal'): 
                continue

            hand_experiences, hand_training_buffer = [], []
            initial_stack = self.env.state.stacks[0]
            
            # Logging setup
            should_log_hand = self._should_log_hand(hand_num, current_episode)
            hand_log = [] if should_log_hand else None
            
            if should_log_hand:
                p0_hand = self._get_hand_cards(0)
                p1_hand = self._get_hand_cards(1)
                role_names = {"AS": "Average Strategy", "BR": "Best Response"}
                hand_log.append(f"[Ep {current_episode}, Hand {hand_num}] P0({p0_role}) dealt {p0_hand}. P1 dealt {p1_hand}.")
            
            done = False
            current_street = "PREFLOP"
            
            while not done:
                current_player = state['to_move']
                old_state = state.copy()
                
                # Make a deep copy of the full GameState object BEFORE the action
                # This preserves the state for the equity calculation later
                game_state_before_action = self.env.state.copy()
                
                # Define the persistent agent IDs for each player seat (absolute)
                p0_agent_id = "best_response_v1" if p0_role == "BR" else "average_strategy_v1"
                p1_agent_id = "average_strategy_v1" if p0_role == "BR" else "best_response_v1"

                # Fetch stats into unambiguous, absolute variables
                stats_p0 = self.stats_tracker.get_player_percentages(p0_agent_id) if self.stats_tracker else {}
                stats_p1 = self.stats_tracker.get_player_percentages(p1_agent_id) if self.stats_tracker else {}
                
                # Track street changes for logging
                if should_log_hand:
                    current_street = self._update_street_logging(state, current_street, hand_log)
                
                # Get action from appropriate policy
                if current_player == 0:  # Primary agent
                    # Extract features from Player 0's perspective
                    features, schema = self.feature_extractor.extract_features(
                        self.env.state, 
                        seat_id=0, 
                        role=p0_role,
                        self_stats=stats_p0,      # P0's perspective: "self" is P0
                        opponent_stats=None if p0_role == "AS" else stats_p1  # P0's perspective: "opponent" is P1
                    )
                    
                    # Intelligent exhaustive dump for interesting scenarios
                    if should_log_hand:
                        self._handle_exhaustive_dump(schema, current_episode, p0_role, hand_log)
                    
                    action, amount, debug_info = p0_policy(features, state)
                    
                    # Store experience data
                    experience = {
                        'features': features,
                        'action': action,
                        'bet_amount': amount if amount is not None else 0,
                        'pot_size': old_state.get('pot', 100)
                    }
                    hand_experiences.append(experience)
                    
                else:  # Opponent (Player 1)
                    # Determine P1's role (opposite of P0)
                    p1_role = "BR" if p0_role == "AS" else "AS"
                    
                    # Extract features from Player 1's perspective
                    features_before, schema_before = self.feature_extractor.extract_features(
                        self.env.state, 
                        seat_id=1, 
                        role=p1_role,
                        self_stats=stats_p1,      # P1's perspective: "self" is P1
                        opponent_stats=None if p1_role == "AS" else stats_p0  # P1's perspective: "opponent" is P0
                    )

                    if collect_range_data:
                        # Capture public features from Player 1's perspective for range training
                        # During AS training (as P0), 
                        public_features = self.feature_extractor.extract_public_features(
                            self.env.state, 
                            seat_id=1,
                            self_stats=stats_p1,      # P1's perspective: "self" is P1
                            opponent_stats=stats_p0   # P1's perspective: "opponent" is P0
                        )
                        hand_training_buffer.append({'features': public_features, 'seat_id': 1})
                    
                    action, amount, _ = p1_policy(features_before, state)
                    

                # Add feature logging for P1 (opponent) immediately
                if should_log_hand and current_player == 1:
                    player_name = f"P{current_player}(OPP)"
                    schema_to_log = schema_before if 'schema_before' in locals() else None
                    if schema_to_log:
                        detailed = self.live_feature_debugger.should_log_detailed_features(schema_to_log, action, amount)
                        feature_log = format_features_for_hand_log(
                            schema_to_log, player_name, action, amount, 
                            detailed=detailed,
                            network_outputs=None,
                            equity_reward=None
                        )
                        hand_log.append(feature_log)
                    
                    # Log the action taken
                    self._log_action(hand_log, current_player, action, amount, old_state)
                
                # Record action for history tracking
                self.history_manager.record_action(current_player, action, amount or 0, state)
                
                # Store the stage BEFORE the action for street transition detection
                current_stage = old_state.get('stage', 0)
                
                # Record action in ActionSequencer
                self._record_action_for_feature_extractor(current_player, action, amount, old_state)
                
                # Take the action and get new state
                new_state, _, done = self.env.step(action, amount)
                
                # Update investment tracking
                if action in [1, 2] and amount and amount > 0:
                    self.feature_extractor.history_tracker.update_investment(current_player, amount)
                
                # Street transition detection with strategic features
                if not done and new_state.get('stage', 0) > current_stage:
                    self._handle_street_transition(current_stage, p0_role, stats_p0, stats_p1)
                
                # Calculate equity-based reward for Player 0 actions
                if current_player == 0 and len(hand_experiences) > 0 and self.reward_shaper:
                    # Generate public features for P1 (the opponent) from P1's perspective
                    opponent_public_features = self.feature_extractor.extract_public_features(
                        game_state_before_action, 
                        seat_id=1,                # We want features for the player in seat 1
                        self_stats=stats_p1,      # From P1's perspective, "self" is P1
                        opponent_stats=stats_p0   # From P1's perspective, "opponent" is P0
                    )

                    equity_reward = self.reward_shaper.calculate_reward(
                        player_id=0, 
                        action=action, 
                        amount=amount, 
                        old_state=old_state,
                        new_state=new_state,
                        game_state_before_action=game_state_before_action,
                        action_sequencer=self.feature_extractor.action_sequencer,
                        opponent_public_features=opponent_public_features,
                        opponent_stats=stats_p1   # P1's stats for range construction
                    )
                    hand_experiences[-1]['equity_reward'] = equity_reward
                    
                    # Add feature logging for P0 after equity reward calculation
                    if should_log_hand:
                        player_name = f"P0({p0_role})"
                        detailed = self.live_feature_debugger.should_log_detailed_features(schema, action, amount)
                        feature_log = format_features_for_hand_log(
                            schema, player_name, action, amount, 
                            detailed=detailed,
                            network_outputs=debug_info,
                            equity_reward=equity_reward
                        )
                        hand_log.append(feature_log)
                        
                        # Log the action taken
                        self._log_action(hand_log, 0, action, amount, old_state)
                
                # Update state for next iteration
                state = new_state
            
            # Post-hand processing
            self._finish_hand(hand_experiences, state, player_map, hand_training_buffer, 
                            should_log_hand, hand_log, current_episode, initial_stack)
            
            hand_profit = self.env.state.stacks[0] - initial_stack
            # print('hand_profit', hand_profit)
            if hand_profit > 0:
                wins += 1
            
            experiences.extend(hand_experiences)

        # Cleanup
        self.flush_range_data()
        self.session_tracker.print_episode_summary()
        
        return experiences, wins / num_hands if num_hands > 0 else 0

    def _start_hand(self, hand_num: int, current_episode: int) -> Dict:
        """Resets the environment and trackers for a new hand."""
        # Handle stack depth simulation
        if self.stack_depth_simulator:
            if self.stack_depth_simulator.should_reset_session(self.stack_depth_simulator.current_stacks):
                self.session_tracker.end_session()
                new_stacks = self.stack_depth_simulator.reset_session()
                state = self.env.reset(preserve_stacks=False)
                self.env.state.stacks = list(new_stacks)
            else:
                state = self.env.reset(preserve_stacks=True)
        else:
            state = self.env.reset(preserve_stacks=False)
        
        # Initialize hand tracking
        self.feature_extractor.history_tracker.initialize_hand_with_blinds(
            self.env.state, hand_number=hand_num
        )
        self.feature_extractor.action_sequencer.new_street()
        self.feature_extractor.hand_analyzer.clear_cache()
        self.history_manager.start_hand(hand_num, current_episode)
        
        return self.env.get_state_dict()

    def _should_log_hand(self, hand_num: int, current_episode: int) -> bool:
        """Determine if this hand should be logged."""
        is_regular_log_hand = (hand_num % 49 == 0)
        is_dump_watch_episode = (current_episode == 11)
        return is_regular_log_hand or is_dump_watch_episode

    def _update_street_logging(self, state: Dict, current_street: str, hand_log: List[str]) -> str:
        """Update street logging and return new street name."""
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
            if len(board) > 0:
                pot_at_street_start = state.get('pot', 100)
                hand_log.append(f"- {new_street} {board} (pot: {pot_at_street_start}): ")
        
        return new_street

    def _handle_exhaustive_dump(self, schema, current_episode: int, p0_role: str, hand_log: List[str]):
        """Handle intelligent exhaustive dump for interesting scenarios."""
        dump_key = f"{p0_role}_{current_episode}"
        is_dump_watch_episode = (current_episode == 11)
        
        if (is_dump_watch_episode and 
            dump_key not in self.exhaustive_dump_triggered and
            self.live_feature_debugger.is_interesting_scenario(schema, self.env.state)):
            
            print(f"\n📸 EXHAUSTIVE DUMP: Interesting scenario detected in Episode {current_episode}")
            try:
                scenario_desc = self.live_feature_debugger.describe_scenario(schema, self.env.state)
                hand_log.append(f"\n{'='*60}\n🎯 SCENARIO: {scenario_desc}\n{'='*60}\n")
                
                context = {
                    'episode': current_episode,
                    'street': self._get_current_street_name(),
                    'stage': self.env.state.stage,
                    'pot': self.env.state.pot,
                    'stacks': self.env.state.stacks,
                    'scenario': scenario_desc,
                    'hand_history': hand_log
                }
                
                hand_log.append(dump_all_features_to_log(schema, f"P0({p0_role})-{scenario_desc}", context))
                print(f"✅ Exhaustive dump queued for scenario: {scenario_desc}")
                self.exhaustive_dump_triggered[dump_key] = True
            except Exception as e:
                print(f"⚠️ Failed to queue scenario dump: {e}")

    def _log_action(self, hand_log: List[str], current_player: int, action: int, amount: Optional[int], old_state: Dict):
        """Log the action taken by a player."""
        player_name = f"P{current_player}"
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
                current_commitment = old_state.get('current_bets', [0])[current_player]
                new_total = current_commitment + amount
                
                if max(old_state.get('current_bets', [0])) == 0:
                    action_str = f"bets {amount}"
                else:
                    action_str = f"raises by {amount} (from {current_commitment} to {new_total})"
            else:
                action_str = "raises"
        
        hand_log.append(f"  {player_name} {action_str}.")

    def _record_action_for_feature_extractor(self, current_player: int, action: int, amount: Optional[int], old_state: Dict):
        """Record action in ActionSequencer for feature extraction."""
        if action == 0:
            action_type = 'fold'
        elif action == 1:
            current_bets = old_state.get('current_bets', [])
            max_bet = max(current_bets) if current_bets else 0
            player_bet = current_bets[current_player] if current_player < len(current_bets) else 0
            to_call = max_bet - player_bet
            
            if to_call == 0:
                action_type = 'check'
            elif amount and amount > 0:
                action_type = 'call'
            else:
                action_type = None  # Skip phantom actions
        elif action == 2:
            current_bets = old_state.get('current_bets', [])
            max_bet = max(current_bets) if current_bets else 0
            action_type = 'bet' if max_bet == 0 else 'raise'
        else:
            action_type = None
        
        if action_type is not None:
            self.feature_extractor.record_action(current_player, action_type, amount or 0)

    def _handle_street_transition(self, current_stage: int, p0_role: str, stats_p0: dict, stats_p1: dict):
        """Handle street transition with strategic features."""
        street_that_just_ended = get_street_name(current_stage)
        
        # Get the final schema from Player 0's perspective with clear stats
        _, final_schema = self.feature_extractor.extract_features(
            self.env.state, 
            seat_id=0, 
            role=p0_role, 
            self_stats=stats_p0,      # P0's perspective: "self" is P0
            opponent_stats=stats_p1   # P0's perspective: "opponent" is P1
        )
        
        # Convert the strategic features object to a dict to be saved
        strategic_data_to_save = None
        if hasattr(final_schema, 'current_strategic') and final_schema.current_strategic:
            strategic_data_to_save = final_schema.current_strategic.to_dict()
        
        # Pass the strategic data to the snapshot saver
        self.feature_extractor.save_street_snapshot(
            self.env.state,
            strategic_features=strategic_data_to_save,
            street=street_that_just_ended
        )
        
        # Reset the ActionSequencer for the new street
        self.feature_extractor.new_street()

    def _finish_hand(self, hand_experiences: List[Dict], final_state: Dict, player_map: Optional[Dict],
                    hand_training_buffer: List[Dict], should_log_hand: bool, hand_log: Optional[List[str]],
                    current_episode: int, initial_stack: int):
        """Process data at the end of a hand."""
        # Complete hand analysis using HandEventIdentifier
        self.history_manager.finish_hand(final_state, player_map)
        
        # Process hand training buffer for range data
        br_hole_cards = self._get_hand_cards(1)
        for buffer_entry in hand_training_buffer:
            self._collect_range_training_data(buffer_entry['features'], br_hole_cards)
        
        # Record session tracking
        self.session_tracker.record_hand_winner(final_state.get('winners', []))
        
        # Update stack depth simulator
        if self.stack_depth_simulator:
            final_stacks = self.env.state.stacks.copy()
            self.stack_depth_simulator.advance_hand(final_stacks)
        
        # Process experiences with combined rewards
        hand_profit = self.env.state.stacks[0] - initial_stack
        for exp in hand_experiences:
            # Traditional end-of-hand reward
            profit_reward = hand_profit / 200.0 * self.profit_weight
            # Equity-based immediate reward
            equity_reward = exp.get('equity_reward', 0.0) * self.equity_weight
            # Combined reward
            exp['reward'] = profit_reward + equity_reward
            exp['reward_profit'] = profit_reward
            exp['reward_equity'] = equity_reward
            exp['episode'] = current_episode
            
            # Add stack depth information
            if self.stack_depth_simulator:
                session_info = self.stack_depth_simulator.get_current_session_info()
                exp['effective_stack'] = session_info['effective_stack']
                exp['session_hand'] = session_info['session_hand']
        
        # Write hand log if needed
        if should_log_hand and hand_log:
            # Add profit reward summary
            profit_reward_unweighted = hand_profit / 200.0
            hand_log.append(f"- HAND PROFIT: {hand_profit} chips (P_Rew: {profit_reward_unweighted:+.4f})")
            
            if self.env.state.winners:
                winner_str = "P0" if 0 in self.env.state.winners else "P1"
                # The profit for P0 is the amount won.
                pot_won = hand_profit if winner_str == "P0" else -hand_profit
                
                # Check if anyone actually won (pot could be split)
                if pot_won > 0:
                    hand_log.append(f"- HAND END: {winner_str} wins pot of {pot_won}.")
                else:
                    # Handle splits or no-profit scenarios
                    hand_log.append("- HAND END: Pot split or no profit.")
            
            try:
                with open("training_output/hand_histories.log", "a") as f:
                    f.write("\n".join(hand_log) + "\n\n")
            except:
                pass

    # === UTILITY METHODS ===
    
    def _get_hand_cards(self, player_id: int) -> List[str]:
        """Extract hand cards for a player from environment state."""
        try:
            if hasattr(self.env.state, 'hole_cards') and self.env.state.hole_cards:
                hand = self.env.state.hole_cards[player_id]
                if hand and len(hand) >= 2:
                    return [card_to_string(hand[0]), card_to_string(hand[1])]
            return ['??', '??']
        except Exception:
            return ['??', '??']

    def _get_board_cards(self) -> List[str]:
        """Extract board cards from environment state."""
        try:
            if hasattr(self.env.state, 'community') and self.env.state.community:
                return [card_to_string(card) for card in self.env.state.community]
            return []
        except Exception:
            return []

    def _get_current_street_name(self) -> str:
        """Get current street name from environment state."""
        return get_street_name(getattr(self.env.state, 'stage', 0))

    def _collect_range_training_data(self, features: List[float], hole_cards: List[str]):
        """Collect training data for range prediction network."""
        try:
            if len(hole_cards) != 2 or '??' in hole_cards:
                return
            
            # Convert card strings to ranks and determine if suited
            card1_rank = self._card_string_to_rank(hole_cards[0])
            card2_rank = self._card_string_to_rank(hole_cards[1])
            suited = hole_cards[0][1] == hole_cards[1][1]
            
            if card1_rank == -1 or card2_rank == -1:
                return
            
            # Get hand properties using the classification function
            hand_properties = classify_hand_properties(card1_rank, card2_rank, suited)
            
            # The 'features' vector is already clean and public-only from extract_public_features()
            # No sanitization is needed here anymore.
            training_sample = {
                'features': features,  # Use the features directly
                'hand_properties': hand_properties
            }
            
            # Add to buffer
            self.range_data_buffer.append(training_sample)
            
            # Write batch if buffer is full
            if len(self.range_data_buffer) >= self.range_data_batch_size:
                self._write_range_data_batch()
                
        except Exception:
            pass  # Silently skip on error

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
        except Exception:
            pass  # Silently skip on error

    def flush_range_data(self):
        """Flush any remaining range training data to file."""
        if self.range_data_buffer:
            self._write_range_data_batch()

    def reset_session_tracking(self):
        """Reset session tracking for new episode."""
        self.session_tracker.reset_episode()

    def print_episode_summary(self):
        """Print episode-level session win summary."""
        self.session_tracker.print_episode_summary()

