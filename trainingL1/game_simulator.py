# trainingL1/game_simulator.py
# Helper classes for game simulation, reward shaping, and session management

from typing import Dict, List, Optional

# 1. Use relative imports (single dot) for files in the same package (trainingL1)
from .equity_calculator import EquityCalculator
from .range_constructors import RangeConstructor

# 2. Use absolute imports (from the project root) for files in other packages
from analyzers.event_identifier import HandEventIdentifier, HandHistory, RawAction, ActionType, Street


class HandHistoryManager:
    """
    Manages the state and creation of a HandHistory object for a single hand.
    Interfaces with the HandEventIdentifier to process the completed hand.
    """
    
    def __init__(self, stats_tracker, event_identifier: HandEventIdentifier, env):
        self.stats_tracker = stats_tracker
        self.event_identifier = event_identifier
        self.env = env
        self.reset()

    def reset(self):
        """Resets the state for a new hand."""
        self.current_hand_actions: List[RawAction] = []
        self.current_hand_id = ""
        self.current_starting_stacks: Dict[int, int] = {}
        self.current_dealer_position = 0
        self.current_community_cards: Dict[Street, List[int]] = {street: [] for street in Street}

    def start_hand(self, hand_num: int, current_episode: int):
        """Initializes tracking for a new hand."""
        self.reset()
        self.current_hand_id = f"ep{current_episode}_h{hand_num}"
        self.current_starting_stacks = dict(enumerate(self.env.state.stacks.copy()))
        self.current_dealer_position = getattr(self.env.state, 'dealer', 0)
        self.update_community_cards()

    def record_action(self, player_id: int, action: int, amount: int, state: Dict):
        """Records a raw action for later analysis by HandEventIdentifier."""
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
        self.update_community_cards()

    def update_community_cards(self):
        """Updates community cards tracking based on current environment state."""
        if not hasattr(self.env.state, 'community'):
            return
            
        board = self.env.state.community
        if len(board) == 0:
            return  # Preflop
        elif len(board) == 3:
            self.current_community_cards[Street.FLOP] = [self._card_string_to_id(self._card_to_string(card)) for card in board]
        elif len(board) == 4:
            self.current_community_cards[Street.TURN] = [self._card_string_to_id(self._card_to_string(card)) for card in board]
        elif len(board) == 5:
            self.current_community_cards[Street.RIVER] = [self._card_string_to_id(self._card_to_string(card)) for card in board]

    def finish_hand(self, final_state: Dict, player_map: Optional[Dict] = None):
        """Completes hand analysis using HandEventIdentifier and updates StatsTracker."""
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

    def _get_hand_cards(self, player_id: int) -> List[str]:
        """Extract hand cards for a player from environment state."""
        try:
            # Access hand from environment state
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
            
            return ['??', '??']  # Unknown cards
        except Exception:
            return ['??', '??']  # Fallback if hand access fails

    def _card_to_string(self, card_id: int) -> str:
        """Convert card ID to string representation."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]

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


class RewardShaper:
    """
    Calculates potential-based reward shaping (PBRS) using equity changes.
    Isolates the complex logic for calculating equity-based rewards.
    """
    
    def __init__(self, equity_calculator: EquityCalculator, range_constructor: RangeConstructor):
        self.equity_calculator = equity_calculator
        self.range_constructor = range_constructor

    def calculate_reward(self, player_id: int, action: int, amount: Optional[int],
                        old_state: Dict, new_state: Dict, 
                        game_state_before_action, action_sequencer,
                        opponent_public_features: List[float], 
                        opponent_stats: Dict) -> float:
        """
        Calculate immediate reward based on equity change.
        
        This is the core PBRS (Potential-Based Reward Shaping) implementation.
        """
        try:
            # Get hand and board
            my_hand = self._get_hand_cards(player_id, old_state)
            old_board = self._get_board_cards(old_state)
            
            # Pass the modernized arguments to the range constructor
            old_opponent_range = self.range_constructor.construct_range(
                public_features=opponent_public_features,
                game_state=game_state_before_action,
                action_sequencer=action_sequencer,
                opponent_stats=opponent_stats
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
                new_board = self._get_board_cards(new_state)
                
                # For simplicity, re-use the same opponent public features for new state
                # A more advanced implementation might re-calculate features for the new state
                new_opponent_range = self.range_constructor.construct_range(
                    public_features=opponent_public_features,
                    game_state=game_state_before_action,  # Re-use for now
                    action_sequencer=action_sequencer,
                    opponent_stats=opponent_stats
                )
                
                new_equity = self.equity_calculator.calculate_equity(
                    my_hand, new_board, new_opponent_range, num_simulations=200
                )
            
            # Calculate equity change
            equity_change = new_equity - old_equity
            
            # Scale by pot size (more important in bigger pots)
            pot_size = old_state.get('pot', 100)
            pot_multiplier = min(pot_size / 200.0, 3.0)  # Cap at 3x
            
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

    def _get_hand_cards(self, player_id: int, state: Dict) -> List[str]:
        """Extract hand cards for a player from state."""
        try:
            if 'hole_cards' in state and state['hole_cards']:
                hand = state['hole_cards'][player_id]
                if hand and len(hand) >= 2:
                    return [self._card_to_string(hand[0]), self._card_to_string(hand[1])]
            return ['??', '??']  # Unknown cards
        except Exception:
            return ['??', '??']

    def _get_board_cards(self, state: Dict) -> List[str]:
        """Extract board cards from state."""
        try:
            community = state.get('community', [])
            if community:
                return [self._card_to_string(card) for card in community]
            return []
        except Exception:
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


class SessionTracker:
    """
    Tracks wins and losses for sessions within an episode.
    A session is a continuous series of hands without a stack reset.
    """
    
    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        """Resets tracking for a new episode."""
        self.episode_session_wins = {'player_0': 0, 'player_1': 0}
        self.reset_session()

    def reset_session(self):
        """Resets tracking for a new session."""
        self.current_session_wins = {'player_0': 0, 'player_1': 0}
        self.session_hands = 0

    def record_hand_winner(self, winners: List[int]):
        """Records the winner of a single hand."""
        self.session_hands += 1
        if winners:
            if 0 in winners:
                self.current_session_wins['player_0'] += 1
            elif 1 in winners:
                self.current_session_wins['player_1'] += 1

    def end_session(self):
        """Counts the session winner and resets for the next session."""
        if self.session_hands == 0:
            return
            
        p0_wins = self.current_session_wins['player_0']
        p1_wins = self.current_session_wins['player_1']

        if p0_wins > p1_wins:
            self.episode_session_wins['player_0'] += 1
        elif p1_wins > p0_wins:
            self.episode_session_wins['player_1'] += 1
        # Tie sessions don't count for either player
        
        self.reset_session()

    def print_episode_summary(self):
        """Prints the episode-level session win summary."""
        total_sessions = self.episode_session_wins['player_0'] + self.episode_session_wins['player_1']
        if total_sessions > 0:
            print(f"🏆 Episode Sessions Won: P0={self.episode_session_wins['player_0']}, P1={self.episode_session_wins['player_1']} ({total_sessions} total)")
        else:
            print("🏆 Episode Sessions Won: No completed sessions")

