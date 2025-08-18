# trainingL1/range_constructors.py
# Range construction logic - both heuristic and neural network based

import torch
import os
import itertools
from typing import List, Tuple, Dict

from range_predictor.range_network import RangeNetwork


class RangeConstructor:
    """
    Fast dynamic range constructor using opponent's live VPIP/PFR stats.
    MODERNIZED to work with ActionSequencer.
    """
    def __init__(self):  # Simplified __init__
        # Pre-ranked list of all 169 starting hands (best to worst)
        self.HAND_RANKINGS = self._create_hand_rankings()
        
        # Cache for hand string to card tuple conversion
        self._hand_cache = self._build_hand_cache()
        
        # Initialize static ranges for fallback
        self._init_static_ranges()
    
    def _adapt_sequencer_to_history(self, game_state, action_sequencer) -> List[Dict]:
        """
        Adapter function to convert modern ActionSequencer output into the
        legacy List[Dict] format required by the other heuristic methods.
        """
        legacy_history = []
        action_log = action_sequencer.get_live_action_sequence()
        
        for seat_id, action_type, amount in action_log:
            action_code = 1  # Default to call
            if action_type == 'fold':
                action_code = 0
            elif action_type in ['bet', 'raise']:
                action_code = 2
            
            legacy_history.append({
                'action': action_code,
                'stage': game_state.stage,
                'seat_id': seat_id
            })
        return legacy_history
    
    def _create_hand_rankings(self) -> List[str]:
        """Create pre-ranked list of all 169 starting hands (best to worst)."""
        # Professional hand ranking based on equity vs random hand
        rankings = [
            # Pocket pairs (high to low)
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22',
            
            # Suited aces (high to low)
            'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
            
            # Offsuit aces (high to low)
            'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
            
            # Suited kings
            'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s',
            
            # Offsuit kings  
            'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o', 'K4o', 'K3o', 'K2o',
            
            # Suited queens
            'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s',
            
            # Offsuit queens
            'QJo', 'QTo', 'Q9o', 'Q8o', 'Q7o', 'Q6o', 'Q5o', 'Q4o', 'Q3o', 'Q2o',
            
            # Suited jacks
            'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s',
            
            # Offsuit jacks  
            'JTo', 'J9o', 'J8o', 'J7o', 'J6o', 'J5o', 'J4o', 'J3o', 'J2o',
            
            # Suited tens
            'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s',
            
            # Offsuit tens
            'T9o', 'T8o', 'T7o', 'T6o', 'T5o', 'T4o', 'T3o', 'T2o',
            
            # Suited nines
            '98s', '97s', '96s', '95s', '94s', '93s', '92s',
            
            # Offsuit nines
            '98o', '97o', '96o', '95o', '94o', '93o', '92o',
            
            # Suited eights
            '87s', '86s', '85s', '84s', '83s', '82s',
            
            # Offsuit eights
            '87o', '86o', '85o', '84o', '83o', '82o',
            
            # Suited sevens
            '76s', '75s', '74s', '73s', '72s',
            
            # Offsuit sevens
            '76o', '75o', '74o', '73o', '72o',
            
            # Suited sixes
            '65s', '64s', '63s', '62s',
            
            # Offsuit sixes
            '65o', '64o', '63o', '62o',
            
            # Suited fives
            '54s', '53s', '52s',
            
            # Offsuit fives
            '54o', '53o', '52o',
            
            # Suited fours
            '43s', '42s',
            
            # Offsuit fours
            '43o', '42o',
            
            # Suited threes
            '32s',
            
            # Offsuit threes (worst hand)
            '32o'
        ]
        return rankings
    
    def _build_hand_cache(self) -> Dict[str, List[Tuple[str, str]]]:
        """Pre-compute all possible card combinations for each hand string."""
        cache = {}
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        suits = ['s', 'h', 'd', 'c']
        
        for hand_str in self.HAND_RANKINGS:
            if len(hand_str) == 2:  # Pocket pair like 'AA'
                rank = hand_str[0]
                combinations = []
                for i, suit1 in enumerate(suits):
                    for suit2 in suits[i+1:]:
                        combinations.append((rank + suit1, rank + suit2))
                cache[hand_str] = combinations
                
            elif hand_str.endswith('s'):  # Suited like 'AKs'
                rank1, rank2 = hand_str[0], hand_str[1]
                combinations = []
                for suit in suits:
                    combinations.append((rank1 + suit, rank2 + suit))
                cache[hand_str] = combinations
                
            elif hand_str.endswith('o'):  # Offsuit like 'AKo'
                rank1, rank2 = hand_str[0], hand_str[1]
                combinations = []
                for suit1 in suits:
                    for suit2 in suits:
                        if suit1 != suit2:
                            combinations.append((rank1 + suit1, rank2 + suit2))
                cache[hand_str] = combinations
                
        return cache
    
    def construct_range_from_stats(self, opponent_stats: Dict, last_action: int = 1) -> Dict[Tuple[str, str], float]:
        """
        Fast stats-based range construction using VPIP/PFR.
        
        Args:
            opponent_stats: Dict with 'vpip' and 'pfr' percentages (0.0 to 1.0)
            last_action: 0=fold, 1=call, 2=raise
            
        Returns:
            Dictionary of {(card1, card2): weight} representing opponent's range
        """
        vpip = opponent_stats.get('vpip', 0.5)  # Default 50% VPIP
        pfr = opponent_stats.get('pfr', 0.2)    # Default 20% PFR
        
        # Determine which slice of hands to use
        if last_action == 2:  # Opponent raised
            # Use top PFR% of hands
            hands_to_include = max(1, int(len(self.HAND_RANKINGS) * pfr))
            selected_hands = self.HAND_RANKINGS[:hands_to_include]
        elif last_action == 1:  # Opponent called
            # Use the "calling range" - between PFR and VPIP
            pfr_cutoff = max(1, int(len(self.HAND_RANKINGS) * pfr))
            vpip_cutoff = max(pfr_cutoff + 1, int(len(self.HAND_RANKINGS) * vpip))
            selected_hands = self.HAND_RANKINGS[pfr_cutoff:vpip_cutoff]
        else:  # Opponent folded (shouldn't happen in equity calc, but handle gracefully)
            # Use a tight range (top 20% of hands)
            hands_to_include = max(1, int(len(self.HAND_RANKINGS) * 0.2))
            selected_hands = self.HAND_RANKINGS[:hands_to_include]
        
        # Convert hand strings to actual card combinations with uniform weights
        final_range = {}
        for hand_str in selected_hands:
            if hand_str in self._hand_cache:
                for card_combo in self._hand_cache[hand_str]:
                    final_range[card_combo] = 1.0  # Uniform weight within the range
        
        return final_range
    
    def _construct_range_from_comprehensive_stats(self, action_history: List[Dict], 
                                                 current_board: List[str], 
                                                 opponent_stats: Dict) -> Dict[Tuple[str, str], float]:
        """
        Advanced range construction using comprehensive poker statistics.
        
        This method analyzes the opponent's action in context of their overall tendencies
        to build a much more accurate range than basic VPIP/PFR.
        """
        board_stage = len(current_board)
        
        if not action_history:
            # No actions yet - use pre-flop opening range based on VPIP
            vpip = opponent_stats.get('vpip', 0.5)
            hands_to_include = max(1, int(len(self.HAND_RANKINGS) * vpip))
            selected_hands = self.HAND_RANKINGS[:hands_to_include]
            
        elif board_stage == 0:  # Pre-flop
            return self._construct_preflop_range(action_history, opponent_stats)
            
        else:  # Post-flop (flop, turn, river)
            return self._construct_postflop_range(action_history, current_board, opponent_stats)
        
        # Convert to final range format
        final_range = {}
        for hand_str in selected_hands:
            if hand_str in self._hand_cache:
                for card_combo in self._hand_cache[hand_str]:
                    final_range[card_combo] = 1.0
        
        return final_range
    
    def _construct_preflop_range(self, action_history: List[Dict], 
                                opponent_stats: Dict) -> Dict[Tuple[str, str], float]:
        """Construct pre-flop range based on opponent's action and tendencies."""
        last_action = action_history[-1].get('action', 1)
        
        vpip = opponent_stats.get('vpip', 0.5)
        pfr = opponent_stats.get('pfr', 0.2)
        three_bet = opponent_stats.get('three_bet', 0.1)
        
        # Determine if this is a 3-bet situation
        num_raises = sum(1 for action in action_history if action.get('action') == 2)
        
        if last_action == 2:  # Opponent raised/bet
            if num_raises == 1:  # First raise (open)
                # Use PFR range
                hands_to_include = max(1, int(len(self.HAND_RANKINGS) * pfr))
            elif num_raises >= 2:  # 3-bet or higher
                # Use tighter 3-bet range
                hands_to_include = max(1, int(len(self.HAND_RANKINGS) * three_bet))
            else:
                hands_to_include = max(1, int(len(self.HAND_RANKINGS) * pfr))
                
            selected_hands = self.HAND_RANKINGS[:hands_to_include]
            
        elif last_action == 1:  # Opponent called
            # Use calling range (between PFR and VPIP)
            pfr_cutoff = max(1, int(len(self.HAND_RANKINGS) * pfr))
            vpip_cutoff = max(pfr_cutoff + 1, int(len(self.HAND_RANKINGS) * vpip))
            selected_hands = self.HAND_RANKINGS[pfr_cutoff:vpip_cutoff]
            
        else:  # Opponent folded (shouldn't happen in equity calc)
            selected_hands = []
        
        # Convert to final range
        final_range = {}
        for hand_str in selected_hands:
            if hand_str in self._hand_cache:
                for card_combo in self._hand_cache[hand_str]:
                    final_range[card_combo] = 1.0
        
        return final_range
    
    def _construct_postflop_range(self, action_history: List[Dict], 
                                 current_board: List[str], 
                                 opponent_stats: Dict) -> Dict[Tuple[str, str], float]:
        """Construct post-flop range using advanced statistics."""
        board_stage = len(current_board)
        last_action = action_history[-1].get('action', 1) if action_history else 1
        
        # Start with pre-flop range
        preflop_actions = [a for a in action_history if a.get('stage', 0) == 0]
        base_range_hands = self._get_preflop_base_range(preflop_actions, opponent_stats)
        
        # Adjust based on post-flop tendencies
        if last_action == 2:  # Opponent bet/raised
            adjustment_factor = self._get_postflop_aggression_factor(board_stage, opponent_stats)
        elif last_action == 1:  # Opponent called
            adjustment_factor = self._get_postflop_calling_factor(board_stage, opponent_stats)
        else:  # Opponent folded
            adjustment_factor = 0.3  # Very tight range
        
        # Apply adjustment to range size
        adjusted_size = max(1, int(len(base_range_hands) * adjustment_factor))
        
        # For betting: use stronger portion of range
        # For calling: use wider portion of range
        if last_action == 2:  # Betting - use strongest hands
            selected_hands = base_range_hands[:adjusted_size]
        else:  # Calling - use wider range including weaker hands
            selected_hands = base_range_hands[:adjusted_size]
        
        # Convert to final range
        final_range = {}
        for hand_str in selected_hands:
            if hand_str in self._hand_cache:
                for card_combo in self._hand_cache[hand_str]:
                    final_range[card_combo] = 1.0
        
        return final_range
    
    def _get_preflop_base_range(self, preflop_actions: List[Dict], 
                               opponent_stats: Dict) -> List[str]:
        """Get the base pre-flop range that opponent likely had."""
        if not preflop_actions:
            vpip = opponent_stats.get('vpip', 0.5)
            hands_to_include = max(1, int(len(self.HAND_RANKINGS) * vpip))
            return self.HAND_RANKINGS[:hands_to_include]
        
        # Use the same logic as pre-flop range construction
        last_preflop_action = preflop_actions[-1].get('action', 1)
        
        if last_preflop_action == 2:  # Raised pre-flop
            pfr = opponent_stats.get('pfr', 0.2)
            hands_to_include = max(1, int(len(self.HAND_RANKINGS) * pfr))
        else:  # Called pre-flop
            vpip = opponent_stats.get('vpip', 0.5) 
            pfr = opponent_stats.get('pfr', 0.2)
            pfr_cutoff = max(1, int(len(self.HAND_RANKINGS) * pfr))
            vpip_cutoff = max(pfr_cutoff + 1, int(len(self.HAND_RANKINGS) * vpip))
            return self.HAND_RANKINGS[pfr_cutoff:vpip_cutoff]
        
        return self.HAND_RANKINGS[:hands_to_include]
    
    def _get_postflop_aggression_factor(self, board_stage: int, 
                                       opponent_stats: Dict) -> float:
        """Calculate how much of their range opponent bets with on this street."""
        
        if board_stage == 3:  # Flop
            cbet_freq = opponent_stats.get('cbet_flop', 0.6)
            agg_freq = opponent_stats.get('aggression_frequency', 0.3)
            return max(cbet_freq, agg_freq)  # Use higher of the two
            
        elif board_stage == 4:  # Turn  
            cbet_freq = opponent_stats.get('cbet_turn', 0.5)
            agg_freq = opponent_stats.get('aggression_frequency', 0.3)
            return max(cbet_freq, agg_freq)
            
        elif board_stage == 5:  # River
            cbet_freq = opponent_stats.get('cbet_river', 0.4)
            agg_freq = opponent_stats.get('aggression_frequency', 0.3)
            return max(cbet_freq, agg_freq)
        
        return 0.4  # Default
    
    def _get_postflop_calling_factor(self, board_stage: int, 
                                    opponent_stats: Dict) -> float:
        """Calculate how much of their range opponent calls with on this street."""
        
        # Calling ranges are typically wider than betting ranges
        base_calling_factor = 0.7  # Start with 70% of range
        
        # Adjust based on fold-to-c-bet tendencies
        if board_stage == 3:  # Flop
            fold_to_cbet = opponent_stats.get('fold_to_cbet_flop', 0.5)
            calling_factor = base_calling_factor * (1.0 - fold_to_cbet)
            
        elif board_stage == 4:  # Turn
            fold_to_cbet = opponent_stats.get('fold_to_cbet_turn', 0.6) 
            calling_factor = base_calling_factor * (1.0 - fold_to_cbet)
            
        elif board_stage == 5:  # River
            fold_to_cbet = opponent_stats.get('fold_to_cbet_river', 0.7)
            calling_factor = base_calling_factor * (1.0 - fold_to_cbet)
        else:
            calling_factor = base_calling_factor
        
        # Use WTSD (Went To Showdown) to adjust for calling station tendencies
        wtsd = opponent_stats.get('wtsd', 0.25)
        if wtsd > 0.35:  # Calling station
            calling_factor *= 1.3  # Wider calling range
        elif wtsd < 0.15:  # Tight player
            calling_factor *= 0.7  # Narrower calling range
        
        return min(calling_factor, 1.0)  # Cap at 100%
    
    def _init_static_ranges(self):
        """Initialize the static ranges (moved from __init__ to avoid unreachable code)."""
        # Heads-up specific ranges with proper weighting
        # BTN (Small Blind) ranges - very wide due to positional advantage
        self.BTN_OPEN_RAISE_WEIGHTED = {
            # Premium hands - always raise
            ('As', 'Ad'): 1.0, ('Ks', 'Kd'): 1.0, ('Qs', 'Qd'): 1.0, ('Js', 'Jd'): 1.0,
            ('Ts', 'Td'): 1.0, ('9s', '9d'): 1.0, ('8s', '8d'): 1.0, ('7s', '7d'): 1.0,
            
            # Strong suited hands - high frequency
            ('As', 'Ks'): 0.95, ('As', 'Qs'): 0.9, ('As', 'Js'): 0.85, ('As', 'Ts'): 0.8,
            ('Ks', 'Qs'): 0.85, ('Ks', 'Js'): 0.8, ('Qs', 'Js'): 0.75,
            
            # Suited connectors - medium frequency
            ('Js', 'Ts'): 0.7, ('Ts', '9s'): 0.65, ('9s', '8s'): 0.6, ('8s', '7s'): 0.55,
            
            # Offsuit broadway - lower frequency
            ('As', 'Kh'): 0.8, ('As', 'Qh'): 0.7, ('Ks', 'Qh'): 0.6,
            
            # Suited aces - bluff hands
            ('As', '5s'): 0.4, ('As', '4s'): 0.35, ('As', '3s'): 0.3, ('As', '2s'): 0.25
        }
        
        # BB defending range vs BTN raise - also wide due to pot odds
        self.BB_DEFEND_WEIGHTED = {
            # Premium hands - always defend (call/3bet)
            ('As', 'Ad'): 1.0, ('Ks', 'Kd'): 1.0, ('Qs', 'Qd'): 1.0, ('Js', 'Jd'): 1.0,
            
            # Strong hands - high defend frequency
            ('As', 'Ks'): 0.9, ('As', 'Qs'): 0.85, ('Ks', 'Qs'): 0.8,
            ('Ts', 'Td'): 0.9, ('9s', '9d'): 0.85, ('8s', '8d'): 0.8,
            
            # Suited connectors - good for calling
            ('Js', 'Ts'): 0.8, ('Ts', '9s'): 0.75, ('9s', '8s'): 0.7,
            ('8s', '7s'): 0.65, ('7s', '6s'): 0.6,
            
            # Weak pairs - decent defending hands
            ('7s', '7d'): 0.75, ('6s', '6d'): 0.7, ('5s', '5d'): 0.65,
            
            # Suited aces - reasonable defense
            ('As', 'Ts'): 0.7, ('As', '9s'): 0.6, ('As', '8s'): 0.5
        }
        
        # Post-flop continuation betting range (when BTN bets flop)
        self.FLOP_CBET_WEIGHTED = {
            # Strong made hands - always bet
            ('As', 'Ad'): 1.0, ('Ks', 'Kd'): 1.0, ('Qs', 'Qd'): 1.0,
            
            # Medium hands - often bet
            ('Js', 'Jd'): 0.8, ('Ts', 'Td'): 0.75, ('9s', '9d'): 0.7,
            
            # Draws and bluffs - sometimes bet
            ('As', 'Ks'): 0.6, ('Js', 'Ts'): 0.5, ('As', '5s'): 0.3
        }
        
        # Calling range vs flop bet
        self.FLOP_CALL_WEIGHTED = {
            # Strong hands that don't want to fold
            ('As', 'Qs'): 0.9, ('Ks', 'Qs'): 0.85, ('Qs', 'Js'): 0.8,
            
            # Draws worth continuing with
            ('Js', 'Ts'): 0.75, ('Ts', '9s'): 0.7, ('9s', '8s'): 0.65,
            
            # Pairs that might be good
            ('9s', '9d'): 0.8, ('8s', '8d'): 0.75, ('7s', '7d'): 0.7
        }
    
    def construct_range(self, game_state, action_sequencer, opponent_stats: Dict, public_features: List[float] = None) -> Dict[Tuple[str, str], float]:
        """
        Modernized entry point for heuristic range construction.
        
        Args:
            game_state: Complete GameState object with all game information
            action_sequencer: Modern ActionSequencer with current street actions
            opponent_stats: Opponent statistics from StatsTracker
            
        Returns:
            Dictionary of {hand: weight} representing opponent's likely range
        """
        # 1. Adapt the new inputs to the old format internally
        adapted_history = self._adapt_sequencer_to_history(game_state, action_sequencer)
        
        # Convert community cards to strings (add import at top if needed)
        current_board = []
        if hasattr(game_state, 'community') and game_state.community:
            try:
                # Need to import card_to_string or implement it here
                current_board = [self._card_to_string(c) for c in game_state.community]
            except:
                # Fallback: simple conversion
                current_board = [str(c) for c in game_state.community]
        
        # 2. Call the original, powerful heuristic logic with adapted data
        if opponent_stats and opponent_stats.get('sample_size', 0) >= 10:
            return self._construct_range_from_comprehensive_stats(
                adapted_history, current_board, opponent_stats
            )
        elif opponent_stats and len(adapted_history) > 0:
            last_action = adapted_history[-1].get('action', 1)
            return self.construct_range_from_stats(opponent_stats, last_action)
        else:
            # Fallback to a basic VPIP range if no other info is available
            vpip = opponent_stats.get('vpip', 0.5) if opponent_stats else 0.5
            return self.construct_range_from_stats({'vpip': vpip, 'pfr': 0.2}, 1)
    
    def _card_to_string(self, card_id: int) -> str:
        """Convert card ID to string representation for board processing."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
        
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _get_3bet_range(self) -> Dict[Tuple[str, str], float]:
        """Return a tighter range for 3-betting scenarios."""
        return {
            # Premium hands for 3-betting
            ('As', 'Ad'): 1.0, ('Ks', 'Kd'): 1.0, ('Qs', 'Qd'): 1.0, ('Js', 'Jd'): 1.0,
            ('As', 'Ks'): 0.9, ('As', 'Qs'): 0.8,
            
            # Some bluff 3-bets with suited aces
            ('As', '5s'): 0.3, ('As', '4s'): 0.25, ('As', '3s'): 0.2
        }
    
    def _combine_ranges(self, weighted_ranges: List[Tuple[Dict, float]]) -> Dict[Tuple[str, str], float]:
        """Combine multiple weighted ranges."""
        combined = {}
        
        for range_dict, weight in weighted_ranges:
            for hand, hand_weight in range_dict.items():
                if hand in combined:
                    combined[hand] = max(combined[hand], hand_weight * weight)
                else:
                    combined[hand] = hand_weight * weight
                    
        return combined

class RangeConstructorNN:
    """
    Neural Network-based Range Constructor.
    
    Uses a trained neural network to predict opponent hand properties
    and builds accurate ranges based on these predictions.
    """
    
    def __init__(self, model_path: str = 'range_predictor/range_predictor.pt', 
                 feature_dim: int = 184):
        """
        Initialize the NN Range Constructor with modernized heuristic fallback.
        
        Args:
            model_path: Path to the trained range prediction model
            feature_dim: Expected feature vector dimension
        """
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.model = None
        self.device = torch.device('cpu')  # Use CPU for inference
        
        # The fallback is now a clean, modern component
        self.fallback_constructor = RangeConstructor()
        
        # Hand rankings for range construction
        self.HAND_RANKINGS = self.fallback_constructor.HAND_RANKINGS
        self._hand_cache = self.fallback_constructor._hand_cache
        
        # Load the trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained range prediction model."""
        try:
            if os.path.exists(self.model_path):
                # Load model checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Create model with correct architecture
                self.model = RangeNetwork(input_dim=self.feature_dim)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                print(f"Loaded range prediction model from {self.model_path}")
            else:
                print(f"Range prediction model not found at {self.model_path}, using fallback")
                self.model = None
        except Exception as e:
            print(f"Error loading range prediction model: {e}, using fallback")
            self.model = None
    
    def construct_range(self, game_state, action_sequencer, opponent_stats: Dict, public_features: List[float] = None) -> Dict[Tuple[str, str], float]:
        """
        Uses NN if available, otherwise falls back to the modernized heuristic constructor.
        
        Args:
            public_features: Complete feature vector for opponent (for NN)
            game_state: Complete GameState object
            action_sequencer: Modern ActionSequencer with current street actions
            opponent_stats: Opponent statistics from StatsTracker
            
        Returns:
            Dictionary of {hand: weight} representing opponent's likely range
        """
        # PRIMARY STRATEGY: Use the Neural Network if it's ready
        if self.model is not None and public_features is not None:
            try:
                return self._construct_range_from_nn(public_features)
            except Exception as e:
                print(f"⚠️ NN range construction failed: {e}. Falling back to heuristics.")
        
        # FALLBACK STRATEGY: Use heuristics for bootstrapping or if NN fails
        # The fallback call is now clean and uses modern components
        return self.fallback_constructor.construct_range(
            game_state, action_sequencer, opponent_stats, public_features
        )
    
    def _construct_range_from_nn(self, features: List[float]) -> Dict[Tuple[str, str], float]:
        """
        Construct range using the predicted hand embedding from the neural network.
        """
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get the predicted embedding vector from the model
        predicted_embedding = self.model.predict_hand_embedding(features_tensor)
        
        # Convert the tensor to a dictionary for easier use
        embedding_dims = [
            'pair_value', 'high_card_value', 'low_card_value', 'suitedness', 
            'connectivity', 'broadway_potential', 'wheel_potential', 'mid_strength_potential'
        ]
        properties = {dim: value.item() for dim, value in zip(embedding_dims, predicted_embedding)}
        
        # Build range based on the predicted embedding properties
        return self._build_range_from_properties(properties)
    
    def _build_range_from_properties(self, properties: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """
        Build a range dictionary by scoring each hand against the predicted embedding.
        """
        range_dict = {}
        
        # Use a wide baseline of hands to score against
        baseline_cutoff = int(len(self.HAND_RANKINGS) * 0.8) # Use top 80% to not miss bluffs
        baseline_hands = self.HAND_RANKINGS[:baseline_cutoff]
        
        for hand_str in baseline_hands:
            hand_combos = self._hand_cache.get(hand_str, [])
            
            # Get the true, objective embedding for this hand_str
            true_embedding = self._get_true_embedding_for_hand(hand_str)
            
            # Calculate the "distance" or "similarity" between the predicted and true embeddings.
            # A lower distance means the hand is a better fit for the predicted range.
            # We use (1 - distance) as a weight, so higher similarity = higher weight.
            distance = self._calculate_embedding_distance(properties, true_embedding)
            weight = max(0, 1.0 - distance)
            
            if weight > 0.15: # Only include hands that are a reasonably good match
                for combo in hand_combos:
                    range_dict[combo] = weight
        
        return range_dict
    
    # --- NEW HELPER METHODS ---
    
    def _get_true_embedding_for_hand(self, hand_str: str) -> Dict[str, float]:
        """
        Generates the objective embedding for a given hand string (e.g., 'AKs').
        This re-uses the logic from your classify_hand_properties function.
        """
        # Note: This requires access to the classification logic.
        from range_predictor.range_dataset import classify_hand_properties
        
        rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
        
        rank1 = rank_map[hand_str[0]]
        rank2 = rank_map[hand_str[1]]
        suited = len(hand_str) == 3 and hand_str[2] == 's'
        
        return classify_hand_properties(rank1, rank2, suited)

    def _calculate_embedding_distance(self, pred_props: Dict[str, float], true_props: Dict[str, float]) -> float:
        """
        Calculates the weighted Mean Squared Error distance between two embeddings.
        """
        distance = 0.0
        # Give more weight to important properties like pairs and suitedness
        weights = {'pair_value': 2.0, 'suitedness': 1.5, 'high_card_value': 1.2}
        
        for key in pred_props:
            pred = pred_props[key]
            true = true_props[key]
            weight = weights.get(key, 1.0)
            distance += weight * ((pred - true) ** 2)
            
        return distance / len(pred_props)
    
