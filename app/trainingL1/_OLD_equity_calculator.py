# app/trainingL1/equity_calculator.py
# FINAL, REFACTORED VERSION - Clean deuces integration

import random
import itertools
import torch
import os
from typing import List, Tuple, Dict
from deuces import Card, Evaluator

try:
    from ..range_predictor.range_network import RangeNetwork
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from range_predictor.range_network import RangeNetwork


class EquityCalculator:
    """
    Calculates poker hand equity using a fast, professional-grade Monte Carlo simulation.
    """
    def __init__(self):
        # Professional hand evaluator using the 'deuces' library
        self.evaluator = Evaluator()

    def calculate_equity(self, my_hand: List[str], board: List[str],
                        opponent_range: Dict[Tuple[str, str], float],
                        num_simulations: int = 200) -> float:
        """
        Calculates hand equity using a streamlined Monte Carlo simulation with deuces.
        """
        if not opponent_range:
            return 0.5

        try:
            # 1. Convert all known cards to deuces format ONCE at the start.
            my_hand_deuces = [Card.new(c) for c in my_hand]
            board_deuces = [Card.new(c) for c in board]
        except ValueError:
            return 0.5  # Return default if card strings are invalid

        # 2. Create a full deuces deck and remove the known cards.
        # This is much cleaner than managing our own integer deck.
        used_cards = set(my_hand_deuces + board_deuces)
        
        # Create full deck manually since deuces doesn't expose Card.DECK
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        full_deck = [Card.new(rank + suit) for rank in ranks for suit in suits]
        deck = [c for c in full_deck if c not in used_cards]

        # 3. Prepare for weighted sampling from the opponent's range.
        hands_in_range = list(opponent_range.keys())
        weights = list(opponent_range.values())
        
        wins, ties, total_sims = 0, 0, 0

        for _ in range(num_simulations):
            # 4. Sample opponent's hand and convert to deuces format.
            opp_hand_str = random.choices(hands_in_range, weights=weights, k=1)[0]
            try:
                opp_hand_deuces = [Card.new(c) for c in opp_hand_str]
            except ValueError:
                continue

            # 5. Check for card conflicts and prepare the simulation deck.
            if any(c in used_cards for c in opp_hand_deuces):
                continue
            
            sim_deck = [c for c in deck if c not in opp_hand_deuces]

            # 6. Deal the runout DIRECTLY from the deuces deck. No more conversions!
            cards_needed = 5 - len(board_deuces)
            if len(sim_deck) < cards_needed:
                continue
            
            runout_deuces = random.sample(sim_deck, cards_needed)
            full_board_deuces = board_deuces + runout_deuces

            # 7. Evaluate with deuces.
            try:
                my_score = self.evaluator.evaluate(my_hand_deuces, full_board_deuces)
                opp_score = self.evaluator.evaluate(opp_hand_deuces, full_board_deuces)

                if my_score < opp_score:
                    wins += 1
                elif my_score == opp_score:
                    ties += 1
                total_sims += 1
            except:
                continue
            
        if total_sims == 0:
            return 0.5
            
        return (wins + ties * 0.5) / total_sims


class RangeConstructor:
    """
    Fast dynamic range constructor using opponent's live VPIP/PFR stats.
    Much more accurate than static ranges for exploitative play.
    """
    def __init__(self, action_selector=None, feature_extractor=None):
        # No longer need these for static version - included for compatibility
        self.action_selector = action_selector
        self.feature_extractor = feature_extractor
        
        # Pre-ranked list of all 169 starting hands (best to worst)
        # This is the core of the fast stats-based approach
        self.HAND_RANKINGS = self._create_hand_rankings()
        
        # Cache for hand string to card tuple conversion
        self._hand_cache = self._build_hand_cache()
        
        # Initialize static ranges for fallback
        self._init_static_ranges()
    
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
    
    def construct_range(self, action_history: List[Dict], current_board: List[str], 
                       current_pot: int, opponent_stats: Dict = None) -> Dict[Tuple[str, str], float]:
        """
        Advanced dynamic range construction using comprehensive opponent stats.
        
        Args:
            action_history: List of actions taken by opponent
            current_board: Current board state  
            current_pot: Current pot size
            opponent_stats: Optional dict with comprehensive poker stats
            
        Returns:
            Dictionary of {hand: weight} representing opponent's likely range
        """
        # NEW: Use comprehensive stats-based approach when available
        if opponent_stats and opponent_stats.get('sample_size', 0) >= 10:
            return self._construct_range_from_comprehensive_stats(
                action_history, current_board, opponent_stats
            )
        
        # LEGACY: Use basic VPIP/PFR approach for small samples
        elif opponent_stats and len(action_history) > 0:
            last_action = action_history[-1].get('action', 1)
            return self.construct_range_from_stats(opponent_stats, last_action)
        
        # FALLBACK: Use static ranges when no stats available
        board_stage = len(current_board)
        
        # Pre-flop Logic
        if board_stage == 0:
            # Simple BTN open-raise (most common scenario)
            if len(action_history) == 1 and action_history[0].get('action') == 2:
                return self.BTN_OPEN_RAISE_WEIGHTED
            
            # BB 3-bet (re-raise) scenario
            elif len(action_history) >= 2 and action_history[-1].get('action') == 2:
                # Tighter range for 3-betting
                return self._get_3bet_range()
            
            # BB calling/defending
            elif len(action_history) >= 1 and action_history[-1].get('action') == 1:
                return self.BB_DEFEND_WEIGHTED
            
            # Default to BTN opening range
            return self.BTN_OPEN_RAISE_WEIGHTED
        
        # Post-flop Logic (Flop = 3 cards, Turn = 4, River = 5)
        elif board_stage >= 3:
            # Determine if this looks like a continuation bet
            # (opponent was pre-flop aggressor and is betting)
            preflop_had_aggression = any(action.get('action') == 2 for action in action_history[:2])
            recent_bet = action_history and action_history[-1].get('action') == 2
            
            if preflop_had_aggression and recent_bet:
                # Looks like a continuation bet
                return self.FLOP_CBET_WEIGHTED
            
            elif action_history and action_history[-1].get('action') == 1:
                # Called/check-called
                return self.FLOP_CALL_WEIGHTED
            
            else:
                # Mixed/unclear action - use combined range
                return self._combine_ranges([
                    (self.FLOP_CBET_WEIGHTED, 0.4),
                    (self.FLOP_CALL_WEIGHTED, 0.6)
                ])
        
        # Default fallback
        return self.BTN_OPEN_RAISE_WEIGHTED
    
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
                 feature_dim: int = 184, fallback_constructor: RangeConstructor = None):
        """
        Initialize the NN Range Constructor.
        
        Args:
            model_path: Path to the trained range prediction model
            feature_dim: Expected feature vector dimension
            fallback_constructor: RangeConstructor to use if NN fails
        """
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.model = None
        self.device = torch.device('cpu')  # Use CPU for inference
        
        # Fallback to traditional range constructor if NN fails
        self.fallback_constructor = fallback_constructor or RangeConstructor()
        
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
    
    def construct_range(self, action_history: List[Dict], current_board: List[str],
                       current_pot: int, opponent_stats: Dict = None, 
                       opponent_features: List[float] = None) -> Dict[Tuple[str, str], float]:
        """
        Construct opponent range using neural network predictions.
        
        Args:
            action_history: List of actions taken by opponent
            current_board: Current board state
            current_pot: Current pot size
            opponent_stats: Opponent statistics (for fallback)
            opponent_features: Complete feature vector for opponent
            
        Returns:
            Dictionary of {hand: weight} representing opponent's likely range
        """
        # Use NN prediction if model is loaded and features are available
        if self.model is not None and opponent_features is not None:
            try:
                return self._construct_range_from_nn(opponent_features)
            except Exception as e:
                print(f"Error in NN range construction: {e}, falling back")
        
        # Fallback to traditional range construction
        return self.fallback_constructor.construct_range(
            action_history, current_board, current_pot, opponent_stats
        )
    
    def _construct_range_from_nn(self, features: List[float]) -> Dict[Tuple[str, str], float]:
        """
        Construct range using neural network predictions.
        
        Args:
            features: Complete feature vector for opponent
            
        Returns:
            Range dictionary with NN-predicted weights
        """
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions from the model
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        # Extract property probabilities
        properties = {}
        for prop_name, prob_tensor in predictions.items():
            properties[prop_name] = prob_tensor.squeeze().item()
        
        # Build range based on predicted properties
        return self._build_range_from_properties(properties)
    
    def _build_range_from_properties(self, properties: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """
        Build a range dictionary from predicted hand properties.
        
        Args:
            properties: Dictionary of hand property probabilities
            
        Returns:
            Range dictionary with appropriate weights
        """
        range_dict = {}
        
        # Start with baseline range (top 40% of hands)
        baseline_cutoff = int(len(self.HAND_RANKINGS) * 0.4)
        baseline_hands = self.HAND_RANKINGS[:baseline_cutoff]
        
        # Convert hand rankings to our format and apply NN adjustments
        for hand_str in baseline_hands:
            # Get all card combinations for this hand
            hand_combos = self._hand_cache.get(hand_str, [])
            
            for combo in hand_combos:
                # Determine base weight (stronger hands get higher weight)
                hand_index = self.HAND_RANKINGS.index(hand_str)
                base_weight = 1.0 - (hand_index / len(self.HAND_RANKINGS))
                
                # Apply NN property adjustments
                final_weight = self._adjust_weight_by_properties(
                    hand_str, base_weight, properties
                )
                
                # Only include hands with meaningful weight
                if final_weight > 0.1:
                    range_dict[combo] = final_weight
        
        # Normalize weights so they sum to reasonable total
        if range_dict:
            total_weight = sum(range_dict.values())
            if total_weight > 0:
                for hand in range_dict:
                    range_dict[hand] /= total_weight
                    range_dict[hand] *= len(range_dict) * 0.1  # Scale appropriately
        
        return range_dict
    
    def _adjust_weight_by_properties(self, hand_str: str, base_weight: float, 
                                   properties: Dict[str, float]) -> float:
        """
        Adjust hand weight based on NN-predicted properties.
        
        Args:
            hand_str: Hand string like 'AA', 'AKs', etc.
            base_weight: Base weight from hand strength
            properties: Predicted hand properties
            
        Returns:
            Adjusted weight for this hand
        """
        multiplier = 1.0
        
        # Check hand category and apply property multipliers
        if self._is_premium_pair(hand_str):
            multiplier *= (1.0 + properties.get('premium_pair', 0.0) * 2.0)
        elif self._is_mid_pair(hand_str):
            multiplier *= (1.0 + properties.get('mid_pair', 0.0) * 1.5)
        elif self._is_small_pair(hand_str):
            multiplier *= (1.0 + properties.get('small_pair', 0.0) * 1.2)
        elif self._is_suited_broadway(hand_str):
            multiplier *= (1.0 + properties.get('suited_broadway', 0.0) * 1.8)
        elif self._is_offsuit_broadway(hand_str):
            multiplier *= (1.0 + properties.get('offsuit_broadway', 0.0) * 1.4)
        elif self._is_suited_connector(hand_str):
            multiplier *= (1.0 + properties.get('suited_connector', 0.0) * 1.3)
        elif self._is_suited_ace(hand_str):
            multiplier *= (1.0 + properties.get('suited_ace', 0.0) * 1.1)
        else:
            # Bluff candidates or other hands
            multiplier *= (1.0 + properties.get('bluff_candidate', 0.0) * 0.8)
        
        return base_weight * multiplier
    
    def _is_premium_pair(self, hand_str: str) -> bool:
        """Check if hand is a premium pair (TT+)."""
        return hand_str in ['AA', 'KK', 'QQ', 'JJ', 'TT']
    
    def _is_mid_pair(self, hand_str: str) -> bool:
        """Check if hand is a mid pair (66-99)."""
        return hand_str in ['99', '88', '77', '66']
    
    def _is_small_pair(self, hand_str: str) -> bool:
        """Check if hand is a small pair (22-55)."""
        return hand_str in ['55', '44', '33', '22']
    
    def _is_suited_broadway(self, hand_str: str) -> bool:
        """Check if hand is suited broadway (AKs-QJs)."""
        return hand_str.endswith('s') and hand_str in ['AKs', 'AQs', 'AJs', 'KQs', 'KJs', 'QJs']
    
    def _is_offsuit_broadway(self, hand_str: str) -> bool:
        """Check if hand is offsuit broadway (AKo-QJo)."""
        return hand_str.endswith('o') and hand_str in ['AKo', 'AQo', 'AJo', 'KQo', 'KJo', 'QJo']
    
    def _is_suited_connector(self, hand_str: str) -> bool:
        """Check if hand is a suited connector."""
        if not hand_str.endswith('s') or len(hand_str) != 3:
            return False
        
        rank_map = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        rank1 = rank_map.get(hand_str[0], int(hand_str[0]) if hand_str[0].isdigit() else 0)
        rank2 = rank_map.get(hand_str[1], int(hand_str[1]) if hand_str[1].isdigit() else 0)
        
        return abs(rank1 - rank2) == 1 and min(rank1, rank2) >= 5
    
    def _is_suited_ace(self, hand_str: str) -> bool:
        """Check if hand is a suited ace (A5s-A2s)."""
        if not hand_str.endswith('s') or len(hand_str) != 3 or hand_str[0] != 'A':
            return False
        
        second_rank = hand_str[1]
        return second_rank in ['5', '4', '3', '2']
    
