# analyzers/strategic_analyzer.py

import copy
import random
from feature_contexts import StaticContext, DynamicContext

# Direct imports - no fallback to avoid circular imports
from trainingL1.equity_calculator import EquityCalculator
from trainingL1.range_constructors import RangeConstructorNN

from .feature_utils import sanitize_features

class StrategicAnalyzer:
    """
    Calculates high-level, range-dependent strategic features using a trained
    RangePredictorNN. This is the core "brain" for advanced concepts.
    """
    def __init__(self, range_constructor, equity_calculator: EquityCalculator):
        """
        Initializes the analyzer with the necessary tools.
        
        Args:
            range_constructor: A RangeConstructor instance (either basic or NN-based with fallback).
            equity_calculator: An instance of the EquityCalculator.
        """
        self.range_constructor = range_constructor
        self.equity_calculator = equity_calculator

    def calculate_features(self, self_context: dict, opponent_context: dict) -> dict:
        """
        The main entry point to calculate all strategic features.
        Receives two context objects containing all necessary information.
        """
        # Step 1: Predict the "base" range for the opponent.
        base_opponent_range = self._predict_opponent_range(opponent_context)
        
        # Step 2: *** CARD REMOVAL PRUNING ***
        # Remove impossible hands from opponent's range based on our known cards
        opponent_range = self._prune_range_for_card_removal(
            base_opponent_range, self_context.get('hand_strings', [])
        )
        
        # Step 3: Predict the self's range as perceived by the opponent.
        self_perceived_range = self._predict_self_perceived_range(self_context)

        # Step 4: Calculate the one essential equity feature (always needed)
        hand_vs_range = self._calculate_hand_vs_range(self_context, opponent_range)

        # --- START FIX ---
        # Only calculate the ultra-expensive features if the NN is trained and active.
        # Otherwise, default them to 0.0 for the bootstrapping phase.
        if hasattr(self.range_constructor, 'model') and self.range_constructor.model is not None:
            range_vs_range_equity = self._calculate_range_vs_range_equity(
                self_perceived_range, opponent_range, self_context.get('board_strings', [])
            )
            implied_odds = self._calculate_implied_odds(self_context, opponent_range, self_perceived_range)
            fold_equity = self._calculate_fold_equity(self_perceived_range, opponent_range, self_context)
            reverse_implied_odds = self._calculate_reverse_implied_odds(self_context, opponent_range)
            playability = self._calculate_playability(self_context, self_perceived_range)
            future_payoff = self._calculate_future_payoff(self_context, opponent_range)
        else:
            # Default values during heuristic bootstrapping
            range_vs_range_equity = 0.5
            implied_odds = 0.0
            fold_equity = 0.0
            reverse_implied_odds = 0.0
            playability = 0.0
            future_payoff = 0.0
        # --- END FIX ---
        
        # Calculate equity delta (change from previous street)
        # equity_delta = self._calculate_equity_delta(self_context, hand_vs_range)
        
        return {
            "implied_odds": implied_odds,
            "hand_vs_range": hand_vs_range,           # Keep for backward compatibility
            "fold_equity": fold_equity,
            "reverse_implied_odds": reverse_implied_odds,
            "range_vs_range": range_vs_range_equity,      # Alias for compatibility
            "future_payoff": future_payoff,
            "playability": playability
        }

    def _predict_opponent_range(self, opponent_context: dict) -> dict:
        """Predicts the opponent's range using the MODERNIZED constructor."""
        # The context dictionary, created by FeatureExtractor, has all the needed info.
        return self.range_constructor.construct_range(
            game_state=opponent_context.get('static_ctx').game_state,
            action_sequencer=opponent_context.get('action_sequencer'),
            opponent_stats=opponent_context.get('opponent_stats'),
            public_features=opponent_context.get('public_features')
        )
    
    def _predict_self_perceived_range(self, self_context: dict) -> dict:
        """Predicts the self's range using the MODERNIZED constructor."""
        return self.range_constructor.construct_range(
            game_state=self_context.get('static_ctx').game_state,
            action_sequencer=self_context.get('action_sequencer'),
            opponent_stats=self_context.get('self_stats'), # Note: we use self_stats here
            public_features=self_context.get('public_features')
        )
    
    def _card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to string representation like '2s'."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _prune_range_for_card_removal(self, player_range: dict, known_cards: list) -> dict:
        """
        Remove impossible hands from a range based on known cards.
        This is the crucial card removal step that makes ranges combinatorially accurate.
        
        Args:
            player_range: Dict of {('card1', 'card2'): weight}
            known_cards: List of card strings we know are taken (e.g., ['Ks', 'Qh'])
            
        Returns:
            Pruned range with impossible combinations removed and weights renormalized
        """
        if not player_range or not known_cards:
            return player_range
        
        # Convert known cards to a set for fast lookup
        known_card_set = set(known_cards)
        
        # Filter out impossible hands
        pruned_range = {}
        for hand_tuple, weight in player_range.items():
            # Check if any card in this hand conflicts with known cards
            if not any(card in known_card_set for card in hand_tuple):
                pruned_range[hand_tuple] = weight
        
        # Renormalize weights so they sum to the same total as before
        if pruned_range:
            old_total = sum(player_range.values())
            new_total = sum(pruned_range.values())
            
            if new_total > 0:
                normalization_factor = old_total / new_total
                for hand_tuple in pruned_range:
                    pruned_range[hand_tuple] *= normalization_factor
        
        return pruned_range
    
    def _analyze_range_composition(self, player_range: dict, board: list) -> dict:
        """
        Analyzes a range against a board and returns its composition.
        This is the KEY function that turns a list of hands into strategic insights.
        
        Args:
            player_range: Dict of {('card1', 'card2'): weight}
            board: List of community card strings
            
        Returns:
            Dict like {'monsters': 0.15, 'strong_made': 0.25, 'draws': 0.30, 'bluffs': 0.30}
        """
        composition = {
            'monsters': 0.0,        # 85%+ equity (nuts, strong two-pair+)
            'strong_made': 0.0,     # 65-85% equity (strong pairs, weaker two-pair)
            'medium_made': 0.0,     # 50-65% equity (medium pairs, weak two-pair)
            'draws': 0.0,           # 35-50% equity (flush draws, straight draws)
            'bluffs': 0.0           # <35% equity (air, weak draws)
        }
        
        if not player_range:
            return composition
                
        # --- START FIX: Define a more realistic baseline range to compare against ---
        # This represents a typical, average range of hands an opponent might have.
        baseline_comparison_range = {
            ('As', 'Ks'): 1.0, ('Qs', 'Js'): 1.0, ('Ts', '9s'): 1.0, ('Ad', 'Qd'): 1.0,
            ('Ah', 'Kh'): 1.0, ('Qh', 'Jh'): 1.0, ('Th', '9h'): 1.0, ('Ac', 'Qc'): 1.0,
            ('As', 'Ad'): 1.0, ('Ks', 'Kd'): 1.0, ('Qs', 'Qd'): 1.0, ('Js', 'Jd'): 1.0,
            ('Ac', 'Kc'): 1.0, ('Qc', 'Jc'): 1.0, ('Tc', '9c'): 1.0, ('As', 'Qc'): 1.0,
        }
        # --- END FIX ---

        total_weight = 0.0
        
        for hand_tuple, weight in player_range.items():
            # Convert hand tuple to list for equity calculator
            hand_list = list(hand_tuple)
            
            # Calculate this hand's equity vs a realistic baseline range
            try:
                # --- START FIX: Use the baseline range for comparison ---
                equity = self.equity_calculator.calculate_equity(
                    hand_list, board, baseline_comparison_range, num_simulations=50
                )
                # --- END FIX ---
            except:
                # Fallback: estimate equity based on hand strength heuristics
                equity = self._estimate_hand_equity(hand_list, board)
            
            # Categorize based on equity
            if equity >= 0.85:
                composition['monsters'] += weight
            elif equity >= 0.65:
                composition['strong_made'] += weight  
            elif equity >= 0.50:
                composition['medium_made'] += weight
            elif equity >= 0.35:
                composition['draws'] += weight
            else:
                composition['bluffs'] += weight
            
            total_weight += weight
        
        # Normalize to percentages
        if total_weight > 0:
            for category in composition:
                composition[category] /= total_weight
        
        return composition
    
    def _estimate_hand_equity(self, hand: list, board: list) -> float:
        """
        Simple fallback equity estimation when equity calculator isn't available.
        Based on basic hand strength heuristics.
        """
        if len(hand) != 2:
            return 0.5
        
        # Very basic heuristic - in a real implementation, this would be more sophisticated
        hand_ranks = []
        for card in hand:
            if len(card) >= 1:
                rank_char = card[0]
                rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                hand_ranks.append(rank_map.get(rank_char, 7))
        
        if len(hand_ranks) == 2:
            avg_rank = sum(hand_ranks) / 2
            # Normalize to 0-1 range, with higher ranks having higher equity
            base_equity = (avg_rank - 2) / 12  # 2-14 -> 0-1
            
            # Adjust for pairs (same rank = higher equity)
            if hand_ranks[0] == hand_ranks[1]:
                base_equity += 0.2
            
            # Adjust for suited (same suit = higher equity) 
            if len(hand) == 2 and len(hand[0]) == 2 and len(hand[1]) == 2:
                if hand[0][1] == hand[1][1]:  # Same suit
                    base_equity += 0.1
            
            return min(max(base_equity, 0.1), 0.9)
        
        return 0.5
    
    def _calculate_hand_vs_range(self, self_context: dict, opponent_range: dict) -> float:
        """
        Calculates standard showdown equity (Hand vs. Range). This is the function
        that was previously in CurrentStreetAnalyzer.
        """
        # Extract self's actual hand and board from context
        static_ctx = self_context.get('static_ctx')
        if not static_ctx or not hasattr(static_ctx, 'hole_cards'):
            return 0.5  # Default equity if no hand info
            
        self_hand_str = [self._card_id_to_string(c) for c in static_ctx.hole_cards]
        board_str = [self._card_id_to_string(c) for c in static_ctx.community]
        
        return self.equity_calculator.calculate_equity(
            self_hand_str, board_str, opponent_range, num_simulations=200
        )

    def _calculate_range_vs_range_equity(self, self_range: dict, opponent_range: dict, board_strings: list) -> float:
        """
        Calculates how the self's entire perceived range fares against the opponent's range.
        Advanced metric that considers both players' range distributions.
        """
        if not self_range or not opponent_range:
            return 0.5

        total_equity = 0.0
        matchups = 0

        # Sample hands from both ranges for simulation
        self_sample = random.sample(list(self_range.keys()), k=min(10, len(self_range)))
        opp_sample = random.sample(list(opponent_range.keys()), k=min(10, len(opponent_range)))

        for h_hand_tuple in self_sample:
            for o_hand_tuple in opp_sample:
                # Ensure no card conflicts
                if h_hand_tuple[0] in o_hand_tuple or h_hand_tuple[1] in o_hand_tuple:
                    continue
                
                # Use the existing EquityCalculator for the matchup
                self_hand_list = list(h_hand_tuple)
                opp_range_single = {o_hand_tuple: 1.0}  # Single hand as range
                
                matchup_equity = self.equity_calculator.calculate_equity(
                    self_hand_list, board_strings, opp_range_single, num_simulations=50
                )
                total_equity += matchup_equity
                matchups += 1
        
        return total_equity / matchups if matchups > 0 else 0.5
        
    def _calculate_implied_odds(self, self_context: dict, opponent_range: dict, self_perceived_range: dict) -> float:
        """
        NEW RANGE-DRIVEN IMPLIED ODDS CALCULATION
        Uses opponent range composition to estimate realistic payoff potential.
        """
        static_ctx = self_context.get('static_ctx')
        if not static_ctx:
            return 0.0
            
        # On river, implied odds are always zero
        if hasattr(static_ctx, 'stage') and static_ctx.stage == 3:
            return 0.0
            
        pot = getattr(static_ctx, 'pot', 100)
        to_call = getattr(static_ctx, 'to_call', 0)
        if to_call <= 0:
            return 0.0

        # Get the board for range analysis
        board = self_context.get('board_strings', [])
        
        # Analyze opponent's range composition on this board
        opp_composition = self._analyze_range_composition(opponent_range, board)
        
        # The more strong hands they have, the more likely they are to pay us off if we hit
        # Strong hands will call future bets, weak hands will fold
        payoff_likelihood = (
            opp_composition['monsters'] * 1.0 +      # Monsters always pay off  
            opp_composition['strong_made'] * 0.8 +   # Strong hands usually pay off
            opp_composition['medium_made'] * 0.4 +   # Medium hands sometimes pay off
            opp_composition['draws'] * 0.2 +         # Draws rarely pay off (they might fold)
            opp_composition['bluffs'] * 0.1          # Bluffs almost never pay off
        )
        
        # Get our current equity to understand improvement potential
        current_equity = self._calculate_hand_vs_range(self_context, opponent_range)
        
        # If we already have strong equity, limited implied odds potential
        if current_equity > 0.6:
            return current_equity * 0.1
        
        # Calculate potential future payoff based on effective stacks
        stacks = getattr(static_ctx, 'stacks', [200, 200])
        effective_stack = min(stacks) - to_call
        
        # Estimate how much we can realistically win if we improve
        potential_gain = effective_stack * payoff_likelihood
        
        # Factor in our improvement probability
        # If we have low equity now, we have more room to improve
        improvement_potential = (1 - current_equity) * 0.35  # 35% chance to improve significantly
        
        # Calculate implied odds ratio
        implied_value = potential_gain * improvement_potential
        implied_odds = implied_value / max(to_call, 1)
        
        # Normalize to 0-1 range
        return min(implied_odds / 10.0, 1.0)

    def _calculate_fold_equity(self, self_perceived_range: dict, opponent_range: dict, self_context: dict) -> float:
        """
        NEW RANGE-DRIVEN FOLD EQUITY CALCULATION
        Uses opponent range composition to estimate realistic folding frequencies.
        """
        static_ctx = self_context.get('static_ctx')
        if not static_ctx:
            return 0.0
            
        pot = getattr(static_ctx, 'pot', 100)
        board = self_context.get('board_strings', [])
        
        # Estimate opponent's folding frequency based on their range strength
        if not opponent_range:
            return 0.3 * pot  # Default estimate
            
        # Analyze opponent's range composition on this specific board
        opp_composition = self._analyze_range_composition(opponent_range, board)
        
        # Different hand categories have different folding frequencies to bets
        fold_probability = (
            opp_composition['monsters'] * 0.05 +        # Monsters almost never fold (5%)
            opp_composition['strong_made'] * 0.15 +     # Strong hands rarely fold (15%)
            opp_composition['medium_made'] * 0.35 +     # Medium hands fold sometimes (35%)
            opp_composition['draws'] * 0.60 +           # Draws fold often (60%)
            opp_composition['bluffs'] * 0.85            # Bluffs fold very often (85%)
        )
        
        # Adjust based on bet sizing context (if we have that info)
        # Larger bets typically generate more folds
        bet_size_factor = 1.0  # Default, could be enhanced with actual bet sizing info
        
        # Adjust based on position and street
        # Later streets typically have lower fold frequencies
        stage = getattr(static_ctx, 'stage', 0)
        street_factor = max(0.7, 1.2 - (stage * 0.15))  # Decreases each street
        
        # Calculate final fold equity
        adjusted_fold_prob = min(fold_probability * bet_size_factor * street_factor, 0.9)
        
        return adjusted_fold_prob * pot
    
    def _calculate_reverse_implied_odds(self, self_context: dict, opponent_range: dict) -> float:
        """
        NEW RANGE-DRIVEN REVERSE IMPLIED ODDS CALCULATION
        Estimates the risk of losing a large pot on future streets even if we improve.
        High risk occurs on draw-heavy boards when our hand is good but not the nuts,
        and the opponent's range contains many better draws.
        """
        static_ctx = self_context.get('static_ctx')
        board_strings = self_context.get('board_strings', [])
        
        # No reverse implied odds on the river
        if not static_ctx or static_ctx.stage == 3:
            return 0.0

        # 1. Analyze Board Texture: Is it "wet" (dangerous)?
        suits_on_board = [card[1] for card in board_strings if len(card) >= 2]
        is_flush_draw_possible = any(suits_on_board.count(s) >= 2 for s in 'shdc')
        
        ranks_on_board = [card[0] for card in board_strings if len(card) >= 1]
        # Simplified straight check - look for connectedness
        rank_values = []
        for rank_char in ranks_on_board:
            if rank_char == 'A':
                rank_values.append(14)
            elif rank_char == 'K':
                rank_values.append(13)
            elif rank_char == 'Q':
                rank_values.append(12)
            elif rank_char == 'J':
                rank_values.append(11)
            elif rank_char == 'T':
                rank_values.append(10)
            else:
                rank_values.append(int(rank_char))
        
        rank_values.sort()
        is_straight_draw_possible = False
        if len(rank_values) >= 2:
            # Check for gaps of 4 or less between ranks
            for i in range(len(rank_values) - 1):
                if rank_values[i+1] - rank_values[i] <= 4:
                    is_straight_draw_possible = True
                    break

        board_danger = 0.0
        if is_flush_draw_possible: 
            board_danger += 0.4
        if is_straight_draw_possible: 
            board_danger += 0.3

        # 2. Analyze Opponent's Range: Are they likely on a draw?
        opp_composition = self._analyze_range_composition(opponent_range, board_strings)
        opponent_draw_heavy = opp_composition['draws']

        # 3. Analyze Our Hand: Is it good, but vulnerable?
        our_equity = self._calculate_hand_vs_range(self_context, opponent_range)
        is_vulnerable_made_hand = 0.60 < our_equity < 0.85  # Good, but not a monster

        # Combine factors: Risk is highest when the board is dangerous, our hand is
        # vulnerable, and the opponent is likely on a draw that beats ours.
        risk_factor = board_danger * opponent_draw_heavy
        if is_vulnerable_made_hand:
            risk_factor *= 1.5  # Amplify risk if we have a hand we're likely to continue with

        return min(risk_factor, 1.0)

    def _calculate_playability(self, self_context: dict, self_perceived_range: dict) -> float:
        """
        NEW RANGE-DRIVEN PLAYABILITY CALCULATION
        Estimates how easy a hand is to play post-flop. High-card, suited,
        and connected hands that fit well with our perceived range are more playable.
        """
        hand_strings = self_context.get('hand_strings', [])
        if not hand_strings or len(hand_strings) < 2:
            return 0.0

        # 1. Intrinsic Properties of the Hand
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        rank1 = rank_map.get(hand_strings[0][0], 7)
        rank2 = rank_map.get(hand_strings[1][0], 7)
        
        is_suited = len(hand_strings[0]) >= 2 and len(hand_strings[1]) >= 2 and hand_strings[0][1] == hand_strings[1][1]
        is_connected = abs(rank1 - rank2) <= 4  # Includes gappers
        avg_rank = (rank1 + rank2) / 2.0

        playability = 0.0
        if is_suited: 
            playability += 0.3
        if is_connected: 
            playability += 0.2
        playability += (avg_rank / 14.0) * 0.5  # High cards are easier to play

        # 2. Cohesion with Perceived Range
        # How well does our actual hand fit the story we're telling?
        if self_perceived_range:
            # Check if our hand (in either order) is in the top half of our perceived range
            sorted_range = sorted(self_perceived_range.items(), key=lambda item: item[1], reverse=True)
            top_half_hands = {hand for hand, weight in sorted_range[:len(sorted_range)//2]}
            
            # Create tuple representation of our hand (normalized order)
            my_hand_tuple = tuple(sorted(hand_strings))
            if my_hand_tuple in top_half_hands:
                playability += 0.2  # Bonus for being a credible part of our range
        
        return min(playability, 1.0)
    
    def _calculate_future_payoff(self, self_context: dict, opponent_range: dict) -> float:
        """
        NEW RANGE-DRIVEN FUTURE PAYOFF CALCULATION
        Estimates the potential money to be won on future streets.
        This is a key component of implied odds - isolates just the reward potential.
        """
        static_ctx = self_context.get('static_ctx')
        board_strings = self_context.get('board_strings', [])
        
        # There's no future payoff on the river
        if not static_ctx or static_ctx.stage == 3:
            return 0.0

        # 1. Determine the maximum amount you can possibly win (effective stack)
        stacks = getattr(static_ctx, 'stacks', [200, 200])
        to_call = getattr(static_ctx, 'to_call', 0)
        effective_stack = min(stacks) - to_call
        if effective_stack <= 0:
            return 0.0

        # 2. Analyze the opponent's range to see how likely they are to pay off
        # An opponent with monsters/strong hands is very likely to call future bets
        opp_composition = self._analyze_range_composition(opponent_range, board_strings)
        
        payoff_likelihood = (
            opp_composition['monsters'] * 1.0 +      # Monsters always call/raise
            opp_composition['strong_made'] * 0.8 +   # Strong hands usually call
            opp_composition['medium_made'] * 0.5 +   # Medium hands might call one bet
            opp_composition['draws'] * 0.3 +         # Draws might call if they have equity
            opp_composition['bluffs'] * 0.1          # Bluffs rarely call (but might bluff-raise)
        )

        # 3. Calculate the expected future value
        potential_gain = effective_stack * payoff_likelihood
        
        # 4. Factor in street-based multiplier (more streets = more betting rounds)
        streets_remaining = 3 - static_ctx.stage  # preflop=3, flop=2, turn=1, river=0
        street_multiplier = 1.0 + (streets_remaining * 0.2)  # More streets = more betting potential
        
        # 5. Normalize the result (as a fraction of a 100 BB starting stack)
        starting_stack = getattr(static_ctx, 'starting_stack', 200)
        
        normalized_payoff = (potential_gain * street_multiplier) / starting_stack
        
        return min(normalized_payoff, 1.0)
    
    def _calculate_equity_delta(self, self_context: dict, current_equity: float) -> float:
        """Calculate change in equity from previous street."""
        static_ctx = self_context.get('static_ctx')
        if not static_ctx:
            return 0.0
            
        current_stage = static_ctx.stage
        
        # No delta for pre-flop
        if current_stage == 0:
            return 0.0
        
        # Get history tracker from context if available
        history_tracker = self_context.get('history_tracker')
        if not history_tracker:
            return 0.0
        
        # Get previous street name
        previous_streets = {1: 'preflop', 2: 'flop', 3: 'turn'}
        previous_street = previous_streets.get(current_stage)
        
        if not previous_street:
            return 0.0
        
        # Look up previous equity from history
        hand_key = f"hand_{history_tracker.get_hand_number()}"
        previous_equity = history_tracker.get_feature_value(
            hand_key, previous_street, "hand_vs_range", current_equity
        )
        
        return current_equity - previous_equity
    
