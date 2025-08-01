# analyzers/strategic_analyzer.py

import copy
import random
from feature_contexts import StaticContext, DynamicContext
from trainingL1.equity_calculator import EquityCalculator, RangeConstructorNN
from .feature_utils import sanitize_features

class StrategicAnalyzer:
    """
    Calculates high-level, range-dependent strategic features using a trained
    RangePredictorNN. This is the core "brain" for advanced concepts.
    """
    def __init__(self, range_constructor_nn: RangeConstructorNN, equity_calculator: EquityCalculator):
        """
        Initializes the analyzer with the necessary tools.
        
        Args:
            range_constructor_nn: An instance of the TRAINED NN-based range constructor.
            equity_calculator: An instance of the EquityCalculator.
        """
        self.range_constructor = range_constructor_nn
        self.equity_calculator = equity_calculator

    def calculate_features(self, self_context: dict, opponent_context: dict) -> dict:
        """
        The main entry point to calculate all strategic features.
        Receives two context objects containing all necessary information.
        """
        # Step 1: Predict the range for the opponent.
        opponent_range = self._predict_opponent_range(opponent_context)
        
        # Step 2: Predict the self's range as perceived by the opponent.
        self_perceived_range = self._predict_self_perceived_range(self_context)

        # Step 3: Use these ranges to calculate advanced metrics.
        equity_vs_range = self._calculate_hand_vs_range_equity(self_context, opponent_range)
        
        range_vs_range_equity = self._calculate_range_vs_range_equity(
            self_perceived_range, opponent_range, self_context.get('board_strings', [])
        )
        
        implied_odds = self._calculate_implied_odds(self_context, opponent_range, self_perceived_range)
        
        fold_equity = self._calculate_fold_equity(self_perceived_range, opponent_range, self_context)

        return {
            "equity_vs_range": equity_vs_range,
            "range_vs_range_equity": range_vs_range_equity,
            "implied_odds": implied_odds,
            "fold_equity": fold_equity,
        }

    def _predict_opponent_range(self, opponent_context: dict) -> dict:
        """
        Predicts the opponent's range using their public features.
        No sanitization is needed because we don't know their private cards.
        """
        return self.range_constructor.construct_range(
            action_history=opponent_context['action_history'],
            current_board=opponent_context['board_strings'],
            current_pot=opponent_context['pot'],
            opponent_stats=opponent_context['stats'],
            opponent_features=opponent_context['features']
        )

    def _predict_self_perceived_range(self, self_context: dict) -> dict:
        """
        Predicts the self's range from the opponent's perspective by hiding
        the self's private hole card information.
        """
        # Sanitize the self's feature vector to remove private hand info
        sanitized_features = sanitize_features(self_context['features'], purpose='perceived_range')
        
        # Call the range constructor with the self's public actions and SANITIZED features
        return self.range_constructor.construct_range(
            action_history=self_context['action_history'],
            current_board=self_context['board_strings'],
            current_pot=self_context['pot'],
            opponent_stats=self_context['stats'],
            opponent_features=sanitized_features
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
    
    def _calculate_hand_vs_range_equity(self, self_context: dict, opponent_range: dict) -> float:
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
        
    def _calculate_equity_vs_range(self, hole_cards: list, community_cards: list, 
                                  opponent_stats: dict = None, opponent_action_history: list = None,
                                  pot_size: int = 100) -> float:
        """
        Calculate equity vs opponent's estimated range.
        """
        # Fallback to hand_strength if equity system is not available
        if not self.equity_calculator or not self.range_constructor:
            return self._calculate_hand_strength(hole_cards, community_cards)
        
        # Convert card IDs to string format for equity calculator
        try:
            hole_strings = [self._card_id_to_string(card) for card in hole_cards[:2]]
            board_strings = [self._card_id_to_string(card) for card in community_cards]
            
            # Use opponent stats and action history to construct their range
            if opponent_stats and opponent_action_history:
                opponent_range = self.range_constructor.construct_range(
                    opponent_action_history, board_strings, pot_size, opponent_stats
                )
            else:
                # Fallback to basic stats-based range
                default_stats = {'vpip': 0.5, 'pfr': 0.2}  # Default loose-passive opponent
                last_action = 1  # Assume opponent called
                if opponent_action_history and len(opponent_action_history) > 0:
                    last_action = opponent_action_history[-1].get('action', 1)
                
                opponent_range = self.range_constructor.construct_range_from_stats(
                    default_stats, last_action
                )
            
            # Calculate equity using the range
            if opponent_range:
                equity = self.equity_calculator.calculate_equity(
                    hole_strings, board_strings, opponent_range, num_simulations=100
                )
                return equity
            else:
                # No valid range constructed, fallback to hand strength
                return self._calculate_hand_strength(hole_cards, community_cards)
                
        except Exception:
            # Any error in equity calculation, fallback to hand strength
            return self._calculate_hand_strength(hole_cards, community_cards)

    def _calculate_implied_odds(self, self_context: dict, opponent_range: dict, self_perceived_range: dict) -> float:
        """
        Calculates implied odds by estimating future street profitability.
        Considers both ranges and potential future betting.
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

        # Estimate outs based on current hand vs range equity
        current_equity = self._calculate_hand_vs_range_equity(self_context, opponent_range)
        
        # Simple heuristic: if we have low equity now but draws available,
        # we have implied odds potential
        if current_equity > 0.4:  # Already strong, limited implied odds
            return current_equity * 0.1
            
        # Estimate improvement potential (simplified)
        board_length = len(self_context.get('board_strings', []))
        cards_to_come = (5 - board_length) if board_length < 5 else 0
        
        if cards_to_come == 0:
            return 0.0
            
        # Rough estimation of improvement probability
        improvement_prob = (1 - current_equity) * 0.3  # 30% chance to improve significantly
        
        # Estimate potential payoff from opponent's remaining stack
        stacks = getattr(static_ctx, 'stacks', [200, 200])
        effective_stack = min(stacks) - to_call
        estimated_payoff = effective_stack * 0.4  # Conservative estimate
        
        # Calculate implied odds ratio
        implied_value = improvement_prob * estimated_payoff
        implied_odds = implied_value / to_call if to_call > 0 else 0.0
        
        # Normalize to reasonable range
        return min(implied_odds / 3.0, 1.0)

    def _calculate_fold_equity(self, self_perceived_range: dict, opponent_range: dict, self_context: dict) -> float:
        """
        Calculates fold equity - the probability that opponent folds to a bet
        times the pot size. Uses range analysis to estimate folding frequency.
        """
        static_ctx = self_context.get('static_ctx')
        if not static_ctx:
            return 0.0
            
        pot = getattr(static_ctx, 'pot', 100)
        
        # Estimate opponent's folding frequency based on their range strength
        # If opponent has a tight range, they're less likely to fold
        if not opponent_range:
            return 0.3 * pot  # Default estimate
            
        # Simple heuristic: count strong vs weak hands in opponent's range
        strong_hands = sum(1 for weight in opponent_range.values() if weight > 0.7)
        total_hands = len(opponent_range)
        
        if total_hands == 0:
            fold_probability = 0.3
        else:
            # More weak hands = higher fold probability
            weak_hand_ratio = (total_hands - strong_hands) / total_hands
            fold_probability = min(weak_hand_ratio * 0.6, 0.8)  # Cap at 80%
        
        return fold_probability * pot
    
    def _calculate_current_implied_odds(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Calculate current implied odds using proper board texture analysis.
        Integrates with board analyzer for full calculation.
        """
        # CRITICAL FIX: Implied odds must be 0.0 on river (no more cards coming)
        if static_ctx.stage == 3:  # River
            return 0.0
        
        current_bets = static_ctx.game_state.current_bets
        max_bet = max(current_bets) if current_bets else 0
        my_bet = current_bets[static_ctx.seat_id]
        to_call = max_bet - my_bet
        pot = static_ctx.pot
        
        if to_call <= 0:
            return 0.0
        
        # Basic pot odds
        pot_odds = to_call / (pot + to_call)
        
        # Advanced future betting potential calculation
        our_stack = static_ctx.game_state.stacks[static_ctx.seat_id]
        active_players = [i for i in range(static_ctx.num_players) 
                         if static_ctx.game_state.stacks[i] > 0 and i != static_ctx.seat_id]
        
        if not active_players:
            return pot_odds
        
        # Calculate effective stack sizes (what we can actually win)
        effective_stacks = []
        for opp_id in active_players:
            opp_stack = static_ctx.game_state.stacks[opp_id]
            effective_stack = min(our_stack, opp_stack)  # Can only win what both players have
            effective_stacks.append(effective_stack)
        
        # Future betting potential based on effective stacks and position
        dealer_pos = static_ctx.game_state.dealer_pos
        relative_position = (static_ctx.seat_id - dealer_pos) % static_ctx.num_players
        position_multiplier = 1.0 + (0.3 * (1.0 - relative_position / static_ctx.num_players))
        
        # Stack-to-pot ratio consideration (SPR)
        avg_effective_stack = sum(effective_stacks) / len(effective_stacks)
        spr = avg_effective_stack / max(pot, 1)
        
        # SPR-based future betting potential
        if spr >= 4:  # Deep stacks - high implied odds potential
            stack_factor = min(3.0, 1.5 + spr * 0.2)
        elif spr >= 2:  # Medium stacks - moderate potential
            stack_factor = 1.0 + spr * 0.25
        else:  # Shallow stacks - low potential
            stack_factor = 0.5 + spr * 0.25
        
        # Board texture consideration for draw potential using board analyzer
        draw_potential = 1.0
        community = static_ctx.community
        if len(community) >= 3:
            board_texture = self.board_analyzer.analyze_texture(community)
            
            # Check for flush draws (2-3 cards of same suit)
            has_flush_draw = any([
                2 <= board_texture.spades_cards_present <= 3,
                2 <= board_texture.hearts_cards_present <= 3,
                2 <= board_texture.clubs_cards_present <= 3,
                2 <= board_texture.diamonds_cards_present <= 3
            ])
            
            # Check for straight draws (look for patterns with 2-4 cards)
            has_straight_draw = any([
                2 <= board_texture.A5_cards_present <= 4,
                2 <= board_texture.S26_cards_present <= 4,
                2 <= board_texture.S37_cards_present <= 4,
                2 <= board_texture.S48_cards_present <= 4,
                2 <= board_texture.S59_cards_present <= 4,
                2 <= board_texture.S6T_cards_present <= 4,
                2 <= board_texture.S7J_cards_present <= 4,
                2 <= board_texture.S8Q_cards_present <= 4,
                2 <= board_texture.S9K_cards_present <= 4,
                2 <= board_texture.TA_cards_present <= 4
            ])
            
            # Check for made hands (flushes/straights)
            has_made_flush = any([
                board_texture.spades_cards_present >= 3,
                board_texture.hearts_cards_present >= 3,
                board_texture.clubs_cards_present >= 3,
                board_texture.diamonds_cards_present >= 3
            ])
            
            has_made_straight = any([
                board_texture.A5_cards_present >= 3,
                board_texture.S26_cards_present >= 3,
                board_texture.S37_cards_present >= 3,
                board_texture.S48_cards_present >= 3,
                board_texture.S59_cards_present >= 3,
                board_texture.S6T_cards_present >= 3,
                board_texture.S7J_cards_present >= 3,
                board_texture.S8Q_cards_present >= 3,
                board_texture.S9K_cards_present >= 3,
                board_texture.TA_cards_present >= 3
            ])
            
            # More draws possible = higher implied odds potential
            if has_flush_draw or has_straight_draw:
                draw_potential *= 1.4
            elif has_made_flush or has_made_straight:
                draw_potential *= 0.7  # Less implied odds when board is scary
        
        # Street-based adjustment (earlier streets have more implied odds potential)
        stage = static_ctx.stage
        street_multiplier = max(0.5, 1.3 - (stage * 0.2))  # Decreases each street
        
        # Opponent count factor (more opponents = more potential)
        opponent_factor = min(1.5, 1.0 + len(active_players) * 0.1)
        
        # Calculate final implied odds
        future_betting_factor = (stack_factor * position_multiplier * draw_potential * 
                               street_multiplier * opponent_factor)
        
        implied_odds = pot_odds * min(future_betting_factor, 4.0)  # Cap at 4x pot odds
        return min(implied_odds, 0.9)  # Cap at 90%
    
    def _calculate_enhanced_implied_odds(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, opponent_stats: dict = None) -> float:
        """
        Enhanced implied odds calculation that considers opponent's range and tendencies.
        """
        # CRITICAL FIX: Implied odds must be 0.0 on river (no more cards coming)
        if static_ctx.stage == 3:  # River
            return 0.0
        
        # Start with the existing calculation
        base_implied_odds = self._calculate_current_implied_odds(static_ctx, dynamic_ctx)
        
        # If no opponent stats, return base calculation
        if not opponent_stats:
            return base_implied_odds
        
        # Adjust based on opponent's calling tendencies (WTSD = Went To Showdown)
        wtsd = opponent_stats.get('wtsd', 0.25)
        range_factor = 1.0
        
        if wtsd > 0.35:  # Calling station - higher implied odds
            range_factor = 1.25
        elif wtsd < 0.20:  # Tight player - lower implied odds
            range_factor = 0.75
        
        # Also consider opponent's aggression frequency
        aggr_freq = opponent_stats.get('aggression_frequency', 0.3)
        if aggr_freq > 0.4:  # Aggressive opponent - higher implied odds potential
            range_factor *= 1.1
        elif aggr_freq < 0.2:  # Passive opponent - lower implied odds
            range_factor *= 0.9
        
        return min(base_implied_odds * range_factor, 0.9)  # Cap at 90%
    
