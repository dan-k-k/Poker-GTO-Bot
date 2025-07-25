# analyzers/current_street_analyzer.py
# Current street features that don't rare yore history tracking
# All features are opponent reproducible - pass different seat_id for hero vs opponent

from feature_contexts import StaticContext, DynamicContext
from .action_sequencer import ActionSequencer
from .board_analyzer import BoardAnalyzer

# Import for hand strength and equity calculations
import sys
import os
import random
from poker_core import HandEvaluator

# Add trainingL1 to path for equity calculator imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trainingL1'))
try:
    from trainingL1.equity_calculator import EquityCalculator, RangeConstructor
except ImportError:
    try:
        from equity_calculator import EquityCalculator, RangeConstructor
    except ImportError:
        # Fallback - equity_vs_range will be set to hand_strength
        EquityCalculator = None
        RangeConstructor = None


class CurrentStreetAnalyzer:
    """
    Analyzer for current street features that don't need history tracking.
    All methods take seat_id parameter for opponent reproducibility.
    """
    
    def __init__(self):
        self.board_analyzer = BoardAnalyzer()  # For texture analysis (implied odds calculation)
        
        # Initialize equity calculation components
        if EquityCalculator and RangeConstructor:
            self.equity_calculator = EquityCalculator()
            self.range_constructor = RangeConstructor()
        else:
            self.equity_calculator = None
            self.range_constructor = None
            
        # Cache for hand strength calculations
        self.hand_strength_cache = {}
    
    def calculate_current_street_sequence(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext, action_sequencer: ActionSequencer) -> dict:
        """
        Calculate current street sequence features for any seat_id using ActionSequencer.
        
        Features:
        - Seat id checked x times
        - Seat id raised (includes initial bet) x times
        - Seat id raised by x% of pot on average
        - Seat id overbet (>100% of pot raise/bet) x times
        - Seat id largebet (>70% of pot raise/bet) x times
        - Seat id is currently facing a check, bet, raise(2bet), 3bet, 4betplus [all monotonic]
        """
        current_bets = static_ctx.game_state.current_bets
        pot = static_ctx.pot
        max_bet = max(current_bets) if current_bets else 0
        seat_bet = current_bets[seat_id]
        to_call = max_bet - seat_bet
        
        # Get actual action sequence from ActionSequencer
        action_sequence = action_sequencer.get_live_action_sequence()
        
        # Count actual actions for this seat
        checked_count = 0.0
        raised_count = 0.0
        total_raise_amount = 0.0
        overbet_count = 0.0
        largebet_count = 0.0
        
        # NEW: Better strategic features
        total_raise_pct_of_pot = 0.0  # Sum of (raise_amount / pot_before_raise)
        total_bet_to_starting_pot = 0.0  # Sum of (total_bet / starting_pot_this_street)
        
        # NEW: Advanced strategic features
        did_check_raise = 0.0  # Player checked then raised on same street
        did_donk_bet = 0.0     # Player bet out of position when not aggressor
        did_3bet = 0.0         # Player made the 3rd bet in sequence
        did_float_bet = 0.0    # Called previous street, bet when checked to
        did_probe_bet = 0.0    # Bet OOP after PF aggressor checked back
        
        # Track player's action pattern for strategic detection
        player_actions_this_street = []
        
        # Track pot size as we iterate through actions
        starting_pot_this_street = static_ctx.game_state.starting_pot_this_round
        pot_at_action = float(starting_pot_this_street)
        
        for action_seat_id, action_type, amount in action_sequence:
            # All players contribute to pot progression
            if action_type in ["bet", "raise"]:
                pot_at_action += amount
            
            # Track this player's specific actions
            if action_seat_id == seat_id:
                # Record action for strategic pattern detection
                player_actions_this_street.append((action_type, amount))
                
                if action_type == "check":
                    checked_count += 1.0
                elif action_type in ["bet", "raise"]:
                    raised_count += 1.0
                    total_raise_amount += amount
                    
                    # Calculate pot BEFORE this player's action (excluding their contribution)
                    pot_before_action = pot_at_action - amount
                    
                    # Feature 1: Raise as % of pot that existed before the raise
                    if pot_before_action > 0:
                        raise_pct = amount / pot_before_action
                        total_raise_pct_of_pot += raise_pct
                        
                        # Check for overbets and large bets (using correct pot size)
                        if amount > pot_before_action:
                            overbet_count += 1.0
                        if amount > pot_before_action * 0.7:
                            largebet_count += 1.0
                    
                    # Feature 2: Total bet commitment relative to starting street pot
                    if starting_pot_this_street > 0:
                        player_total_bet = current_bets[seat_id]
                        bet_to_starting_pot_ratio = player_total_bet / starting_pot_this_street
                        total_bet_to_starting_pot += bet_to_starting_pot_ratio
        
        # Calculate averages
        if raised_count > 0:
            avg_raise_pct_of_pot = total_raise_pct_of_pot / raised_count
            avg_bet_to_starting_pot = total_bet_to_starting_pot / raised_count
        else:
            avg_raise_pct_of_pot = 0.0
            avg_bet_to_starting_pot = 0.0
        
        # Detect strategic patterns
        did_check_raise = self._detect_check_raise(player_actions_this_street)
        did_donk_bet = self._detect_donk_bet(seat_id, action_sequence, static_ctx, dynamic_ctx)
        did_3bet = self._detect_3bet(seat_id, action_sequence)
        did_float_bet = self._detect_float_bet(seat_id, static_ctx, dynamic_ctx)
        did_probe_bet = self._detect_probe_bet(seat_id, action_sequence, static_ctx, dynamic_ctx)
        
        # What this seat is currently facing (monotonic)
        is_facing_check = 1.0 if max_bet == 0 else 0.0
        is_facing_bet = 1.0 if max_bet > 0 and to_call > 0 else 0.0
        
        # Count betting rounds from action sequence
        betting_rounds = self._count_betting_rounds(action_sequence)
        
        is_facing_raise = 1.0 if betting_rounds >= 2 and to_call > 0 else 0.0
        is_facing_3bet = 1.0 if betting_rounds >= 3 and to_call > 0 else 0.0
        is_facing_4betplus = 1.0 if betting_rounds >= 4 and to_call > 0 else 0.0
        
        return {
            "checked_count": checked_count,
            "raised_count": raised_count,
            "raise_pct_of_pot": min(avg_raise_pct_of_pot, 3.0) / 3.0,  # Normalize to 0-1, cap at 300%
            "bet_to_starting_pot": min(avg_bet_to_starting_pot, 5.0) / 5.0,  # Normalize to 0-1, cap at 500%
            "overbet_count": overbet_count,
            "largebet_count": largebet_count,
            "is_facing_check": is_facing_check,
            "is_facing_bet": is_facing_bet,
            "is_facing_raise": is_facing_raise,
            "is_facing_3bet": is_facing_3bet,
            "is_facing_4betplus": is_facing_4betplus,
            # Strategic features
            "did_check_raise": did_check_raise,
            "did_donk_bet": did_donk_bet,
            "did_3bet": did_3bet,
            "did_float_bet": did_float_bet,
            "did_probe_bet": did_probe_bet
        }
    
    def calculate_current_street_stack(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate current street stack features for any seat_id.
        
        Features:
        - Seat id stack in BB
        - Seat id pot size ratio (pot size / total money)
        - Seat id call cost ratio (to call / stack)
        - Seat id pot odds (to call / (pot + to call))
        - Seat id stack size ratio (stack / total money)
        - Seat id amount committed this street in BB*
        - Seat id amount committed this street / starting pot this street*
        - Seat id amount committed this street / seat id starting stack this street*
        - Seat id total commitment (across all streets: % of stack committed)
        - Seat id total commitment in BB*
        - Seat id stack is smaller than pot
        """
        stack = static_ctx.game_state.stacks[seat_id]
        current_bet = static_ctx.game_state.current_bets[seat_id]
        max_bet = max(static_ctx.game_state.current_bets) if static_ctx.game_state.current_bets else 0
        to_call = max_bet - current_bet
        pot = static_ctx.pot
        
        # Get big blind and total money
        big_blind = dynamic_ctx.history_tracker.get_big_blind()
        all_stacks = static_ctx.game_state.stacks
        total_money = sum(all_stacks) + pot
        
        # Stack in big blinds
        stack_in_bb = stack / max(big_blind, 1)
        
        # Basic ratios
        pot_size_ratio = pot / max(total_money, 1)
        call_cost_ratio = to_call / max(stack, 1)
        pot_odds = to_call / max(pot + to_call, 1) if to_call > 0 else 0.0
        stack_size_ratio = stack / max(total_money, 1)
        
        # Current street commitment (from current_bets)
        current_street_commitment = current_bet
        current_street_commitment_bb = current_street_commitment / max(big_blind, 1)
        
        # Starting pot this street calculation
        # The pot in game_state represents the starting pot before current street betting
        # This is the pot that players are betting into
        starting_pot_this_street = pot
        
        current_street_commitment_vs_starting_pot = current_street_commitment / max(starting_pot_this_street, 1)
        
        # Starting stack this street calculation  
        player_total_invested = dynamic_ctx.history_tracker.get_player_investment_all()
        if seat_id < len(player_total_invested):
            total_invested_all_streets = player_total_invested[seat_id]
            starting_stack_this_hand = stack + total_invested_all_streets
            # Starting stack this street = stack before making any bets this street
            # This is the stack the player had at the beginning of this betting round
            starting_stack_this_street = stack + current_street_commitment
            
            # Calculate commitment vs starting stack
            # If commitment equals or exceeds starting stack, it's all-in (1.0)
            if current_street_commitment >= starting_stack_this_street:
                current_street_commitment_vs_starting_stack = 1.0  # All-in = 100% commitment
            else:
                current_street_commitment_vs_starting_stack = current_street_commitment / max(starting_stack_this_street, 1)
            
            # Total commitment across all streets
            total_commitment = total_invested_all_streets / max(starting_stack_this_hand, 1)
            total_commitment_bb = total_invested_all_streets / max(big_blind, 1)
        else:
            current_street_commitment_vs_starting_stack = 0.0
            total_commitment = 0.0
            total_commitment_bb = 0.0
        
        # Stack comparison to pot
        stack_smaller_than_pot = 1.0 if stack < pot else 0.0
        
        return {
            "stack_in_bb": min(stack_in_bb / 200.0, 1.0),  # Normalize over 200bb scale (matches StackDepthSimulator total)
            "pot_size_ratio": pot_size_ratio,
            "call_cost_ratio": min(call_cost_ratio, 1.0),
            "pot_odds": pot_odds,
            "stack_size_ratio": stack_size_ratio,
            # NEW: Current street commitment features
            "current_street_commitment_bb": current_street_commitment_bb,
            "current_street_commitment_vs_starting_pot": min(current_street_commitment_vs_starting_pot, 3.0) / 3.0,  # Normalize
            "current_street_commitment_vs_starting_stack": min(current_street_commitment_vs_starting_stack, 1.0),
            # Total commitment features
            "total_commitment": min(total_commitment, 1.0),
            "total_commitment_bb": total_commitment_bb,
            "stack_smaller_than_pot": stack_smaller_than_pot
        }
    
    def calculate_current_position(self, seat_id: int, static_ctx: StaticContext) -> dict:
        """
        Calculate current position features for any seat_id.
        
        Features:
        - Seat id is early position
        - Seat id is dealer
        - Seat id is SB (heads up, this = dealer)
        - Seat id is BB
        """
        dealer_pos = static_ctx.game_state.dealer_pos
        num_players = static_ctx.num_players
        
        # Position calculations
        is_dealer = 1.0 if seat_id == dealer_pos else 0.0
        
        # For heads-up: dealer = SB, other = BB
        if num_players == 2:
            is_sb = is_dealer
            is_bb = 1.0 if seat_id != dealer_pos else 0.0
            is_early_position = is_sb  # SB acts first preflop = early position
        else:
            # Multi-way position calculation
            relative_position = (seat_id - dealer_pos) % num_players
            is_early_position = 1.0 if relative_position <= num_players // 2 else 0.0
            is_sb = 1.0 if relative_position == 1 else 0.0
            is_bb = 1.0 if relative_position == 2 else 0.0
        
        return {
            "is_early_position": is_early_position,
            "is_dealer": is_dealer,
            "is_sb": is_sb,
            "is_bb": is_bb
        }
    
    def calculate_current_stage(self, static_ctx: StaticContext) -> dict:
        """
        Calculate current stage features (not seat-specific).
        
        Features:
        - is_preflopplus
        - is_flopplus  
        - is_turnplus
        - is_river
        """
        stage = static_ctx.stage
        
        return {
            "is_preflopplus": 1.0,  # Always true (we're at least on preflop)
            "is_flopplus": 1.0 if stage >= 1 else 0.0,
            "is_turnplus": 1.0 if stage >= 2 else 0.0,
            "is_river": 1.0 if stage == 3 else 0.0
        }
    
    def calculate_current_street_additional(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, 
                                           opponent_stats: dict = None, opponent_action_history: list = None) -> dict:
        """
        Calculate current additional hero-only features.
        
        Features:
        - Effective stack to pot ratio (min(stacks) / pot)
        - Implied odds (enhanced with opponent range consideration)
        - Hand strength (Monte Carlo vs random hands)
        - Equity vs range (equity vs opponent's estimated range)
        - Equity delta (change from previous street)
        - SPR delta (change from previous street)  
        - Pot size delta (change from previous street)
        """
        stacks = static_ctx.game_state.stacks
        pot = static_ctx.pot
        hole_cards = static_ctx.hole_cards
        community_cards = static_ctx.community
        
        # Effective stack to pot ratio
        if len(stacks) >= 2 and pot > 0:
            effective_stack = min(stacks)
            effective_spr = effective_stack / pot
        else:
            effective_spr = 0.0
        
        # Enhanced implied odds calculation (considers opponent range if available)
        implied_odds = self._calculate_enhanced_implied_odds(static_ctx, dynamic_ctx, opponent_stats)
        
        # Hand strength via Monte Carlo simulation
        hand_strength = self._calculate_hand_strength(hole_cards, community_cards)
        
        # Equity vs opponent's estimated range
        equity_vs_range = self._calculate_equity_vs_range(
            hole_cards, community_cards, opponent_stats, opponent_action_history, pot
        )
        
        # Delta features (change from previous street)
        equity_delta = self._calculate_equity_delta(static_ctx, dynamic_ctx, equity_vs_range)
        spr_delta = self._calculate_spr_delta(static_ctx, dynamic_ctx, effective_spr)
        pot_size_delta = self._calculate_pot_size_delta(static_ctx, dynamic_ctx, pot)
        
        return {
            "effective_spr": effective_spr,
            "implied_odds": implied_odds,
            "hand_strength": hand_strength,
            "equity_vs_range": equity_vs_range,
            "equity_delta": equity_delta,
            "spr_delta": spr_delta,
            "pot_size_delta": pot_size_delta
        }
    
    def _calculate_current_implied_odds(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Calculate current implied odds using proper board texture analysis.
        Integrates with board analyzer for full calculation.
        """
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
    
    def _count_betting_rounds(self, action_sequence: list) -> int:
        """
        Count the number of betting rounds from the action sequence.
        Each aggressive action (bet/raise) starts a new betting round.
        """
        betting_rounds = 0
        
        for seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                betting_rounds += 1
        
        return betting_rounds
    
    def _calculate_enhanced_implied_odds(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, opponent_stats: dict = None) -> float:
        """
        Enhanced implied odds calculation that considers opponent's range and tendencies.
        """
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
    
    def _calculate_hand_strength(self, hole_cards: list, community_cards: list) -> float:
        """
        Calculate hand strength via Monte Carlo simulation vs random hands.
        """
        # Create cache key
        cache_key = tuple(sorted(hole_cards + community_cards))
        if cache_key in self.hand_strength_cache:
            return self.hand_strength_cache[cache_key]
        
        if len(hole_cards) < 2:
            return 0.5  # Default for invalid hands
        
        # Monte Carlo simulation
        wins = 0
        ties = 0
        trials = 200  # Reduced for speed
        
        # Create deck without known cards
        known_cards = set(hole_cards + community_cards)
        deck = [i for i in range(52) if i not in known_cards]
        
        for _ in range(trials):
            # Deal random opponent hand
            if len(deck) < 2:
                break
            opp_cards = random.sample(deck, 2)
            remaining_deck = [c for c in deck if c not in opp_cards]
            
            # Complete the board if needed
            cards_needed = 5 - len(community_cards)
            if cards_needed > 0 and len(remaining_deck) >= cards_needed:
                board_completion = random.sample(remaining_deck, cards_needed)
                final_board = community_cards + board_completion
            else:
                final_board = community_cards + [0] * cards_needed  # Pad if needed
            
            # Evaluate hands (simplified - using card ranks)
            my_strength = self._evaluate_hand_strength(hole_cards, final_board)
            opp_strength = self._evaluate_hand_strength(opp_cards, final_board)
            
            if my_strength > opp_strength:
                wins += 1
            elif my_strength == opp_strength:
                ties += 1
        
        if trials == 0:
            strength = 0.5
        else:
            strength = (wins + ties * 0.5) / trials
        
        # Cache the result
        self.hand_strength_cache[cache_key] = strength
        return strength
    
    def _evaluate_hand_strength(self, hole_cards: list, board: list) -> float:
        """
        Simplified hand strength evaluation for Monte Carlo simulation.
        """
        all_cards = hole_cards + board
        if len(all_cards) < 5:
            all_cards += [0] * (5 - len(all_cards))  # Pad with low cards
        
        # Convert to ranks for simple evaluation
        ranks = [card // 4 for card in all_cards[:5]]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Simple hand ranking (higher is better)
        counts = sorted(rank_counts.values(), reverse=True)
        if counts[0] == 4:  # Four of a kind
            return 7.0 + max(ranks)
        elif counts[0] == 3 and counts[1] == 2:  # Full house
            return 6.0 + max(ranks)
        elif counts[0] == 3:  # Three of a kind
            return 3.0 + max(ranks)
        elif counts[0] == 2 and counts[1] == 2:  # Two pair
            return 2.0 + max(ranks)
        elif counts[0] == 2:  # One pair
            return 1.0 + max(ranks)
        else:  # High card
            return max(ranks) / 13.0
    
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
    
    def _card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to string format for equity calculator."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _calculate_equity_delta(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, current_equity: float) -> float:
        """Calculate change in equity from previous street."""
        current_stage = static_ctx.stage
        
        # No delta for pre-flop
        if current_stage == 0:
            return 0.0
        
        # Get previous street name
        previous_streets = {1: 'preflop', 2: 'flop', 3: 'turn'}
        previous_street = previous_streets.get(current_stage)
        
        if not previous_street:
            return 0.0
        
        # Look up previous equity from history
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        previous_equity = dynamic_ctx.history_tracker.get_feature_value(
            hand_key, previous_street, "equity_vs_range", current_equity
        )
        
        return current_equity - previous_equity
    
    def _calculate_spr_delta(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, current_spr: float) -> float:
        """Calculate change in SPR from previous street."""
        current_stage = static_ctx.stage
        
        # No delta for pre-flop
        if current_stage == 0:
            return 0.0
        
        # Get previous street name
        previous_streets = {1: 'preflop', 2: 'flop', 3: 'turn'}
        previous_street = previous_streets.get(current_stage)
        
        if not previous_street:
            return 0.0
        
        # Look up previous SPR from history
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        previous_spr = dynamic_ctx.history_tracker.get_feature_value(
            hand_key, previous_street, "effective_spr", current_spr
        )
        
        return current_spr - previous_spr
    
    def _calculate_pot_size_delta(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext, current_pot: int) -> float:
        """Calculate change in pot size from previous street."""
        current_stage = static_ctx.stage
        
        # No delta for pre-flop
        if current_stage == 0:
            return 0.0
        
        # Get previous street name
        previous_streets = {1: 'preflop', 2: 'flop', 3: 'turn'}
        previous_street = previous_streets.get(current_stage)
        
        if not previous_street:
            return 0.0
        
        # For pot size, we need to reconstruct what the pot was at the end of the previous street
        # This is approximately the starting_pot_this_round in the game state
        previous_pot = static_ctx.game_state.starting_pot_this_round
        
        # Return the change, normalized by big blind for consistency
        big_blind = dynamic_ctx.history_tracker.get_big_blind()
        delta_chips = current_pot - previous_pot
        return delta_chips / max(big_blind, 1)
    
    def _detect_check_raise(self, player_actions: list) -> float:
        """
        Detect if player check-raised (checked then raised on same street).
        
        Args:
            player_actions: List of (action_type, amount) tuples for this player
            
        Returns:
            1.0 if check-raise detected, 0.0 otherwise
        """
        if len(player_actions) < 2:
            return 0.0
        
        # Look for check followed by raise pattern
        for i in range(len(player_actions) - 1):
            if (player_actions[i][0] == "check" and 
                player_actions[i + 1][0] in ["bet", "raise"]):
                return 1.0
        
        return 0.0
    
    def _detect_donk_bet(self, seat_id: int, action_sequence: list, 
                        static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Detect if player made a donk bet (bet out of position when not the previous street aggressor).
        
        A donk bet is when:
        1. Player bets first on a street (not preflop)
        2. Player is out of position 
        3. Player was not the aggressor on the previous street
        """
        if static_ctx.game_state.stage == 0:  # Can't donk bet preflop
            return 0.0
        
        # Check if this player made the first bet this street
        first_aggressive_action_seat = None
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                first_aggressive_action_seat = action_seat_id
                break
        
        if first_aggressive_action_seat != seat_id:
            return 0.0
        
        # Check if player is out of position (not dealer in heads-up)
        # In heads-up: dealer is SB and acts first postflop (out of position)
        dealer_seat = static_ctx.game_state.dealer_seat
        is_out_of_position = (seat_id == dealer_seat)
        
        if not is_out_of_position:
            return 0.0
        
        # Check if player was NOT the aggressor on previous street
        previous_street_aggressor = dynamic_ctx.history_tracker.get_last_aggressor()
        if previous_street_aggressor == seat_id:
            return 0.0
        
        return 1.0
    
    def _detect_3bet(self, seat_id: int, action_sequence: list) -> float:
        """
        Detect if player made a 3-bet (third bet in the sequence).
        
        3-bet is the third aggressive action in a betting sequence.
        """
        aggressive_actions = []
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                aggressive_actions.append(action_seat_id)
        
        # Player made the 3-bet if they made the 3rd aggressive action
        if len(aggressive_actions) >= 3 and aggressive_actions[2] == seat_id:
            return 1.0
        
        return 0.0
    
    def _detect_float_bet(self, seat_id: int, static_ctx: StaticContext, 
                         dynamic_ctx: DynamicContext) -> float:
        """
        Detect if player made a float bet.
        
        Float bet: Called a bet on previous street in position, then bet when checked to.
        This requires checking history across streets.
        """
        current_stage = static_ctx.game_state.stage
        
        # Need at least flop to have a float bet
        if current_stage < 2:  # Need turn or river
            return 0.0
        
        # Check if player is in position (dealer in heads-up)
        dealer_seat = static_ctx.game_state.dealer_seat
        is_in_position = (seat_id == dealer_seat)
        
        if not is_in_position:
            return 0.0
        
        # Get previous street history
        previous_street_history = dynamic_ctx.history_tracker.get_street_history(current_stage - 1)
        if not previous_street_history:
            return 0.0
        
        # Check if player called on previous street
        player_called_previous = False
        opponent_bet_previous = False
        
        for action in previous_street_history:
            if action.get('seat_id') == seat_id and action.get('action') == 'call':
                player_called_previous = True
            elif action.get('seat_id') != seat_id and action.get('action') in ['bet', 'raise']:
                opponent_bet_previous = True
        
        if not (player_called_previous and opponent_bet_previous):
            return 0.0
        
        # Check if opponent checked this street and player bet
        current_street_actions = dynamic_ctx.history_tracker.get_current_street_actions()
        opponent_checked = False
        player_bet = False
        
        for action in current_street_actions:
            if action.get('seat_id') != seat_id and action.get('action') == 'check':
                opponent_checked = True
            elif action.get('seat_id') == seat_id and action.get('action') in ['bet', 'raise']:
                player_bet = True
        
        if opponent_checked and player_bet:
            return 1.0
        
        return 0.0
    
    def _detect_probe_bet(self, seat_id: int, action_sequence: list,
                         static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Detect if player made a probe bet.
        
        Probe bet: Bet out of position after the preflop aggressor checked back on previous street.
        """
        current_stage = static_ctx.game_state.stage
        
        # Need at least turn for probe bet
        if current_stage < 2:
            return 0.0
        
        # Check if player is out of position
        dealer_seat = static_ctx.game_state.dealer_seat
        is_out_of_position = (seat_id == dealer_seat)
        
        if not is_out_of_position:
            return 0.0
        
        # Check if player bet first this street
        first_bet_this_street = None
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                first_bet_this_street = action_seat_id
                break
        
        if first_bet_this_street != seat_id:
            return 0.0
        
        # Check if preflop aggressor checked back on previous street
        preflop_aggressor = dynamic_ctx.history_tracker.get_preflop_aggressor()
        if preflop_aggressor is None:
            return 0.0
        
        previous_street_history = dynamic_ctx.history_tracker.get_street_history(current_stage - 1)
        if not previous_street_history:
            return 0.0
        
        # Check if preflop aggressor checked on previous street
        pf_aggressor_checked = False
        for action in previous_street_history:
            if (action.get('seat_id') == preflop_aggressor and 
                action.get('action') == 'check'):
                pf_aggressor_checked = True
                break
        
        if pf_aggressor_checked:
            return 1.0
        
        return 0.0
    
