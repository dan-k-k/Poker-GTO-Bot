# analyzers/current_street_analyzer.py
# Current street features that don't rare yore history tracking
# All features are opponent reproducible - pass different seat_id for self vs opponent

from feature_contexts import StaticContext, DynamicContext
from .action_sequencer import ActionSequencer
from .board_analyzer import BoardAnalyzer
from .hand_analyzer import HandAnalyzer

# Import for equity calculations
import sys
import os
import math
from poker_core import HandEvaluator

# Add trainingL1 to path for equity calculator imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trainingL1'))
try:
    from Poker.trainingL1._OLD_equity_calculator import EquityCalculator, RangeConstructor
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
        self.hand_analyzer = HandAnalyzer()  # For accurate preflop hand strength lookup
        
        # Initialize equity calculation components
        if EquityCalculator and RangeConstructor:
            self.equity_calculator = EquityCalculator()
            self.range_constructor = RangeConstructor()
        else:
            self.equity_calculator = None
            self.range_constructor = None
            
        # Hand strength calculations now delegated to HandAnalyzer
    
    def calculate_current_street_sequence(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext, action_sequencer: ActionSequencer) -> dict:
        """
        Calculate current street sequence features for any seat_id using ActionSequencer.
        
        Features:
        - Seat id checked x times
        - Seat id called x times
        - Seat id raised (includes initial bet) x times
        - Seat id raised BY x% of pot on average
        - Seat id raised TO x% of street’s starting pot
        - Seat id overbet (>100% of pot raise/bet) x times
        - Seat id largebet (>70% of pot raise/bet) x times
        - Seat id did check raise, donk, 3bet, float, probe bet [all monotonic]

        - Current only: Seat id is currently facing a check, bet, raise(2bet), 3bet, 4betplus [all monotonic]
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
        called_count = 0.0
        raised_count = 0.0
        total_raise_amount = 0.0
        overbet_count = 0.0
        largebet_count = 0.0
        smallbet_count = 0.0  # NEW: Count of small bets (<=35% of pot)
        
        # NEW: Better strategic features
        raise_pct_list = []  # List of (raise_amount / pot_before_raise) for averaging
        
        # NEW: Advanced strategic features
        did_check_raise = 0.0  # Player checked then raised on same street
        did_donk_bet = 0.0     # Player bet out of position when not aggressor
        did_3bet = 0.0         # Player made the 3rd bet in sequence
        did_float_bet = 0.0    # Called previous street, bet when checked to
        did_probe_bet = 0.0    # Bet OOP after PF aggressor checked back
        did_go_all_in = 0.0    # Player went all-in on this street
        
        # Track player's action pattern for strategic detection
        player_actions_this_street = []
        
        # Track pot size as we iterate through actions
        starting_pot_this_street = static_ctx.game_state.starting_pot_this_round
        pot_at_action = float(starting_pot_this_street)
        
        # NEW: Reconstruct player stacks at the START of the street for all-in calculations
        num_players = static_ctx.num_players
        starting_stacks_this_street = [s + c for s, c in zip(static_ctx.game_state.stacks, static_ctx.game_state.current_bets)]
        wagered_this_street = [0.0] * num_players
        
        
        for action_seat_id, action_type, amount in action_sequence:
            # ✅ NEW: Calculate stack BEFORE this action to check for all-ins
            stack_before_action = starting_stacks_this_street[action_seat_id] - wagered_this_street[action_seat_id]
            
            # ✅ FIX: Calculate pot BEFORE updating it for "Raise BY" accuracy
            # Track this player's specific actions BEFORE updating pot state
            if action_seat_id == seat_id:
                # Record action for strategic pattern detection
                player_actions_this_street.append((action_type, amount))
                
                if action_type == "check":
                    checked_count += 1.0
                elif action_type == "call":
                    called_count += 1.0
                elif action_type in ["bet", "raise"]:
                    raised_count += 1.0
                    total_raise_amount += amount
                    
                    # ✅ NEW: Check for All-In (use small tolerance for floating point safety)
                    if amount >= stack_before_action - 0.1:
                        did_go_all_in = 1.0
                    
                    # ✅ "Raise BY" Calculation: amount / pot_at_current_moment
                    if pot_at_action > 0:
                        # 'amount' represents the chips being added (raise BY amount)
                        # pot_at_action is the pot BEFORE this player's action
                        raise_pct = amount / pot_at_action
                        raise_pct_list.append(raise_pct)
                        
                        # Check for overbets, large bets, and small bets
                        if amount > pot_at_action:
                            overbet_count += 1.0
                        if amount > pot_at_action * 0.7:
                            largebet_count += 1.0
                        # ✅ NEW: Check for Small Bet (<=35% of pot)
                        if amount <= pot_at_action * 0.35:
                            smallbet_count += 1.0
            
            # Update simulated state for the next action in the loop
            if amount:
                if action_type in ["call", "bet", "raise"]:
                    wagered_this_street[action_seat_id] += amount
                pot_at_action += amount
        
        # Calculate final values
        if raise_pct_list:
            avg_raise_pct_of_pot = sum(raise_pct_list) / len(raise_pct_list)
        else:
            avg_raise_pct_of_pot = 0.0
        
        # ✅ FIXED: Calculate "Aggressive Commitment" - only counts wagers up to last aggressive action
        aggro_commit_total = 0.0
        
        # Find the index of the player's last aggressive action
        last_aggro_action_index = -1
        for i in range(len(action_sequence) - 1, -1, -1):
            action_seat_id, action_type, _ = action_sequence[i]
            if action_seat_id == seat_id and action_type in ["bet", "raise"]:
                last_aggro_action_index = i
                break
                
        # If an aggressive action was found, sum all their wagers up to that point
        if last_aggro_action_index != -1:
            for i in range(last_aggro_action_index + 1):
                action_seat_id, action_type, amount = action_sequence[i]
                if action_seat_id == seat_id and action_type in ["call", "bet", "raise"]:
                    aggro_commit_total += amount

        player_total_bet_this_street = aggro_commit_total

        if starting_pot_this_street > 0:
            aggro_commit_ratio = player_total_bet_this_street / starting_pot_this_street
        else:
            aggro_commit_ratio = 0.0
        
        # Detect strategic patterns
        did_check_raise = self._detect_check_raise(player_actions_this_street)
        did_donk_bet = self._detect_donk_bet(seat_id, action_sequence, static_ctx, dynamic_ctx)
        did_3bet = self._detect_3bet(seat_id, action_sequence)
        did_float_bet = self._detect_float_bet(seat_id, action_sequencer, static_ctx, dynamic_ctx)
        did_probe_bet = self._detect_probe_bet(seat_id, action_sequence, static_ctx, dynamic_ctx)
        did_cbet = self._detect_cbet(seat_id, action_sequence, static_ctx, dynamic_ctx)
        did_open_overbet = self._detect_open_overbet(seat_id, action_sequence, starting_pot_this_street)
        did_open_largebet = self._detect_open_largebet(seat_id, action_sequence, starting_pot_this_street)
        
        # Normalize action counts with logarithmic scaling (strategic importance decreases at higher counts)
        MAX_LOG_ACTIONS = 1.609437912  # ln(1 + 4) for 4bet cap on a single street
        
        return {
            "checked_count": min(math.log1p(checked_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            "called_count": min(math.log1p(called_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            "raised_count": min(math.log1p(raised_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            "avg_raise_pct_of_pot": min(avg_raise_pct_of_pot, 3.0)/3.0,  # Cap at 300% but keep as actual ratio
            "aggro_commit_ratio": min(aggro_commit_ratio, 5.0)/5.0,  # Cap at 500% but keep as actual ratio
            "overbet_count": min(math.log1p(overbet_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            "largebet_count": min(math.log1p(largebet_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            "smallbet_count": min(math.log1p(smallbet_count), MAX_LOG_ACTIONS) / MAX_LOG_ACTIONS,
            # Strategic features
            "did_go_all_in": did_go_all_in,
            "did_check_raise": did_check_raise,
            "did_donk_bet": did_donk_bet,
            "did_3bet": did_3bet,
            "did_float_bet": did_float_bet,
            "did_probe_bet": did_probe_bet,
            "did_cbet": did_cbet,
            "did_open_overbet": did_open_overbet,
            "did_open_largebet": did_open_largebet
        }
    
    def calculate_current_street_stack(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext, action_sequencer=None) -> dict:
        """
        Calculate current street stack features for any seat_id.
        
        Features:
        - Seat id stack in BB
        - Seat id pot size ratio (pot size / total money)
        - Seat id call cost ratio (to call / stack)
        - Seat id pot odds (to call / (pot + to call))
        - Seat id stack size ratio (stack / total money)
        - Seat id amount committed this street in BB
        - Seat id amount committed this street / starting pot this street
        - Seat id amount committed this street / seat id starting stack this street
        - Seat id total commitment (cumulative %)
        - Seat id total commitment in BB
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
        current_street_commitment_bb_raw = current_street_commitment / max(big_blind, 1)
        current_street_commitment_bb = min(current_street_commitment_bb_raw / 200.0, 1.0)  # Normalize over 200BB scale
        
        # Starting pot this street calculation
        # The pot in game_state represents the starting pot before current street betting
        # This is the pot that players are betting into
        starting_pot_this_street = static_ctx.game_state.starting_pot_this_round # <--- ✅ FIX
        
        current_street_commitment_vs_starting_pot = current_street_commitment / max(starting_pot_this_street, 1)
        
        # Starting stack this street calculation  
        player_total_invested = dynamic_ctx.history_tracker.get_player_investment_all()
        if seat_id < len(player_total_invested):
            total_invested_all_streets = player_total_invested[seat_id]
            # ✅ FIX: Get the true starting stack directly from game state for this hand
            starting_stack_this_hand = static_ctx.game_state.starting_stacks_this_hand[seat_id]
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
            # ✅ FIX: Normalize total_commitment_bb like other BB features (0-1 scale)
            total_commitment_bb_raw = total_invested_all_streets / max(big_blind, 1)
            total_commitment_bb = min(total_commitment_bb_raw / 200.0, 1.0)  # Normalize over 200BB scale
        else:
            current_street_commitment_vs_starting_stack = 0.0
            total_commitment = 0.0
            total_commitment_bb = 0.0
        
        # Stack comparison to pot
        stack_smaller_than_pot = 1.0 if stack < pot else 0.0
        
        # NEW: Calculate last pot odds and call cost faced this street
        last_pot_odds_faced, last_call_cost_faced = self._calculate_last_odds_faced(
            seat_id, static_ctx, action_sequencer
        )
        
        return {
            "stack_in_bb": min(stack_in_bb / 200.0, 1.0),  # Normalize over 200bb scale (matches StackDepthSimulator total)
            "pot_size_ratio": pot_size_ratio,
            "call_cost_ratio": min(call_cost_ratio, 1.0),
            "pot_odds": pot_odds,
            "stack_size_ratio": stack_size_ratio,
            # NEW: Current street commitment features
            "current_street_commitment_bb": current_street_commitment_bb,
            "current_street_commitment_vs_starting_pot": min(current_street_commitment_vs_starting_pot, 5.0) / 5.0,  # Normalize
            "current_street_commitment_vs_starting_stack": min(current_street_commitment_vs_starting_stack, 1.0),
            # Total commitment features
            "total_commitment_pct": min(total_commitment, 1.0),
            "total_commitment_bb": total_commitment_bb,
            "stack_smaller_than_pot": stack_smaller_than_pot,
            # NEW: Last odds faced features
            "last_pot_odds_faced_this_street": last_pot_odds_faced,
            "last_call_cost_faced_this_street": last_call_cost_faced
        }
    
    def calculate_current_position(self, seat_id: int, static_ctx: StaticContext) -> dict:
        """
        Calculate current position features for any seat_id, correctly
        differentiating between pre-flop and post-flop action.
        
        Features:
        - Seat id is OOP (out of position/disadvantageous)
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
        else:
            relative_position = (seat_id - dealer_pos) % num_players
            is_sb = 1.0 if relative_position == 1 else 0.0
            is_bb = 1.0 if relative_position == 2 else 0.0
        
        # Use shared helper for OOP logic
        is_OOP = 1.0 if self._is_out_of_position(seat_id, static_ctx) else 0.0
        
        return {
            "is_OOP": is_OOP,
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
                                           action_sequencer=None, opponent_stats: dict = None, opponent_action_history: list = None) -> dict:
        """
        Calculate current additional self-only features.
        
        Features:
        - Effective stack to pot ratio (min(stacks) / pot)
        - Implied odds (enhanced with opponent range consideration)
        - Hand strength (Monte Carlo vs random hands)
        - Equity vs range (equity vs opponent's estimated range)
        - Deltas: Equity, SPR, Pot size
        """
        stacks = static_ctx.game_state.stacks
        pot = static_ctx.pot
        hole_cards = static_ctx.hole_cards
        community_cards = static_ctx.community
        
        # Effective stack to pot ratio (logarithmically normalized)
        if len(stacks) >= 2 and pot > 0:
            effective_stack = min(stacks)
            raw_spr = effective_stack / pot
            
            capped_spr = min(raw_spr, 30.0) # Cap at 30 to prevent extreme deep-stack values from dominating
            log_spr = math.log1p(capped_spr) # log1p(x) = log(1+x), safely handles SPR of 0
            MAX_LOG_SPR = 3.4339872044851463 # Normalize to 0-1 scale
            effective_spr = log_spr / MAX_LOG_SPR
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
        spr_delta = self._calculate_spr_delta(static_ctx, dynamic_ctx, effective_spr)
        pot_size_delta = self._calculate_pot_size_delta(static_ctx, dynamic_ctx, pot)
        
        # Decision context features (Self only - what is the self currently facing?)
        current_bets = static_ctx.game_state.current_bets
        max_bet = max(current_bets) if current_bets else 0
        self_bet = current_bets[static_ctx.seat_id]
        to_call = max_bet - self_bet
        
        # Count betting rounds to determine if facing raise/3bet/4bet+
        if action_sequencer is not None:
            action_sequence = action_sequencer.get_live_action_sequence()
            betting_rounds = self._count_betting_rounds(action_sequence)
        else:
            betting_rounds = 0
            action_sequence = []
        
        # ✅ FIXED "is_facing" LOGIC
        # A player is only "facing a check" if there is no bet AND someone has acted before them.
        is_facing_check = 1.0 if max_bet == 0 and len(action_sequence) > 0 else 0.0
        is_facing_bet = 1.0 if max_bet > 0 and to_call > 0 else 0.0
        
        # Street-aware betting round logic
        # Preflop: BB counts as 1st bet, so thresholds are 1 lower
        # Postflop: No forced bets, so use standard thresholds
        is_preflop = len(community_cards) == 0
        
        if is_preflop:
            # Preflop: BB is 1st bet, so raise=2bet, 3bet=3bet, 4bet+=4bet+
            is_facing_raise = 1.0 if betting_rounds >= 1 and to_call > 0 else 0.0
            is_facing_3bet = 1.0 if betting_rounds >= 2 and to_call > 0 else 0.0
            is_facing_4betplus = 1.0 if betting_rounds >= 3 and to_call > 0 else 0.0
        else:
            # Postflop: Standard logic (no forced bets)
            is_facing_raise = 1.0 if betting_rounds >= 2 and to_call > 0 else 0.0
            is_facing_3bet = 1.0 if betting_rounds >= 3 and to_call > 0 else 0.0
            is_facing_4betplus = 1.0 if betting_rounds >= 4 and to_call > 0 else 0.0
        
        return {
            "effective_spr": effective_spr,
            "implied_odds": implied_odds,
            "hand_strength": hand_strength,
            "equity_vs_range": equity_vs_range,
            "spr_delta": spr_delta,
            "pot_size_delta": pot_size_delta,
            "is_facing_check": is_facing_check,
            "is_facing_bet": is_facing_bet,
            "is_facing_raise": is_facing_raise,
            "is_facing_3bet": is_facing_3bet,
            "is_facing_4betplus": is_facing_4betplus
        }
    
    ### helpers:

    def _is_out_of_position(self, seat_id: int, static_ctx: StaticContext) -> bool:
        """Shared helper to determine if a player is OOP."""
        dealer_pos = static_ctx.game_state.dealer_pos
        num_players = static_ctx.num_players
        stage = static_ctx.stage
        
        if num_players == 2:
            if stage == 0:  # Pre-flop
                return seat_id == dealer_pos  # SB acts first preflop
            else:  # Post-flop (flop, turn, river)
                return seat_id != dealer_pos  # BB acts first postflop
        else:
            relative_position = (seat_id - dealer_pos) % num_players
            
            if stage == 0:  # Pre-flop Logic
                OOP_count = max(1, num_players // 3)
                return relative_position > 2 and relative_position <= (2 + OOP_count)
            else:  # Post-flop Logic
                return relative_position > 0 and relative_position <= (num_players // 2)

    def _calculate_last_odds_faced(self, seat_id: int, static_ctx: StaticContext, action_sequencer) -> tuple[float, float]:
        """
        Calculate the last pot odds and call cost ratio this player faced before their last action.
        Returns (last_pot_odds_faced, last_call_cost_faced).
        """
        try:
            # ✅ FIX: Get the action sequence from the live ActionSequencer
            if action_sequencer is None:
                return 0.0, 0.0
                
            action_sequence = action_sequencer.get_live_action_sequence()
            if not action_sequence:
                return 0.0, 0.0
            
            # Simulate the betting state through the action sequence
            num_players = len(static_ctx.game_state.stacks)
            current_bets = [0] * num_players
            pot_at_action = static_ctx.game_state.starting_pot_this_round
            
            # Calculate stacks before current street actions (reconstruct from current state)
            player_stacks = static_ctx.game_state.stacks[:]
            for i in range(num_players):
                # Add back the chips they committed this street to get their pre-street stack
                player_stacks[i] += static_ctx.game_state.current_bets[i]
            
            last_odds_faced = 0.0
            last_call_cost_faced = 0.0
            
            for action_seat_id, action_type, amount in action_sequence:
                # Before this action, calculate what this player was facing
                if action_seat_id == seat_id:
                    max_bet_before = max(current_bets) if current_bets else 0
                    player_bet_before = current_bets[seat_id]
                    to_call_before = max_bet_before - player_bet_before
                    
                    if to_call_before > 0:
                        # This player was facing a bet - calculate their odds
                        total_pot_if_called = pot_at_action + to_call_before
                        pot_odds = to_call_before / max(total_pot_if_called, 1)
                        call_cost_ratio = to_call_before / max(player_stacks[seat_id], 1)
                        
                        # Store these as the last odds this player faced
                        last_odds_faced = pot_odds
                        last_call_cost_faced = min(call_cost_ratio, 1.0)
                
                # Process the action to update game state
                if action_type in ["call", "bet", "raise"]:
                    # amount is the additional chips being committed (not total bet)
                    contribution = amount
                    current_bets[action_seat_id] += amount
                    pot_at_action += contribution
                    player_stacks[action_seat_id] -= contribution
            
            return last_odds_faced, last_call_cost_faced
            
        except Exception:
            # Fallback to 0.0 if any error occurs
            return 0.0, 0.0
    
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

    def _calculate_hand_strength(self, hole_cards: list, community_cards: list) -> float:
        """
        ✅ Delegates hand strength calculation to the specialist HandAnalyzer.
        """
        return self.hand_analyzer._calculate_hand_strength(hole_cards, community_cards)
    
    # _evaluate_hand_strength method moved to HandAnalyzer
        
    def _card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to string format for equity calculator."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    
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
        """Calculate the change in pot size as a ratio of total money from the previous street."""
        current_stage = static_ctx.stage
        
        # No delta for pre-flop
        if current_stage == 0:
            return 0.0
        
        # Get the total money in play (use starting stacks for a stable denominator)
        total_money = float(sum(static_ctx.game_state.starting_stacks_this_hand))
        if total_money == 0:
            return 0.0

        # Get the pot size at the START of this street
        previous_pot = float(static_ctx.game_state.starting_pot_this_round)
        
        # Calculate the pot ratio for the previous state and the current state
        previous_pot_ratio = previous_pot / total_money
        current_pot_ratio = float(current_pot) / total_money
        
        # The delta is the difference between these two normalized ratios
        pot_size_delta = current_pot_ratio - previous_pot_ratio
        
        return pot_size_delta
    
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
        
        # ✅ FIX: Use the shared helper for consistent and correct OOP check
        if not self._is_out_of_position(seat_id, static_ctx):
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
    
    def _detect_float_bet(self, seat_id: int, action_sequencer, static_ctx: StaticContext, 
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
        
        # ✅ FIX: Use the shared helper for consistent and correct position check
        # Float bets are made by the in-position player
        if self._is_out_of_position(seat_id, static_ctx):
            return 0.0
        
        """
        Detects a float bet:
        1. Occurs on the Turn or River.
        2. Player is In Position.
        3. Player CALLED a BET on the previous street.
        4. Player BETS first on the current street after opponent checks.
        """
        if current_stage not in [2, 3]:  # Must be Turn or River
            return 0.0

        # Position check already done above - this is redundant
            
        # --- THE FIX: Get previous street's RAW action log from HistoryTracker ---
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        prev_street_name = 'flop' if current_stage == 2 else 'turn'
        prev_street_snapshot = dynamic_ctx.history_tracker.get_snapshot(hand_key, prev_street_name)
        prev_street_log = prev_street_snapshot.get('raw_action_log', [])
        # --- END THE FIX ---

        if not prev_street_log:
            return 0.0

        # Condition 3: Player must have called a bet on the previous street
        player_called_vs_bet = False
        for s_id, act_type, amt in prev_street_log:
            if s_id == seat_id and act_type == 'call' and amt > 0:
                player_called_vs_bet = True
                break
        if not player_called_vs_bet:
            return 0.0

        # Condition 4: Opponent must check first and Player must bet first on current street
        current_actions = action_sequencer.get_live_action_sequence()
        if not current_actions:
            return 0.0
            
        opponent_checked_first = current_actions[0][0] != seat_id and current_actions[0][1] == 'check'
        player_bet_this_street = any(s_id == seat_id and act_type == 'bet' for s_id, act_type, amt in current_actions)

        if opponent_checked_first and player_bet_this_street:
            return 1.0

        return 0.0
    
    def _detect_probe_bet(self, seat_id: int, action_sequence: list,
                         static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Detect if player made a probe bet.
        
        Probe bet: Bet out of position after the previous street's aggressor checked back.
        """
        current_stage = static_ctx.game_state.stage
        
        # Must be on Turn (stage 2) or River (stage 3)
        if current_stage < 2:
            return 0.0
        
        # Use shared helper for consistent OOP check
        if not self._is_out_of_position(seat_id, static_ctx):
            return 0.0
        
        # Player must have made the first bet on this street
        first_bet_this_street = None
        for action_seat_id, action_type, _ in action_sequence:
            if action_type in ["bet", "raise"]:
                first_bet_this_street = action_seat_id
                break
        
        if first_bet_this_street != seat_id:
            return 0.0
        
        # Get the aggressor from the previous street (not just preflop)
        previous_street_aggressor = dynamic_ctx.history_tracker.get_last_aggressor()
        
        # Can't be a probe bet if there was no previous aggressor, or if YOU were the aggressor
        if previous_street_aggressor is None or previous_street_aggressor == seat_id:
            return 0.0
        
        # Check if the previous street's aggressor checked back
        previous_street_history = dynamic_ctx.history_tracker.get_street_history(current_stage - 1)
        if not previous_street_history:
            return 0.0
        
        aggressor_checked_back = False
        for action in previous_street_history:
            if (action.get('seat_id') == previous_street_aggressor and 
                action.get('action') == 'check'):
                aggressor_checked_back = True
                break
        
        if aggressor_checked_back:
            return 1.0
        
        return 0.0
    
    def _detect_cbet(self, seat_id: int, action_sequence: list, 
                     static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> float:
        """
        Unified C-bet detection: Did this player continue aggression from the previous street?
        
        This works for any post-flop street:
        - Flop: continues from preflop aggressor
        - Turn: continues from flop aggressor  
        - River: continues from turn aggressor
        
        Returns 1.0 if player was previous street aggressor AND made first bet/raise this street.
        """
        stage = static_ctx.game_state.stage  # 0=preflop, 1=flop, 2=turn, 3=river
        
        # C-bet can only happen post-flop
        if stage == 0:
            return 0.0
        
        # Check if this player was the aggressor from the previous street
        previous_street_aggressor = dynamic_ctx.history_tracker.get_last_aggressor()
        if previous_street_aggressor != seat_id:
            return 0.0
            
        # Find the first bet/raise on the current street
        first_aggressive_action = None
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                first_aggressive_action = (action_seat_id, action_type, amount)
                break
        
        # If there was a bet/raise and this player made it, it's a C-bet
        if first_aggressive_action and first_aggressive_action[0] == seat_id:
            return 1.0
            
        return 0.0
    
    def _detect_open_largebet(self, seat_id: int, action_sequence: list, starting_pot: float) -> float:
        """
        Detect if this player made an opening large bet (first aggressive action that's >70% pot).
        
        This distinguishes between:
        - Opening large bet: Player bets >70% pot as first aggressive action
        - Raise large bet: Player large bets in response to opponent's bet/raise
        
        Returns 1.0 if player made opening large bet, 0.0 otherwise.
        """
        if starting_pot <= 0:
            return 0.0
            
        # Find the first aggressive action (bet/raise) on this street
        first_aggressive_action = None
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                first_aggressive_action = (action_seat_id, action_type, amount)
                break
        
        # Check if this player made the first aggressive action AND it was a large bet (>70% pot)
        if (first_aggressive_action and 
            first_aggressive_action[0] == seat_id and 
            first_aggressive_action[2] > starting_pot * 0.7):
            return 1.0
            
        return 0.0
    
    def _detect_open_overbet(self, seat_id: int, action_sequence: list, starting_pot: float) -> float:
        """
        Detect if this player made an opening overbet (first aggressive action that's >100% pot).
        
        This distinguishes between:
        - Opening overbet: Player bets >100% pot as first aggressive action
        - Raise overbet: Player overbets in response to opponent's bet/raise
        
        Returns 1.0 if player made opening overbet, 0.0 otherwise.
        """
        if starting_pot <= 0:
            return 0.0
            
        # Find the first aggressive action (bet/raise) on this street
        first_aggressive_action = None
        for action_seat_id, action_type, amount in action_sequence:
            if action_type in ["bet", "raise"]:
                first_aggressive_action = (action_seat_id, action_type, amount)
                break
        
        # Check if this player made the first aggressive action AND it was an overbet
        if (first_aggressive_action and 
            first_aggressive_action[0] == seat_id and 
            first_aggressive_action[2] > starting_pot):
            return 1.0
            
        return 0.0
    
