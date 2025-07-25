# analyzers/history_analyzer.py
# History tracked features with proper HistoryTracker integration
# Uses ActionSequencer for accurate historical data - no more guessing!

from feature_contexts import StaticContext, DynamicContext
from .current_street_analyzer import CurrentStreetAnalyzer


class HistoryAnalyzer:
    """
    Analyzer for features that require history tracking across streets.
    
    WRITE Operation: Takes ActionSequencer's complete log and saves accurate summaries.
    READ Operation: Retrieves previously saved summaries for historical context.
    
    Key Principle: Use ActionSequencer as the single source of truth for betting sequences.
    """
    
    def __init__(self):
        self.current_street_analyzer = CurrentStreetAnalyzer()  # For implied odds calculation
    
    def save_street_snapshot(self, action_log: list, static_ctx: StaticContext, dynamic_ctx: DynamicContext, street: str = None) -> None:
        """
        Save accurate historical data using ActionSequencer's complete log.
        Called ONCE at the end of each street with the final, complete action sequence.
        
        Args:
            action_log: Complete action log from ActionSequencer for this street
            static_ctx: Static game context
            dynamic_ctx: Dynamic context with HistoryTracker
            street: Explicit street name (overrides stage detection)
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        if street is None:
            street = self._get_street_name(static_ctx.stage)
        # Use provided street parameter if given, otherwise infer from stage
        
        # Process accurate action log for sequence features
        sequence_data = self._process_action_log_for_history(action_log, static_ctx)
        
        # Get final financial state at end of street
        stack_data = self._get_final_stack_state_for_history(static_ctx, dynamic_ctx)
        
        # Get hero-only additional features
        additional_data = self._get_final_additional_state_for_history(static_ctx, dynamic_ctx)
        
        # Combine all data and save to HistoryTracker
        final_snapshot = {**sequence_data, **stack_data, **additional_data}
        dynamic_ctx.history_tracker.save_snapshot(hand_key, street, final_snapshot)
    
    def calculate_sequence_history(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate sequence history features for any seat_id.
        
        SequenceHistory Features (per street):
        - Seat id checked x times
        - Seat id raised (includes initial bet) x times 
        - Seat id raised by x% of pot on average
        - Seat id overbet (>100% of pot raise) at least once
        - Seat id largebet (>70% of pot raise/bet) x times
        - Seat id was first raiser/bettor
        - Seat id was last raiser/bettor
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            prefix = f"seat_{seat_id}_"
            
            checked_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}checked_count", 0.0)
            raised_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}raised_count", 0.0)
            avg_raise_pct = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}avg_raise_pct", 0.0)
            overbet_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}overbet_count", 0.0)
            largebet_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}largebet_count", 0.0)
            is_first_bettor = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}is_first_bettor", 0.0)
            is_last_bettor = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}is_last_bettor", 0.0)
            
            result[f"{street}_checked_count"] = checked_count
            result[f"{street}_raised_count"] = raised_count
            result[f"{street}_avg_raise_pct"] = avg_raise_pct
            result[f"{street}_overbet_once"] = overbet_count  # Renamed to match spec
            result[f"{street}_largebet_count"] = largebet_count
            result[f"{street}_was_first_bettor"] = is_first_bettor
            result[f"{street}_was_last_bettor"] = is_last_bettor
        
        return result
    
    def calculate_stack_history(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate stack history features for any seat_id.
        
        StackHistory Features (per street):
        - Seat id stack in BB
        - Seat id pot size ratio (pot size / total money)
        - Seat id call cost ratio (to call / stack)
        - Seat id pot odds (to call / (pot + to call))
        - Seat id stack size ratio (stack / total money)
        - Seat id amount committed this street in BB
        - Seat id total commitment (across all streets so far: % of stack committed)
        - Seat id total commitment (across all streets so far: in BB)
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            prefix = f"seat_{seat_id}_"
            
            stack_in_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}stack_in_bb", 0.0)
            pot_size_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}pot_size_ratio", 0.0)
            call_cost_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}call_cost_ratio", 0.0)
            pot_odds = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}pot_odds", 0.0)
            stack_size_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}stack_size_ratio", 0.0)
            current_street_commitment_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}current_street_commitment_bb", 0.0)
            total_commitment = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}total_commitment", 0.0)
            total_commitment_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}total_commitment_bb", 0.0)
            
            result[f"{street}_stack_in_bb"] = stack_in_bb
            result[f"{street}_pot_size_ratio"] = pot_size_ratio
            result[f"{street}_call_cost_ratio"] = call_cost_ratio
            result[f"{street}_pot_odds"] = pot_odds
            result[f"{street}_stack_size_ratio"] = stack_size_ratio
            result[f"{street}_current_street_commitment_bb"] = current_street_commitment_bb
            result[f"{street}_total_commitment"] = total_commitment
            result[f"{street}_total_commitment_bb"] = total_commitment_bb
        
        return result
    
    def calculate_additional_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate additional hero-only features from history.
        
        Additional Features (per street, hero only):
        - Effective stack to pot ratio (min(stacks) / pot)
        - Implied odds
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            effective_spr = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, "effective_spr", 0.0)
            implied_odds = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, "implied_odds", 0.0)
            
            result[f"{street}_effective_spr"] = effective_spr
            result[f"{street}_implied_odds"] = implied_odds
        
        return result
    
    # =========================================================================
    # INTERNAL HELPERS for accurate data processing
    # =========================================================================
    
    def _process_action_log_for_history(self, action_log: list, static_ctx: StaticContext) -> dict:
        """
        Process completed ActionSequencer log to generate accurate historical summary.
        
        Args:
            action_log: List of (seat_id, action_type, amount) tuples from ActionSequencer
            static_ctx: Static context for pot information
            
        Returns:
            Dict with accurate sequence features for all players
        """
        summary = {}
        pot_at_start = getattr(static_ctx.game_state, 'starting_pot_this_round', static_ctx.pot)
        
        # Find all aggressive actions for first/last bettor detection
        aggressive_actions = [(s, a, amt) for s, a, amt in action_log if a in ['bet', 'raise']]
        aggressors = [s for s, a, amt in aggressive_actions]
        
        for seat_id in range(static_ctx.num_players):
            prefix = f"seat_{seat_id}_"
            
            # Accurately count actions for this seat
            seat_actions = [(s, a, amt) for s, a, amt in action_log if s == seat_id]
            
            checked_count = float(sum(1 for s, a, amt in seat_actions if a == 'check'))
            
            seat_aggressive_actions = [(s, a, amt) for s, a, amt in seat_actions if a in ['bet', 'raise']]
            raised_count = float(len(seat_aggressive_actions))
            
            # Calculate average raise percentage
            if seat_aggressive_actions:
                total_raise_amount = sum(amt for s, a, amt in seat_aggressive_actions)
                avg_raise_pct = (total_raise_amount / len(seat_aggressive_actions)) / max(pot_at_start, 1)
                avg_raise_pct = min(avg_raise_pct, 3.0) / 3.0  # Normalize to 0-1
            else:
                avg_raise_pct = 0.0
            
            # Count overbets and large bets accurately
            overbet_count = float(sum(1 for s, a, amt in seat_aggressive_actions if amt > pot_at_start))
            largebet_count = float(sum(1 for s, a, amt in seat_aggressive_actions if amt > pot_at_start * 0.7))
            
            # First and last bettor (accurate from action sequence)
            is_first_bettor = 1.0 if aggressors and aggressors[0] == seat_id else 0.0
            is_last_bettor = 1.0 if aggressors and aggressors[-1] == seat_id else 0.0
            
            # Save to summary
            summary[f"{prefix}checked_count"] = checked_count
            summary[f"{prefix}raised_count"] = raised_count
            summary[f"{prefix}avg_raise_pct"] = avg_raise_pct
            summary[f"{prefix}overbet_count"] = overbet_count
            summary[f"{prefix}largebet_count"] = largebet_count
            summary[f"{prefix}is_first_bettor"] = is_first_bettor
            summary[f"{prefix}is_last_bettor"] = is_last_bettor
        
        return summary
    
    def _get_final_stack_state_for_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Get final financial state at the end of a street for ALL players.
        """
        summary = {}
        current_bets = static_ctx.game_state.current_bets
        stacks = static_ctx.game_state.stacks
        pot = static_ctx.pot
        max_bet = max(current_bets) if current_bets else 0
        
        big_blind = dynamic_ctx.history_tracker.get_big_blind()
        player_total_invested = dynamic_ctx.history_tracker.get_player_investment_all()
        total_money = sum(stacks) + pot
        
        for seat_id in range(static_ctx.num_players):
            prefix = f"seat_{seat_id}_"
            stack = stacks[seat_id]
            current_bet = current_bets[seat_id]
            to_call = max_bet - current_bet
            
            # Stack metrics
            stack_in_bb = stack / max(big_blind, 1)
            pot_size_ratio = pot / max(total_money, 1)
            call_cost_ratio = to_call / max(stack, 1)
            pot_odds = to_call / max(pot + to_call, 1) if to_call > 0 else 0.0
            stack_size_ratio = stack / max(total_money, 1)
            
            # Total commitment calculation
            if seat_id < len(player_total_invested):
                total_invested = player_total_invested[seat_id]
                starting_stack = stack + total_invested
                total_commitment = total_invested / max(starting_stack, 1)
                total_commitment_bb = total_invested / max(big_blind, 1)
            else:
                total_commitment = 0.0
                total_commitment_bb = 0.0
            
            summary[f"{prefix}stack_in_bb"] = stack_in_bb
            summary[f"{prefix}pot_size_ratio"] = pot_size_ratio
            summary[f"{prefix}call_cost_ratio"] = call_cost_ratio   # currently, anything with to_call is always 0 at end of street. remove from history?
            summary[f"{prefix}pot_odds"] = pot_odds                 # also uses to_call
            summary[f"{prefix}stack_size_ratio"] = stack_size_ratio
            summary[f"{prefix}current_street_commitment_bb"] = current_bet / max(big_blind, 1)
            summary[f"{prefix}total_commitment"] = total_commitment
            summary[f"{prefix}total_commitment_bb"] = total_commitment_bb
        
        return summary
    
    def _get_final_additional_state_for_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Get final hero-only metrics at the end of a street.
        """
        summary = {}
        stacks = static_ctx.game_state.stacks
        pot = static_ctx.pot
        
        # Effective SPR
        if len(stacks) >= 2 and pot > 0:
            effective_stack = min(stacks)
            effective_spr = effective_stack / pot
        else:
            effective_spr = 0.0
        
        # Implied odds using CurrentStreetAnalyzer's advanced calculation
        implied_odds = self.current_street_analyzer._calculate_current_implied_odds(static_ctx, dynamic_ctx)
        
        summary["effective_spr"] = effective_spr
        summary["implied_odds"] = implied_odds
        
        return summary
    
    
    def _get_street_name(self, stage: int) -> str:
        """Convert stage number to street name."""
        stage_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return stage_map.get(stage, 'preflop')
    
