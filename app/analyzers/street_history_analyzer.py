# app/analyzers/street_history_analyzer.py
# History tracked features with proper HistoryTracker integration
# Uses ActionSequencer for accurate historical data - no more guessing!

import copy
from feature_contexts import StaticContext, DynamicContext
from .current_street_analyzer import CurrentStreetAnalyzer


class StreetHistoryAnalyzer:
    """
    Analyzer for features that require history tracking across streets.
    
    WRITE Operation: Takes ActionSequencer's complete log and saves accurate summaries.
    READ Operation: Retrieves previously saved summaries for historical context.
    
    Key Principle: Use ActionSequencer as the single source of truth for betting sequences.
    """
    
    def __init__(self):
        # Reuse CurrentStreetAnalyzer logic instead of duplicating
        self.current_street_analyzer = CurrentStreetAnalyzer()
    
    def save_street_snapshot(self, action_log: list, static_ctx: StaticContext, dynamic_ctx: DynamicContext, 
                             strategic_features: dict = None, street: str = None) -> None:
        """
        Save accurate historical data using ActionSequencer's complete log.
        Called ONCE at the end of each street with the final, complete action sequence.
        
        Args:
            action_log: Complete action log from ActionSequencer for this street
            static_ctx: Static game context
            dynamic_ctx: Dynamic context with HistoryTracker
            strategic_features: Optional dict of strategic features calculated by StrategicAnalyzer
            street: Explicit street name (overrides stage detection)
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        if street is None:
            street = self._get_street_name(static_ctx.stage)
        # Use provided street parameter if given, otherwise infer from stage
        
        # Process accurate action log for sequence features
        sequence_data = self._process_action_log_for_history(action_log, static_ctx, dynamic_ctx)
        
        # Get final financial state at end of street (needs action_log for final call reconstruction)
        stack_data = self._get_final_stack_state_for_history(action_log, static_ctx, dynamic_ctx)
        
        # Get self-only additional features
        additional_data = self._get_final_additional_state_for_history(static_ctx, dynamic_ctx)
        
        # Combine all data and save to HistoryTracker
        final_snapshot = {**sequence_data, **stack_data, **additional_data}
        
        # If strategic features were calculated, add them to the snapshot
        if strategic_features:
            final_snapshot.update(strategic_features)
        
        # --- Add raw action log for deep pattern analysis ---
        final_snapshot['raw_action_log'] = action_log
        # ------
        
        dynamic_ctx.history_tracker.save_snapshot(hand_key, street, final_snapshot)
    
    def calculate_sequence_history(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate sequence history features for any seat_id.
        
        * needs extra care (not simple snapshot)

        SequenceHistory Features (per street):
        - Seat id checked x times
        - Seat id called x times
        - Seat id raised (includes initial bet) x times 
        - Seat id raised BY x% of current pot (average*)
        - Seat id raised TO x% of street's starting pot (average*)
        - Seat id overbet (>100% of pot raise/bet) x times
        - Seat id largebet (>70% of pot raise/bet) x times
        - Seat id did check raise, donk, 3bet, float, probe bet [all monotonic]

        - History only: Seat id was first raiser/bettor
        -               Seat id was last raiser/bettor
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            prefix = f"seat_{seat_id}_"
            
            checked_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}checked_count", 0.0)
            called_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}called_count", 0.0)
            raised_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}raised_count", 0.0)
            avg_raise_pct_of_pot = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}avg_raise_pct_of_pot", 0.0)
            aggro_commit_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}aggro_commit_ratio", 0.0)
            overbet_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}overbet_count", 0.0)
            largebet_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}largebet_count", 0.0)
            smallbet_count = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}smallbet_count", 0.0)
            is_first_bettor = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}is_first_bettor", 0.0)
            is_last_bettor = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}is_last_bettor", 0.0)
            
            # Strategic features
            did_check_raise = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_check_raise", 0.0)
            did_donk_bet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_donk_bet", 0.0)
            did_3bet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_3bet", 0.0)
            did_float_bet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_float_bet", 0.0)
            did_probe_bet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_probe_bet", 0.0)
            did_cbet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_cbet", 0.0)
            did_go_all_in = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_go_all_in", 0.0)
            did_open_overbet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_open_overbet", 0.0)
            did_open_largebet = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}did_open_largebet", 0.0)
            
            result[f"{street}_checked_count"] = checked_count
            result[f"{street}_called_count"] = called_count
            result[f"{street}_raised_count"] = raised_count
            result[f"{street}_avg_raise_pct_of_pot"] = avg_raise_pct_of_pot
            result[f"{street}_aggro_commit_ratio"] = aggro_commit_ratio
            result[f"{street}_overbet_count"] = overbet_count
            result[f"{street}_largebet_count"] = largebet_count
            result[f"{street}_smallbet_count"] = smallbet_count
            result[f"{street}_was_first_bettor"] = is_first_bettor
            result[f"{street}_was_last_bettor"] = is_last_bettor
            # Strategic features
            result[f"{street}_did_check_raise"] = did_check_raise
            result[f"{street}_did_donk_bet"] = did_donk_bet
            result[f"{street}_did_3bet"] = did_3bet
            result[f"{street}_did_float_bet"] = did_float_bet
            result[f"{street}_did_probe_bet"] = did_probe_bet
            result[f"{street}_did_cbet"] = did_cbet
            result[f"{street}_did_go_all_in"] = did_go_all_in
            result[f"{street}_did_open_overbet"] = did_open_overbet
            result[f"{street}_did_open_largebet"] = did_open_largebet
        
        return result
    
    def calculate_stack_history(self, seat_id: int, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate stack history features for any seat_id.
        
        * needs extra care (cannot simply take final snapshot)

        StackHistory Features (per street):
        - Seat id stack in BB
        - Seat id pot size ratio (pot size / total money)
        - Seat id final* call cost ratio (to call / stack) [RECONSTRUCTED FROM ACTION LOG]
        - Seat id final call* pot odds (to call / (pot + to call)) [RECONSTRUCTED FROM ACTION LOG]
        - Seat id stack size ratio (stack / total money)
        - Seat id amount committed this street in BB
        - Seat id amount committed this street / starting pot this street
        - Seat id amount committed this street / seat id starting stack this street
        - Seat id total commitment (cumulative % so far)
        - Seat id total commitment in BB
        - Seat id stack is smaller than pot
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            prefix = f"seat_{seat_id}_"
            
            stack_in_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}stack_in_bb", 0.0)
            pot_size_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}pot_size_ratio", 0.0)
            final_call_cost_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}final_call_cost_ratio", 0.0)
            final_call_pot_odds = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}final_call_pot_odds", 0.0)
            stack_size_ratio = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}stack_size_ratio", 0.0)
            current_street_commitment_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}current_street_commitment_bb", 0.0)
            current_street_commitment_vs_starting_pot = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}current_street_commitment_vs_starting_pot", 0.0)
            current_street_commitment_vs_starting_stack = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}current_street_commitment_vs_starting_stack", 0.0)
            total_commitment_pct = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}total_commitment_pct", 0.0)
            total_commitment_bb = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, f"{prefix}total_commitment_bb", 0.0)
            
            result[f"{street}_stack_in_bb"] = stack_in_bb
            result[f"{street}_pot_size_ratio"] = pot_size_ratio
            result[f"{street}_final_call_cost_ratio"] = final_call_cost_ratio
            result[f"{street}_final_call_pot_odds"] = final_call_pot_odds
            result[f"{street}_stack_size_ratio"] = stack_size_ratio
            result[f"{street}_current_street_commitment_bb"] = current_street_commitment_bb
            result[f"{street}_current_street_commitment_vs_starting_pot"] = current_street_commitment_vs_starting_pot
            result[f"{street}_current_street_commitment_vs_starting_stack"] = current_street_commitment_vs_starting_stack
            result[f"{street}_total_commitment_pct"] = total_commitment_pct
            result[f"{street}_total_commitment_bb"] = total_commitment_bb
        
        return result
    
    def calculate_additional_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Calculate additional self-only features from history.
        
        * needs care (cannot take final snapshot; use the final to_call)

        Additional Features (per street, self only):
        - Effective stack to pot ratio (min(stacks) / pot)
        - Final* implied odds
        - Hand strength
        - Equity vs range
        
        Returns features for preflop, flop, turn, river from HistoryTracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        
        result = {}
        for street in ["preflop", "flop", "turn", "river"]:
            effective_spr = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, "effective_spr", 0.0)
            hand_strength = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, "hand_strength", 0.0)
            
            result[f"{street}_effective_spr"] = effective_spr
            result[f"{street}_hand_strength"] = hand_strength
        
        return result
    
    def calculate_strategic_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Reads strategic history features for the self from the tracker.
        
        Returns all strategic features across all streets that were calculated
        by the StrategicAnalyzer and saved to the history tracker.
        """
        hand_key = f"hand_{dynamic_ctx.history_tracker.get_hand_number()}"
        result = {}
        
        # This mirrors the structure of the StrategicHistoryFeatures dataclass
        strategic_feature_names = [
            "implied_odds", "hand_vs_range", "range_vs_range_equity", "fold_equity",
            "showdown_equity", "reverse_implied_odds", "range_vs_range", "future_payoff", "playability"
        ]
        
        for street in ["preflop", "flop", "turn", "river"]:
            for feature_name in strategic_feature_names:
                # Construct the key as it appears in the final schema (e.g., "flop_fold_equity")
                output_key = f"{street}_{feature_name}"
                # Read the base name from the snapshot (e.g., "fold_equity")
                value = dynamic_ctx.history_tracker.get_feature_value(hand_key, street, feature_name, 0.0)
                result[output_key] = value
        
        return result
    
    # =========================================================================
    # INTERNAL HELPERS for accurate data processing
    # =========================================================================
    
    def _process_action_log_for_history(self, action_log: list, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Reuse CurrentStreetAnalyzer logic instead of duplicating calculations.
        
        Create a corrected context for historical calculation, as the passed-in
        static_ctx reflects the END of the street, not the beginning.
        
        Args:
            action_log: List of (seat_id, action_type, amount) tuples from ActionSequencer
            static_ctx: Static context for pot information
            dynamic_ctx: Dynamic context for history tracker
            
        Returns:
            Dict with accurate sequence features for all players (reusing existing logic)
        """
        from .action_sequencer import ActionSequencer
        
        # Create a corrected context for historical calculation, as the passed-in
        # static_ctx reflects the END of the street, not the beginning.
        
        # 1. Calculate the true starting pot by working backward from the final pot.
        final_pot = static_ctx.pot
        total_wagered_this_street = sum(amount for _, _, amount in action_log if amount)
        true_starting_pot = final_pot - total_wagered_this_street
        
        # 2. Create a deep copy of the game state to avoid side effects.
        corrected_game_state = copy.deepcopy(static_ctx.game_state)
        
        # 3. Overwrite the stale value with the correct one.
        corrected_game_state.starting_pot_this_round = true_starting_pot
        
        # 4. Create a new, corrected StaticContext for this historical analysis.
        corrected_static_ctx = StaticContext(game_state=corrected_game_state, seat_id=static_ctx.seat_id)
        
        # Create temporary ActionSequencer from completed log
        temp_sequencer = ActionSequencer()
        temp_sequencer.current_street_log = action_log
        
        summary = {}
        
        # Use existing CurrentStreetAnalyzer for ALL players
        for seat_id in range(static_ctx.num_players):
            prefix = f"seat_{seat_id}_"
            
            # 5. Pass the CORRECTED context to the calculation function.
            sequence_features = self.current_street_analyzer.calculate_current_street_sequence(
                seat_id, corrected_static_ctx, dynamic_ctx, temp_sequencer
            )
            
            # Add the features with history prefix, filtering out transient features
            for key, value in sequence_features.items():
                if not key.startswith('is_facing'):  # Filter out real-time facing features
                    summary[f"{prefix}{key}"] = value
            
            # Add history-specific features (first/last bettor)
            aggressive_actions = [s for s, a, _ in action_log if a in ['bet', 'raise']]
            summary[f"{prefix}is_first_bettor"] = 1.0 if aggressive_actions and aggressive_actions[0] == seat_id else 0.0
            summary[f"{prefix}is_last_bettor"] = 1.0 if aggressive_actions and aggressive_actions[-1] == seat_id else 0.0
        
        return summary
    
    def _get_final_stack_state_for_history(self, action_log: list, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Reuse CurrentStreetAnalyzer stack logic with history-specific additions.
        Correctly calculate commitment features using action_log instead of stale current_bets.
        """
        summary = {}
        
        # Calculate special "final call" metrics using reconstruction logic
        final_call_metrics = self._calculate_final_call_metrics(action_log, static_ctx, dynamic_ctx)
        
        # Get big blind for normalization
        big_blind = dynamic_ctx.history_tracker.get_big_blind()
        
        # Calculate true starting pot (same fix as in _process_action_log_for_history)
        final_pot = static_ctx.pot
        total_wagered_this_street = sum(amount for _, _, amount in action_log if amount)
        true_starting_pot = final_pot - total_wagered_this_street
        
        # Use existing CurrentStreetAnalyzer for ALL players
        for seat_id in range(static_ctx.num_players):
            prefix = f"seat_{seat_id}_"
            
            # REUSE existing stack calculation logic (no ActionSequencer for historical data)
            stack_features = self.current_street_analyzer.calculate_current_street_stack(
                seat_id, static_ctx, dynamic_ctx, None
            )
            
            # Add the features with history prefix, filtering out incorrect ones due to stale state
            for key, value in stack_features.items():
                if key not in ['call_cost_ratio', 'pot_odds', 'current_street_commitment_bb', 
                               'current_street_commitment_vs_starting_pot', 
                               'current_street_commitment_vs_starting_stack']:
                    summary[f"{prefix}{key}"] = value
            
            # Re-calculate commitment features using the reliable action_log
            
            # 1. Sum the total bet for this player on this street from the log
            player_total_bet_this_street = sum(
                amount for s_id, act_type, amount in action_log 
                if s_id == seat_id and amount
            )
            
            # 2. Re-calculate the commitment features using this correct value
            commitment_bb = min((player_total_bet_this_street / max(big_blind, 1)) / 200.0, 1.0)
            
            commitment_vs_pot = min(player_total_bet_this_street / max(true_starting_pot, 1), 5.0) / 5.0
            
            # Calculate starting stack for this street (stack + what they've bet this street)
            current_stack_bb = stack_features.get("stack_in_bb", 0.0)
            starting_stack_street = (current_stack_bb * 200 * big_blind) + player_total_bet_this_street
            commitment_vs_stack = player_total_bet_this_street / max(starting_stack_street, 1)
            
            # 3. Add the corrected values to the summary
            summary[f"{prefix}current_street_commitment_bb"] = commitment_bb
            summary[f"{prefix}current_street_commitment_vs_starting_pot"] = commitment_vs_pot
            summary[f"{prefix}current_street_commitment_vs_starting_stack"] = commitment_vs_stack
            
            # Add the special, reconstructed "final call" metrics
            player_call_metrics = final_call_metrics.get(seat_id, {'pot_odds': 0.0, 'call_cost_ratio': 0.0, 'implied_odds': 0.0})
            summary[f"{prefix}final_call_pot_odds"] = player_call_metrics['pot_odds']
            summary[f"{prefix}final_call_cost_ratio"] = player_call_metrics['call_cost_ratio']
            # ✅ Implied odds is now correctly stored with other final call metrics
            summary[f"{prefix}final_call_implied_odds"] = player_call_metrics['implied_odds']
        
        return summary
    
    def _get_final_additional_state_for_history(self, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Explicitly calculates features from the self's perspective,
        regardless of who acted last on the street.
        """
        self_seat_id = 0
        self_static_ctx = StaticContext(game_state=static_ctx.game_state, seat_id=self_seat_id)
        
        additional_features = self.current_street_analyzer.calculate_current_street_additional(
            self_static_ctx, dynamic_ctx
        )
        
        # 4. Extract and return the accurately calculated features for the self
        snapshot_features = {
            "effective_spr": additional_features.get("effective_spr", 0.0),
            "hand_strength": additional_features.get("hand_strength", 0.0)
        }
        return snapshot_features
    
    def _calculate_final_call_metrics(self, action_log: list, static_ctx: StaticContext, dynamic_ctx: DynamicContext) -> dict:
        """
        Reconstructs the betting round to find metrics at the moment of each player's final call.
        TODO: INCLUDE IMPLIED ODDS FROM STRATEGIC ANALYZER.
        """
        final_call_metrics = {}
        
        # Get the state at the START of the street
        pot_at_start = getattr(static_ctx.game_state, 'starting_pot_this_round', static_ctx.pot)
        sim_pot = float(pot_at_start)
        sim_bets = [0] * static_ctx.num_players
        
        # Calculate each player's stack at the start of this specific street
        starting_stacks_this_street = [s + c for s, c in zip(static_ctx.game_state.stacks, static_ctx.game_state.current_bets)]

        # Simulate the betting round action by action
        for seat_id, action_type, amount in action_log:
            # If a player calls, capture the "at that moment" metrics
            if action_type == 'call':
                max_bet_at_moment = max(sim_bets)
                to_call_at_moment = max_bet_at_moment - sim_bets[seat_id]
                
                # Reconstruct the player's stack at the moment of the call
                stack_at_moment = starting_stacks_this_street[seat_id] - sim_bets[seat_id]

                # Calculate the metrics using the reconstructed state
                pot_odds = to_call_at_moment / max(sim_pot + to_call_at_moment, 1)
                call_cost_ratio = to_call_at_moment / max(stack_at_moment, 1)
                
                # ✅ Calculate implied odds using existing logic with reconstructed state
                # For now, use a simpler approach - calculate implied odds directly
                # This avoids complex GameState reconstruction
                implied_odds = 0.0  # TODO: Implement proper implied odds calculation for history
                
                # Store it (if player calls again, this overwrites, capturing FINAL call)
                final_call_metrics[seat_id] = {
                    'pot_odds': pot_odds, 
                    'call_cost_ratio': call_cost_ratio,
                    'implied_odds': implied_odds
                }
            
            # Update the simulated state for the next action
            if amount:
                sim_pot += amount
                sim_bets[seat_id] += amount
                
        return final_call_metrics
    
    def _get_street_name(self, stage: int) -> str:
        """Convert stage number to street name."""
        stage_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return stage_map.get(stage, 'preflop')
    
