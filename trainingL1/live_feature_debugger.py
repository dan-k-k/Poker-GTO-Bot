# trainingL1/live_feature_debugger.py
# Live feature debugging during training - formats features into human-readable strings
# for integration with hand_histories.log

from poker_feature_schema import PokerFeatureSchema
from typing import List, Optional
import numpy as np

class LiveFeatureDebugger:
    """
    Formats the feature schema into human-readable strings for logging during live training.
    Integrates with the hand_histories.log system to show what the AI actually sees.
    """
    
    def __init__(self):
        self.suits = ['♣', '♦', '♥', '♠']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    def card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to readable string like 'A♠'."""
        if card_id < 0 or card_id > 51:
            return "??"
        rank = self.ranks[card_id // 4]
        suit = self.suits[card_id % 4]
        return f"{rank}{suit}"
    
    def extract_cards_from_one_hot(self, schema: PokerFeatureSchema, card_type: str = "hole") -> List[str]:
        """Extract card strings from one-hot encoding in schema."""
        cards = []
        if card_type == "hole":
            # Extract from my_hand hole card features
            for i in range(52):
                if hasattr(schema.my_hand, f'hole_{i}') and getattr(schema.my_hand, f'hole_{i}') == 1.0:
                    cards.append(self.card_id_to_string(i))
        elif card_type == "board":
            # Extract from board card features
            for i in range(52):
                if hasattr(schema.board, f'card_{i}') and getattr(schema.board, f'card_{i}') == 1.0:
                    cards.append(self.card_id_to_string(i))
        return cards
    
    def get_made_hand_string(self, schema: PokerFeatureSchema) -> str:
        """Get the best made hand from monotonic flags."""
        hand = schema.my_hand
        
        # Check from highest to lowest
        if hand.royal_flush == 1.0:
            return "Royal Flush"
        elif hand.straight_flush == 1.0:
            return "Straight Flush"
        elif hand.at_least_four_kind == 1.0:
            return "Four of a Kind"
        elif hand.at_least_full_house == 1.0:
            return "Full House"
        elif hand.at_least_flush == 1.0:
            return "Flush"
        elif hand.at_least_straight == 1.0:
            return "Straight"
        elif hand.at_least_three_kind == 1.0:
            return "Three of a Kind"
        elif hand.at_least_two_pair == 1.0:
            return "Two Pair"
        elif hand.at_least_pair == 1.0:
            return "Pair"
        else:
            return "High Card"
    
    def get_kickers_string(self, schema: PokerFeatureSchema) -> str:
        """Get kicker information in readable format."""
        hand = schema.my_hand
        kickers = []
        
        for i in range(1, 6):  # Up to 5 kickers
            kicker_val = getattr(hand, f'kicker_{i}', 0)
            if kicker_val > 0:
                # Un-normalize: feature stores (rank+1)/13, convert back to rank
                if kicker_val <= 1.0:  # Normalized value
                    kicker_rank_index = int(kicker_val * 13.0) - 1
                    if 0 <= kicker_rank_index < 13:
                        kickers.append(self.ranks[kicker_rank_index])
                else:  # Direct rank value (fallback)
                    if 1 <= kicker_val <= 13:
                        kickers.append(self.ranks[int(kicker_val) - 1])
        
        return ' '.join(kickers) if kickers else "None"
    
    def get_position_string(self, schema: PokerFeatureSchema) -> str:
        """Get position in readable format."""
        pos = schema.self_current_position
        
        positions = []
        if pos.is_dealer == 1.0:
            positions.append("Dealer")
        if pos.is_sb == 1.0:
            positions.append("SB")
        if pos.is_bb == 1.0:
            positions.append("BB")
        if pos.is_OOP == 1.0 and not positions:
            positions.append("OOP")
        
        return ', '.join(positions) if positions else "Unknown"
    
    def get_street_string(self, schema: PokerFeatureSchema) -> str:
        """Get current street in readable format."""
        stage = schema.current_stage
        
        if stage.is_river == 1.0:
            return "River"
        elif stage.is_turnplus == 1.0:
            return "Turn"
        elif stage.is_flopplus == 1.0:
            return "Flop"
        elif stage.is_preflopplus == 1.0:
            return "Preflop"
        else:
            return "Unknown"
    
    def get_facing_string(self, schema: PokerFeatureSchema) -> str:
        """Get what the player is currently facing."""
        # Get facing features from additional features (moved from sequence features)
        additional = schema.current_additional
        
        facing = []
        if additional.is_facing_4betplus == 1.0:
            facing.append("4-bet+")
        elif additional.is_facing_3bet == 1.0:
            facing.append("3-bet")
        elif additional.is_facing_raise == 1.0:
            facing.append("Raise")
        elif additional.is_facing_bet == 1.0:
            facing.append("Bet")
        elif additional.is_facing_check == 1.0:
            facing.append("Check")
        
        return ', '.join(facing) if facing else "Nothing"
    
    def format_concise_features(self, schema: PokerFeatureSchema, player_name: str = "P0") -> str:
        """
        Format the most important features into a concise, single-line summary.
        Perfect for action logging in hand histories.
        """
        # Get essential information
        hole_cards = self.extract_cards_from_one_hot(schema, "hole")
        made_hand = self.get_made_hand_string(schema)
        hand_strength = schema.current_additional.hand_strength
        hand_vs_range = schema.current_strategic.hand_vs_range
        
        # Stack and pot info
        stack = schema.self_current_stack
        actual_stack_bb = stack.stack_in_bb * 200.0 if stack.stack_in_bb <= 1.0 else stack.stack_in_bb
        pot_odds = stack.pot_odds
        commitment = stack.total_commitment_pct
        
        # Position and facing
        position = self.get_position_string(schema)
        facing = self.get_facing_string(schema)
        
        # Format as single line
        hole_str = ' '.join(hole_cards) if hole_cards else "??"
        
        return f"    [{player_name}: {hole_str} ({made_hand}) | Str: {hand_strength:.1%} | Eq: {hand_vs_range:.1%} | Stack: {actual_stack_bb:.0f}BB | Odds: {pot_odds:.2f} | T. Commit: {commitment:.1%} | Pos: {position} | Facing: {facing}]"
    
    def format_detailed_features(self, schema: PokerFeatureSchema, player_name: str = "P0") -> str:
        """
        Format a comprehensive multi-line feature summary for detailed analysis.
        Use this for key decision points or when debugging specific hands.
        """
        lines = [f"\n    === {player_name} FEATURE ANALYSIS ==="]
        
        # Hand information
        hole_cards = self.extract_cards_from_one_hot(schema, "hole")
        board_cards = self.extract_cards_from_one_hot(schema, "board")
        made_hand = self.get_made_hand_string(schema)
        kickers = self.get_kickers_string(schema)
        
        lines.append(f"     Cards: {' '.join(hole_cards)} (hole) + {' '.join(board_cards)} (board)")
        lines.append(f"     Hand: {made_hand} | Kickers: {kickers}")
        
        # Strength metrics
        hand_strength = schema.current_additional.hand_strength
        hand_vs_range = schema.current_strategic.hand_vs_range if hasattr(schema.current_strategic, 'hand_vs_range') else 0.0
        lines.append(f"    📊 Strength: {hand_strength:.1%} (raw) | {hand_vs_range:.1%} (vs range)")
        
        # Delta features (if applicable)
        additional_deltas = schema.current_additional
        strategic_deltas = schema.current_strategic
        equity_delta = strategic_deltas.equity_delta if hasattr(strategic_deltas, 'equity_delta') else 0.0
        if equity_delta != 0 or additional_deltas.spr_delta != 0 or additional_deltas.pot_size_delta != 0:
            lines.append(f"    📈 Deltas: Equity {equity_delta:+.1%} | SPR {additional_deltas.spr_delta:+.2f} | Pot {additional_deltas.pot_size_delta:+.1f}BB")
        
        # Stack and pot dynamics
        stack = schema.self_current_stack
        actual_stack_bb = stack.stack_in_bb * 200.0 if stack.stack_in_bb <= 1.0 else stack.stack_in_bb
        lines.append(f"     Stack: {actual_stack_bb:.1f}BB | Pot Odds: {stack.pot_odds:.3f} | Call Cost: {stack.call_cost_ratio:.3f}")
        lines.append(f"     Commitment: {stack.total_commitment_pct:.1%} total | {stack.current_street_commitment_vs_starting_stack:.1%} this street")
        
        # Strategic features
        additional = schema.current_additional
        lines.append(f"     Strategic: SPR {additional.effective_spr:.2f} | Implied Odds {additional.implied_odds:.3f}")
        
        # Position and action context
        position = self.get_position_string(schema)
        street = self.get_street_string(schema)
        facing = self.get_facing_string(schema)
        lines.append(f"     Context: {street} | {position} | Facing: {facing}")
        
        # Action history (current street)
        seq = schema.self_current_sequence
        if seq.checked_count > 0 or seq.raised_count > 0:
            aggression = seq.raised_count / (seq.checked_count + seq.raised_count) if (seq.checked_count + seq.raised_count) > 0 else 0
            lines.append(f"     Actions: Check×{seq.checked_count:.0f} | Raise×{seq.raised_count:.0f} | Aggression: {aggression:.1%}")
        
        # Strategic actions on current street
        strategic_actions = self.get_current_street_strategic_actions(schema)
        if strategic_actions:
            lines.append(f"     Strategic: {strategic_actions}")
        
        lines.append(f"    =======================================")
        
        return '\n'.join(lines)
    
    def format_delta_analysis(self, schema: PokerFeatureSchema) -> Optional[str]:
        """
        Format delta feature analysis if significant changes occurred.
        Only returns text if there are meaningful deltas to report.
        """
        deltas = schema.current_additional
        
        # Check if any deltas are significant
        strategic_deltas = schema.current_strategic
        equity_delta = strategic_deltas.equity_delta if hasattr(strategic_deltas, 'equity_delta') else 0.0
        significant_equity = abs(equity_delta) > 0.05  # 5% change
        significant_spr = abs(deltas.spr_delta) > 0.5  # 0.5 SPR change
        significant_pot = abs(deltas.pot_size_delta) > 1.0  # 1BB change
        
        if not (significant_equity or significant_spr or significant_pot):
            return None
        
        lines = ["    💫 SIGNIFICANT CHANGES FROM LAST STREET:"]
        
        if significant_equity:
            direction = "improved" if equity_delta > 0 else "worsened"
            lines.append(f"       Equity {direction} by {abs(equity_delta):.1%}")
        
        if significant_spr:
            direction = "increased" if deltas.spr_delta > 0 else "decreased"
            lines.append(f"       SPR {direction} by {abs(deltas.spr_delta):.2f}")
        
        if significant_pot:
            lines.append(f"       Pot grew by {deltas.pot_size_delta:.1f} BB")
        
        return '\n'.join(lines)
    
    def format_opponent_read(self, schema: PokerFeatureSchema) -> str:
        """
        Format opponent analysis from available features.
        """
        opp_stack = schema.opponent_current_stack
        opp_seq = schema.opponent_current_sequence
        
        # Basic opponent info
        opp_stack_bb = opp_stack.stack_in_bb * 200.0 if opp_stack.stack_in_bb <= 1.0 else opp_stack.stack_in_bb
        opp_commitment = opp_stack.total_commitment_pct
        
        # Opponent aggression this street
        opp_total_actions = opp_seq.checked_count + opp_seq.raised_count
        opp_aggression = opp_seq.raised_count / opp_total_actions if opp_total_actions > 0 else 0
        
        # Get opponent's strategic actions this street
        opp_strategic = self.get_opponent_strategic_actions(schema)
        strategic_text = f" | Strategic: {opp_strategic}" if opp_strategic else ""
        
        return f"     Opponent: {opp_stack_bb:.0f}BB stack | {opp_commitment:.1%} committed | {opp_aggression:.1%} aggression{strategic_text}"
    
    def format_opponent_model_stats(self, schema: PokerFeatureSchema) -> str:
        """
        Format detailed opponent modeling statistics from the stats tracker.
        Shows core stats to verify the opponent modeling system is working.
        """
        opp_model = schema.opponent_model
        
        # Check if we have meaningful stats (sample size > 0)
        if opp_model.sample_size < 1.0:
            return "     Opponent Model: No statistical data yet"
        
        lines = [f"     OPPONENT MODEL STATS (Sample: {opp_model.sample_size:.0f} hands)"]
        
        # Core preflop stats
        lines.append(f"       Core: VPIP {opp_model.vpip:.1%} | PFR {opp_model.pfr:.1%} | 3-bet {opp_model.three_bet_preflop:.1%}")
        
        # Post-flop aggression
        lines.append(f"       Aggression: C-bet {opp_model.cbet_flop:.1%}/{opp_model.cbet_turn:.1%}/{opp_model.cbet_river:.1%} | Overall {opp_model.aggression_frequency:.1%}")
        
        # Strategic patterns (show if any are > 5%)
        strategic_patterns = []
        if opp_model.double_barrel > 0.05:
            strategic_patterns.append(f"Double {opp_model.double_barrel:.1%}")
        if opp_model.float_bet > 0.05:
            strategic_patterns.append(f"Float {opp_model.float_bet:.1%}")
        if opp_model.donk_bet_flop > 0.05:
            strategic_patterns.append(f"Donk {opp_model.donk_bet_flop:.1%}")
        if opp_model.checkraise_flop > 0.05:
            strategic_patterns.append(f"C/R {opp_model.checkraise_flop:.1%}")
        
        if strategic_patterns:
            lines.append(f"       Patterns: {' | '.join(strategic_patterns)}")
        
        # Fold tendencies
        lines.append(f"       Defense: Fold vs C-bet {opp_model.fold_to_cbet_flop:.1%}/{opp_model.fold_to_cbet_turn:.1%}/{opp_model.fold_to_cbet_river:.1%}")
        
        # Showdown stats (if available)
        if opp_model.wtsd > 0:
            lines.append(f"       Showdown: WTSD {opp_model.wtsd:.1%} | Win Rate {opp_model.showdown_win_rate:.1%}")
        
        # Bet sizing info
        if opp_model.avg_bet_size > 0:
            lines.append(f"       Sizing: Avg {opp_model.avg_bet_size:.1f} BB | Pot Ratio {opp_model.avg_pot_ratio:.2f}")
        
        # Per-street strategic actions (if any show patterns)
        street_patterns = []
        if opp_model.three_bet_flop > 0.02 or opp_model.three_bet_turn > 0.02:
            street_patterns.append(f"3-bet F/T/R: {opp_model.three_bet_flop:.1%}/{opp_model.three_bet_turn:.1%}/{opp_model.three_bet_river:.1%}")
        if opp_model.donk_bet_flop > 0.02 or opp_model.donk_bet_turn > 0.02:
            street_patterns.append(f"Donk F/T/R: {opp_model.donk_bet_flop:.1%}/{opp_model.donk_bet_turn:.1%}/{opp_model.donk_bet_river:.1%}")
        if opp_model.checkraise_flop > 0.02 or opp_model.checkraise_turn > 0.02:
            street_patterns.append(f"C/R F/T/R: {opp_model.checkraise_flop:.1%}/{opp_model.checkraise_turn:.1%}/{opp_model.checkraise_river:.1%}")
        
        if street_patterns:
            lines.append(f"       Street Patterns: {' | '.join(street_patterns)}")
        
        return '\n'.join(lines)
    
    def get_current_street_strategic_actions(self, schema: PokerFeatureSchema) -> str:
        """
        Get strategic actions performed on the current street (self only).
        Shows: check-raise, donk bet, 3-bet, float bet, probe bet [all monotonic]
        """
        seq = schema.self_current_sequence
        actions = []
        
        # Check each strategic action (they're all boolean/monotonic)
        if seq.did_check_raise == 1.0:
            actions.append("Check-raise")
        if seq.did_donk_bet == 1.0:
            actions.append("Donk bet")
        if seq.did_3bet == 1.0:
            actions.append("3-bet")
        if seq.did_float_bet == 1.0:
            actions.append("Float bet")
        if seq.did_probe_bet == 1.0:
            actions.append("Probe bet")
        
        return ' | '.join(actions) if actions else ""
    
    def get_opponent_strategic_actions(self, schema: PokerFeatureSchema) -> str:
        """
        Get strategic actions performed by opponent on the current street.
        Shows: check-raise, donk bet, 3-bet, float bet, probe bet [all monotonic]
        """
        opp_seq = schema.opponent_current_sequence
        actions = []
        
        # Check each strategic action (they're all boolean/monotonic)
        if opp_seq.did_check_raise == 1.0:
            actions.append("Check-raise")
        if opp_seq.did_donk_bet == 1.0:
            actions.append("Donk bet")
        if opp_seq.did_3bet == 1.0:
            actions.append("3-bet")
        if opp_seq.did_float_bet == 1.0:
            actions.append("Float bet")
        if opp_seq.did_probe_bet == 1.0:
            actions.append("Probe bet")
        
        return ' | '.join(actions) if actions else ""
    
    def should_log_detailed_features(self, schema: PokerFeatureSchema, action: int, amount: Optional[int]) -> bool:
        """
        Determine if this situation warrants detailed feature logging.
        Returns False to disable all detailed logging.
        """
        return False

    def format_exhaustive_dump(self, schema: PokerFeatureSchema, player_name: str = "P0", context: dict = None) -> str:
        """
        Formats a complete dump of EVERY feature in the schema, highlighting non-zero values.
        Designed for deep debugging to verify all feature calculations.
        Shows all 614+ features with their actual values - both zero and non-zero.
        """
        from dataclasses import fields
        lines = [f"\n{'='*25} EXHAUSTIVE FEATURE DUMP for {player_name} {'='*25}"]
        
        # Add context information if provided
        if context:
            lines.append(f"Episode: {context.get('episode', 'N/A')} | Hand: {context.get('hand', 'N/A')} | Street: {context.get('street', 'N/A')} (Stage {context.get('stage', 'N/A')})")
            lines.append(f"Pot: {context.get('pot', 'N/A')} | Stacks: {context.get('stacks', 'N/A')}")
            lines.append(f"Scenario: {context.get('scenario', 'N/A')}")
            lines.append(f"{'='*75}")
        
        total_feature_count = len(schema.to_vector())
        lines.append(f"Total features in schema: {total_feature_count}")
        
        feature_count = 0
        
        for group_field in fields(schema):
            group_name = group_field.name
            group_obj = getattr(schema, group_name)
            
            if not hasattr(group_obj, '__dataclass_fields__'): 
                continue

            group_lines = []
            group_feature_count = 0
            has_active = False

            for feature_field in fields(group_obj):
                feature_name = feature_field.name
                value = getattr(group_obj, feature_name)
                
                if hasattr(value, '__dataclass_fields__'): # Nested dataclass
                    nested_lines = []
                    nested_count = len(fields(value))
                    nested_has_active = False
                    for nested_field in fields(value):
                        nested_val = getattr(value, nested_field.name)
                        marker = " ★" if nested_val != 0.0 else ""
                        if marker: nested_has_active = True
                        nested_lines.append(f"      {nested_field.name}: {nested_val:.6f}{marker}")
                    
                    group_lines.append(f"    {feature_name}: ({nested_count} features)")
                    group_lines.extend(nested_lines)
                    group_feature_count += nested_count
                    if nested_has_active: has_active = True
                else: # Simple feature
                    group_feature_count += 1
                    marker = " ★" if value != 0.0 else ""
                    if marker: has_active = True
                    group_lines.append(f"    {feature_name}: {value:.6f}{marker}")

            active_indicator = " (HAS ACTIVE)" if has_active else " (all zeros)"
            lines.append(f"\n--- {group_name} ({group_feature_count} features){active_indicator} ---")
            lines.extend(group_lines)
            feature_count += group_feature_count

        lines.append(f"\n{'='*25} SUMMARY {'='*25}")
        lines.append(f"Total features processed: {feature_count}")
        lines.append(f"Features marked with ★ are non-zero (active)")
        lines.append(f"{'=' * (59 + len(player_name))}\n")
        
        return '\n'.join(lines)

    def is_interesting_scenario(self, schema: PokerFeatureSchema, game_state) -> bool:
        """
        Detects if the current game situation is interesting enough for an exhaustive dump.
        Looks for 3-bets on any street, large flop bets, or any all-in.
        """
        self_seq = schema.self_current_sequence
        opp_seq = schema.opponent_current_sequence

        # --- Trigger 1: Any all-in situation on any street ---
        if self_seq.did_go_all_in == 1.0 or opp_seq.did_go_all_in == 1.0:
            return True

        # --- Trigger 2: 3-bet or 4-bet on ANY street. --- 
        # We removed the `if is_preflop:` condition.
        if self_seq.did_3bet == 1.0 or opp_seq.did_3bet == 1.0:
            return True

        return False
    
    def describe_scenario(self, schema: PokerFeatureSchema, game_state) -> str:
        """
        Create a descriptive string for the interesting scenario.
        """
        stage_names = ['Preflop', 'Flop', 'Turn', 'River']
        stage = game_state.stage
        street = stage_names[min(stage, 3)]
        
        # Check for strategic actions
        self_actions = []
        opp_actions = []
        
        self_seq = schema.self_current_sequence
        opp_seq = schema.opponent_current_sequence
        
        if self_seq.did_3bet == 1.0:
            self_actions.append("3bet")
        if self_seq.did_float_bet == 1.0:
            self_actions.append("float")
        if self_seq.did_check_raise == 1.0:
            self_actions.append("checkraise")
        if self_seq.did_probe_bet == 1.0:
            self_actions.append("probe")
        
        if opp_seq.did_3bet == 1.0:
            opp_actions.append("3bet")
        if opp_seq.did_float_bet == 1.0:
            opp_actions.append("float")
        if opp_seq.did_check_raise == 1.0:
            opp_actions.append("checkraise")
        if opp_seq.did_probe_bet == 1.0:
            opp_actions.append("probe")
        
        # Build description
        parts = [street]
        
        if self_actions:
            parts.append(f"Self-{'+'.join(self_actions)}")
        if opp_actions:
            parts.append(f"Opp-{'+'.join(opp_actions)}")
        
        # Add commitment info if high
        commitment = schema.self_current_stack.total_commitment_pct
        if commitment > 0.5:
            parts.append(f"HighCommit-{commitment:.1%}")
        
        # Add SPR info if low
        spr = schema.current_additional.effective_spr
        if spr < 2.0:
            parts.append(f"LowSPR-{spr:.1f}")
        
        # Add equity if close
        equity = schema.current_strategic.hand_vs_range if hasattr(schema.current_strategic, 'hand_vs_range') else 0.0
        if 0.4 <= equity <= 0.6:
            parts.append(f"CloseEquity-{equity:.1%}")
        
        return "_".join(parts)


# Convenience function for quick integration
def format_features_for_hand_log(schema: PokerFeatureSchema, player_name: str = "P0", 
                                action: int = None, amount: Optional[int] = None, 
                                detailed: bool = False) -> str:
    """
    Quick function to format features for hand history logging.
    
    Args:
        schema: Complete feature schema from FeatureExtractor
        player_name: Player identifier (e.g., "P0", "P1")
        action: Action taken (0=fold, 1=call, 2=bet/raise) - used to determine detail level
        amount: Bet amount (if applicable)
        detailed: Force detailed output regardless of heuristics
    
    Returns:
        Formatted string ready for hand_log.append()
    """
    debugger = LiveFeatureDebugger()
    
    # Determine detail level
    if detailed or (action is not None and debugger.should_log_detailed_features(schema, action, amount)):
        result = debugger.format_detailed_features(schema, player_name)
        
        # Add delta analysis if available
        delta_info = debugger.format_delta_analysis(schema)
        if delta_info:
            result += f"\n{delta_info}"
        
        # Add opponent read
        result += f"\n{debugger.format_opponent_read(schema)}"
        
        # Add opponent model stats (the key addition!)
        result += f"\n{debugger.format_opponent_model_stats(schema)}"
        
        return result
    else:
        # Concise single-line format
        return debugger.format_concise_features(schema, player_name)


def dump_all_features_to_log(schema: PokerFeatureSchema, player_name: str = "P0", context: dict = None) -> str:
    """
    Convenience function to generate the exhaustive feature dump for logging.
    Use this to verify that ALL 614 features are working correctly.
    """
    debugger = LiveFeatureDebugger()
    return debugger.format_exhaustive_dump(schema, player_name, context)
    
