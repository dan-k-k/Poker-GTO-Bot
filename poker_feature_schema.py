# poker_feature_schema.py
# Master feature schema organized by poker concepts
# Makes features instantly findable and self-documenting

from dataclasses import dataclass, fields, field
from typing import List, Tuple, Optional

# =============================================================================
# BOARD & HAND FEATURES  
# =============================================================================

@dataclass
class TextureFeatureSet:
    """Flush and straight potential features that can be used for both board and hand analysis [59 features]."""
    # Flush Threat Map (16 features: 4 suits × 4 features each)
    # Spades flush analysis
    spades_cards_present: float = 0.0
    spades_high_card_rank: float = 0.0
    spades_straight_flush_cards: float = 0.0
    spades_straight_flush_gaps: float = 0.0
    # Hearts flush analysis
    hearts_cards_present: float = 0.0
    hearts_high_card_rank: float = 0.0
    hearts_straight_flush_cards: float = 0.0
    hearts_straight_flush_gaps: float = 0.0
    # Clubs flush analysis
    clubs_cards_present: float = 0.0
    clubs_high_card_rank: float = 0.0
    clubs_straight_flush_cards: float = 0.0
    clubs_straight_flush_gaps: float = 0.0
    # Diamonds flush analysis
    diamonds_cards_present: float = 0.0
    diamonds_high_card_rank: float = 0.0
    diamonds_straight_flush_cards: float = 0.0
    diamonds_straight_flush_gaps: float = 0.0
    
    # Rank Count Vector (13 features)
    rank_2_count: float = 0.0
    rank_3_count: float = 0.0
    rank_4_count: float = 0.0
    rank_5_count: float = 0.0
    rank_6_count: float = 0.0
    rank_7_count: float = 0.0
    rank_8_count: float = 0.0
    rank_9_count: float = 0.0
    rank_T_count: float = 0.0
    rank_J_count: float = 0.0
    rank_Q_count: float = 0.0
    rank_K_count: float = 0.0
    rank_A_count: float = 0.0
    
    # Straight Threat Map (30 features: 10 straights × 3 features each)
    # A-5 straight (wheel)
    A5_cards_present: float = 0.0
    A5_internal_gaps: float = 0.0
    A5_is_open_ended: float = 0.0
    # 2-6 straight
    S26_cards_present: float = 0.0
    S26_internal_gaps: float = 0.0
    S26_is_open_ended: float = 0.0
    # 3-7 straight
    S37_cards_present: float = 0.0
    S37_internal_gaps: float = 0.0
    S37_is_open_ended: float = 0.0
    # 4-8 straight
    S48_cards_present: float = 0.0
    S48_internal_gaps: float = 0.0
    S48_is_open_ended: float = 0.0
    # 5-9 straight
    S59_cards_present: float = 0.0
    S59_internal_gaps: float = 0.0
    S59_is_open_ended: float = 0.0
    # 6-T straight
    S6T_cards_present: float = 0.0
    S6T_internal_gaps: float = 0.0
    S6T_is_open_ended: float = 0.0
    # 7-J straight
    S7J_cards_present: float = 0.0
    S7J_internal_gaps: float = 0.0
    S7J_is_open_ended: float = 0.0
    # 8-Q straight
    S8Q_cards_present: float = 0.0
    S8Q_internal_gaps: float = 0.0
    S8Q_is_open_ended: float = 0.0
    # 9-K straight
    S9K_cards_present: float = 0.0
    S9K_internal_gaps: float = 0.0
    S9K_is_open_ended: float = 0.0
    # T-A straight (broadway)
    TA_cards_present: float = 0.0
    TA_internal_gaps: float = 0.0
    TA_is_open_ended: float = 0.0
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class MyHandFeatures:
    """Everything about my hole cards and their strength (185 features = 67 strength + 59 hole texture + 59 hand texture)."""
    # Hole card one-hot encoding (52 features)
    hole_0: float = 0.0; hole_1: float = 0.0; hole_2: float = 0.0; hole_3: float = 0.0; hole_4: float = 0.0
    hole_5: float = 0.0; hole_6: float = 0.0; hole_7: float = 0.0; hole_8: float = 0.0; hole_9: float = 0.0
    hole_10: float = 0.0; hole_11: float = 0.0; hole_12: float = 0.0; hole_13: float = 0.0; hole_14: float = 0.0
    hole_15: float = 0.0; hole_16: float = 0.0; hole_17: float = 0.0; hole_18: float = 0.0; hole_19: float = 0.0
    hole_20: float = 0.0; hole_21: float = 0.0; hole_22: float = 0.0; hole_23: float = 0.0; hole_24: float = 0.0
    hole_25: float = 0.0; hole_26: float = 0.0; hole_27: float = 0.0; hole_28: float = 0.0; hole_29: float = 0.0
    hole_30: float = 0.0; hole_31: float = 0.0; hole_32: float = 0.0; hole_33: float = 0.0; hole_34: float = 0.0
    hole_35: float = 0.0; hole_36: float = 0.0; hole_37: float = 0.0; hole_38: float = 0.0; hole_39: float = 0.0
    hole_40: float = 0.0; hole_41: float = 0.0; hole_42: float = 0.0; hole_43: float = 0.0; hole_44: float = 0.0
    hole_45: float = 0.0; hole_46: float = 0.0; hole_47: float = 0.0; hole_48: float = 0.0; hole_49: float = 0.0
    hole_50: float = 0.0; hole_51: float = 0.0
    
    # Hole card rank analysis (4 features - intrinsic potential only)
    hole_integral_of_highness: float = 0.0
    hole_highest_rank: float = 0.0  
    hole_lowest_rank: float = 0.0
    hole_rank_spread: float = 0.0
    
    # Hole card texture (59 features) - comprehensive analysis of hole cards only
    hole_card_texture: TextureFeatureSet = field(default_factory=TextureFeatureSet)
    
    # Raw strength calculation - MOVED TO ADDITIONAL FEATURES
    
    # Monotonic strength categories (9 features)
    at_least_pair: float = 0.0
    at_least_two_pair: float = 0.0
    at_least_three_kind: float = 0.0
    at_least_straight: float = 0.0
    at_least_flush: float = 0.0
    at_least_full_house: float = 0.0
    at_least_four_kind: float = 0.0
    straight_flush: float = 0.0
    royal_flush: float = 0.0
    
    # Kicker information (5 features)
    kicker_1: float = 0.0
    kicker_2: float = 0.0
    kicker_3: float = 0.0
    kicker_4: float = 0.0
    kicker_5: float = 0.0
    
    # MADE hand texture (24 features) - flush and straight potential of my hole cards + board
    hand_texture: TextureFeatureSet = field(default_factory=TextureFeatureSet)
    
    def to_list(self) -> List[float]:
        result = []
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, 'to_list'):  # Handle nested dataclass
                result.extend(value.to_list())
            else:
                result.append(value)
        return result


@dataclass
class BoardFeatures:
    """Everything about the community cards and board texture (120 features)."""
    # One-hot card encoding (52 features)
    card_0: float = 0.0; card_1: float = 0.0; card_2: float = 0.0; card_3: float = 0.0; card_4: float = 0.0
    card_5: float = 0.0; card_6: float = 0.0; card_7: float = 0.0; card_8: float = 0.0; card_9: float = 0.0
    card_10: float = 0.0; card_11: float = 0.0; card_12: float = 0.0; card_13: float = 0.0; card_14: float = 0.0
    card_15: float = 0.0; card_16: float = 0.0; card_17: float = 0.0; card_18: float = 0.0; card_19: float = 0.0
    card_20: float = 0.0; card_21: float = 0.0; card_22: float = 0.0; card_23: float = 0.0; card_24: float = 0.0
    card_25: float = 0.0; card_26: float = 0.0; card_27: float = 0.0; card_28: float = 0.0; card_29: float = 0.0
    card_30: float = 0.0; card_31: float = 0.0; card_32: float = 0.0; card_33: float = 0.0; card_34: float = 0.0
    card_35: float = 0.0; card_36: float = 0.0; card_37: float = 0.0; card_38: float = 0.0; card_39: float = 0.0
    card_40: float = 0.0; card_41: float = 0.0; card_42: float = 0.0; card_43: float = 0.0; card_44: float = 0.0
    card_45: float = 0.0; card_46: float = 0.0; card_47: float = 0.0; card_48: float = 0.0; card_49: float = 0.0
    card_50: float = 0.0; card_51: float = 0.0
    
    # Board texture (24 features) - flush and straight potential of just the board
    board_texture: TextureFeatureSet = field(default_factory=TextureFeatureSet)
    
    # Board composition (4 features)
    has_one_pair: float = 0.0
    has_two_pairs: float = 0.0
    has_trips: float = 0.0
    has_quads: float = 0.0
    
    # Rank analysis (5 features)
    integral_of_highness: float = 0.0
    highest_rank: float = 0.0
    lowest_rank: float = 0.0
    rank_spread: float = 0.0
    board_coordination: float = 0.0
    
    def to_list(self) -> List[float]:
        result = []
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, 'to_list'):  # Handle nested dataclass
                result.extend(value.to_list())
            else:
                result.append(value)
        return result


# =============================================================================
# CURRENT STREET FEATURES (No history tracking, opponent reproducible)  
# =============================================================================

@dataclass
class CurrentStreetSequenceFeatures:
    """Current street sequence features for any seat_id [OPPONENT REPRODUCIBLE]"""
    # From CurrentStreetAnalyzer.calculate_current_street_sequence()
    checked_count: float = 0.0                      # Seat id checked x times
    raised_count: float = 0.0                       # Seat id raised (includes initial bet) x times
    raise_pct_of_pot: float = 0.0                   # Seat id raised by x% of pot before raise (strategic sizing)
    bet_to_starting_pot: float = 0.0                # Seat id total bet as x% of starting street pot (commitment level)
    overbet_count: float = 0.0                      # Seat id overbet (>100% of pot raise/bet) x times
    largebet_count: float = 0.0                     # Seat id largebet (>70% of pot raise/bet) x times
    is_facing_check: float = 0.0                    # Seat id is currently facing a check
    is_facing_bet: float = 0.0                      # Seat id is currently facing a bet
    is_facing_raise: float = 0.0                    # Seat id is currently facing a raise (2bet)
    is_facing_3bet: float = 0.0                     # Seat id is currently facing a 3bet
    is_facing_4betplus: float = 0.0                 # Seat id is currently facing a 4bet+ [all monotonic]
    # Strategic features
    did_check_raise: float = 0.0                    # Seat id check-raised on this street
    did_donk_bet: float = 0.0                       # Seat id made donk bet (bet OOP, not prev street aggressor)
    did_3bet: float = 0.0                           # Seat id made 3-bet this street
    did_float_bet: float = 0.0                      # Seat id made float bet (called prev street IP, bet when checked to)
    did_probe_bet: float = 0.0                      # Seat id made probe bet (bet OOP after PF aggressor checked back)
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class CurrentStreetStackFeatures:
    """Current street stack features for any seat_id [OPPONENT REPRODUCIBLE]"""
    # From CurrentStreetAnalyzer.calculate_current_street_stack()
    stack_in_bb: float = 0.0                        # Seat id stack in BB
    pot_size_ratio: float = 0.0                     # Seat id pot size ratio (pot size / total money)
    call_cost_ratio: float = 0.0                    # Seat id call cost ratio (to call / stack)
    pot_odds: float = 0.0                           # Seat id pot odds (to call / (pot + to call))
    stack_size_ratio: float = 0.0                   # Seat id stack size ratio (stack / total money)
    current_street_commitment_bb: float = 0.0       # Seat id amount committed this street in BB
    current_street_commitment_vs_starting_pot: float = 0.0  # Seat id amount committed this street / starting pot this street
    current_street_commitment_vs_starting_stack: float = 0.0  # Seat id amount committed this street / seat id starting stack this street
    total_commitment: float = 0.0                   # Seat id total commitment (across all streets: % of stack committed)
    total_commitment_bb: float = 0.0                # Seat id total commitment in BB
    stack_smaller_than_pot: float = 0.0             # Seat id stack is smaller than pot
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class CurrentPositionFeatures:
    """Current position features for any seat_id [OPPONENT REPRODUCIBLE]"""
    # From CurrentStreetAnalyzer.calculate_current_position()
    is_early_position: float = 0.0                  # Seat id is early position
    is_dealer: float = 0.0                          # Seat id is dealer
    is_sb: float = 0.0                              # Seat id is SB (heads up, this = dealer)
    is_bb: float = 0.0                              # Seat id is BB
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class CurrentStageFeatures:
    """Current stage features (not seat-specific)"""
    # From CurrentStreetAnalyzer.calculate_current_stage()
    is_preflopplus: float = 0.0                     # is_preflopplus
    is_flopplus: float = 0.0                        # is_flopplus
    is_turnplus: float = 0.0                        # is_turnplus
    is_river: float = 0.0                           # is_river
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class CurrentAdditionalFeatures:
    """Current additional features (Hero only)"""
    # From CurrentStreetAnalyzer.calculate_current_street_additional()
    effective_spr: float = 0.0                      # Effective stack to pot ratio (min(stacks) / pot)
    implied_odds: float = 0.0                       # Implied odds (now considers opponent range)
    
    # Hand strength features (moved here for history tracking)
    hand_strength: float = 0.0                      # MC sims vs random hands
    equity_vs_range: float = 0.0                    # Equity vs opponent's estimated range
    
    # Delta features (change from previous street)
    equity_delta: float = 0.0                       # Change in equity_vs_range from previous street
    spr_delta: float = 0.0                          # Change in effective SPR from previous street
    pot_size_delta: float = 0.0                     # Change in pot size from previous street
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]

# =============================================================================
# HISTORY FEATURES (History tracked, opponent reproducible for some)
# =============================================================================

@dataclass
class SequenceHistoryFeatures:
    """Sequence history features for any seat_id [HISTORY TRACKED AND OPPONENT REPRODUCIBLE]"""
    # From HistoryAnalyzer.calculate_sequence_history() - per street data
    # Preflop features
    preflop_checked_count: float = 0.0               # Seat id checked x times
    preflop_raised_count: float = 0.0                # Seat id raised (includes initial bet) x times
    preflop_avg_raise_pct: float = 0.0               # Seat id raised by x% of pot on average
    preflop_overbet_once: float = 0.0                # Seat id overbet (>100% of pot raise) at least once
    preflop_largebet_count: float = 0.0              # Seat id largebet (>70% of pot raise/bet) x times
    preflop_was_first_bettor: float = 0.0            # Seat id was first raiser/bettor
    preflop_was_last_bettor: float = 0.0             # Seat id was last raiser/bettor
    # Strategic features
    preflop_did_check_raise: float = 0.0             # Seat id check-raised preflop
    preflop_did_3bet: float = 0.0                    # Seat id made 3-bet preflop
    preflop_did_donk_bet: float = 0.0                # Seat id made donk bet preflop
    preflop_did_float_bet: float = 0.0               # Seat id made float bet preflop
    preflop_did_probe_bet: float = 0.0               # Seat id made probe bet preflop
    
    # Flop features
    flop_checked_count: float = 0.0
    flop_raised_count: float = 0.0
    flop_avg_raise_pct: float = 0.0
    flop_overbet_once: float = 0.0
    flop_largebet_count: float = 0.0
    flop_was_first_bettor: float = 0.0
    flop_was_last_bettor: float = 0.0
    # Strategic features
    flop_did_check_raise: float = 0.0
    flop_did_3bet: float = 0.0
    flop_did_donk_bet: float = 0.0
    flop_did_float_bet: float = 0.0
    flop_did_probe_bet: float = 0.0
    
    # Turn features
    turn_checked_count: float = 0.0
    turn_raised_count: float = 0.0
    turn_avg_raise_pct: float = 0.0
    turn_overbet_once: float = 0.0
    turn_largebet_count: float = 0.0
    turn_was_first_bettor: float = 0.0
    turn_was_last_bettor: float = 0.0
    # Strategic features
    turn_did_check_raise: float = 0.0
    turn_did_3bet: float = 0.0
    turn_did_donk_bet: float = 0.0
    turn_did_float_bet: float = 0.0
    turn_did_probe_bet: float = 0.0
    
    # River features
    river_checked_count: float = 0.0
    river_raised_count: float = 0.0
    river_avg_raise_pct: float = 0.0
    river_overbet_once: float = 0.0
    river_largebet_count: float = 0.0
    river_was_first_bettor: float = 0.0
    river_was_last_bettor: float = 0.0
    # Strategic features
    river_did_check_raise: float = 0.0
    river_did_3bet: float = 0.0
    river_did_donk_bet: float = 0.0
    river_did_float_bet: float = 0.0
    river_did_probe_bet: float = 0.0
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class StackHistoryFeatures:
    """Stack history features for any seat_id [HISTORY TRACKED AND OPPONENT REPRODUCIBLE]"""
    # From HistoryAnalyzer.calculate_stack_history() - per street data
    # Preflop features
    preflop_stack_in_bb: float = 0.0                 # Seat id stack in BB
    preflop_pot_size_ratio: float = 0.0              # Seat id pot size ratio (pot size / total money)
    preflop_call_cost_ratio: float = 0.0             # Seat id call cost ratio (to call / stack)
    preflop_pot_odds: float = 0.0                    # Seat id pot odds (to call / (pot + to call))
    preflop_stack_size_ratio: float = 0.0            # Seat id stack size ratio (stack / total money)
    preflop_current_street_commitment_bb: float = 0.0  # Seat id amount committed this street in BB
    preflop_total_commitment: float = 0.0            # Seat id total commitment (across all streets so far: % of stack committed)
    preflop_total_commitment_bb: float = 0.0         # Seat id total commitment (across all streets so far: in BB)
    
    # Flop features
    flop_stack_in_bb: float = 0.0
    flop_pot_size_ratio: float = 0.0
    flop_call_cost_ratio: float = 0.0
    flop_pot_odds: float = 0.0
    flop_stack_size_ratio: float = 0.0
    flop_current_street_commitment_bb: float = 0.0
    flop_total_commitment: float = 0.0
    flop_total_commitment_bb: float = 0.0
    
    # Turn features
    turn_stack_in_bb: float = 0.0
    turn_pot_size_ratio: float = 0.0
    turn_call_cost_ratio: float = 0.0
    turn_pot_odds: float = 0.0
    turn_stack_size_ratio: float = 0.0
    turn_current_street_commitment_bb: float = 0.0
    turn_total_commitment: float = 0.0
    turn_total_commitment_bb: float = 0.0
    
    # River features
    river_stack_in_bb: float = 0.0
    river_pot_size_ratio: float = 0.0
    river_call_cost_ratio: float = 0.0
    river_pot_odds: float = 0.0
    river_stack_size_ratio: float = 0.0
    river_current_street_commitment_bb: float = 0.0
    river_total_commitment: float = 0.0
    river_total_commitment_bb: float = 0.0
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class AdditionalHistoryFeatures:
    """Additional history features (Hero only) [HISTORY TRACKED]"""
    # From HistoryAnalyzer.calculate_additional_history() - per street data
    preflop_effective_spr: float = 0.0               # Effective stack to pot ratio (min(stacks) / pot)
    preflop_implied_odds: float = 0.0                # Implied odds
    preflop_hand_strength: float = 0.0               # MC sims vs random hands
    preflop_equity_vs_range: float = 0.0             # Equity vs opponent's estimated range
    
    flop_effective_spr: float = 0.0
    flop_implied_odds: float = 0.0
    flop_hand_strength: float = 0.0
    flop_equity_vs_range: float = 0.0
    
    turn_effective_spr: float = 0.0
    turn_implied_odds: float = 0.0
    turn_hand_strength: float = 0.0
    turn_equity_vs_range: float = 0.0
    
    river_effective_spr: float = 0.0
    river_implied_odds: float = 0.0
    river_hand_strength: float = 0.0
    river_equity_vs_range: float = 0.0
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]

# =============================================================================
# OPPONENT MODEL FEATURES (Per-street strategic statistics)
# =============================================================================

@dataclass
class OpponentModelFeatures:
    """Opponent modeling features based on per-street strategic statistics [56 features]."""
    # Meta statistics
    total_hands: float = 0.0                           # Total hands observed
    sample_size: float = 0.0                           # Recent sample size (sliding window)
    
    # Pre-flop core statistics
    vpip: float = 0.0                                   # Voluntarily Put In Pot %
    pfr: float = 0.0                                    # Pre-Flop Raise %
    three_bet: float = 0.0                              # Overall 3-bet % (legacy)
    fold_to_three_bet: float = 0.0                      # Fold to 3-bet %
    limp: float = 0.0                                   # Limp %
    preflop_fold_rate: float = 0.0                      # Preflop fold rate
    
    # Per-street strategic actions - 3-bet
    three_bet_preflop: float = 0.0                      # 3-bet preflop %
    three_bet_flop: float = 0.0                         # 3-bet flop % (much rarer)
    three_bet_turn: float = 0.0                         # 3-bet turn % (very rare)
    three_bet_river: float = 0.0                        # 3-bet river % (extremely rare)
    
    # Per-street strategic actions - Donk bet
    donk_bet_flop: float = 0.0                          # Donk bet flop % (OOP, not prev aggressor)
    donk_bet_turn: float = 0.0                          # Donk bet turn %
    donk_bet_river: float = 0.0                         # Donk bet river %
    
    # Per-street strategic actions - Probe bet
    probe_bet_turn: float = 0.0                         # Probe bet turn % (OOP after PF aggressor checked)
    probe_bet_river: float = 0.0                        # Probe bet river %
    
    # Per-street strategic actions - Check-raise
    checkraise_flop: float = 0.0                        # Check-raise flop %
    checkraise_turn: float = 0.0                        # Check-raise turn %
    checkraise_river: float = 0.0                       # Check-raise river %
    
    # Float bet (call flop IP, bet when checked to)
    float_bet: float = 0.0                              # Float bet %
    
    # Post-flop aggression by street
    cbet_flop: float = 0.0                              # Continuation bet flop %
    cbet_turn: float = 0.0                              # Continuation bet turn %
    cbet_river: float = 0.0                             # Continuation bet river %
    aggression_frequency: float = 0.0                   # Overall post-flop aggression %
    
    # Post-flop defense by street
    fold_to_cbet_flop: float = 0.0                      # Fold to c-bet flop %
    fold_to_cbet_turn: float = 0.0                      # Fold to c-bet turn %
    fold_to_cbet_river: float = 0.0                     # Fold to c-bet river %
    
    # Street-specific fold rates
    flop_fold_rate: float = 0.0                         # Fold rate on flop
    turn_fold_rate: float = 0.0                         # Fold rate on turn
    river_fold_rate: float = 0.0                        # Fold rate on river
    fold_frequency: float = 0.0                         # Overall fold frequency
    
    # Showdown tendencies
    wtsd: float = 0.0                                   # Went To Showdown %
    showdown_win_rate: float = 0.0                      # Won $ at Showdown %
    
    # Bet sizing and all-in tendencies
    all_in_frequency: float = 0.0                       # All-in frequency %
    avg_bet_size: float = 0.0                           # Average bet size (BB)
    avg_pot_ratio: float = 0.0                          # Average bet-to-pot ratio
    
    # === ADVANCED BETTING PATTERNS ===
    
    # Multi-Street Aggression Patterns
    double_barrel: float = 0.0                          # % bet turn after c-betting flop
    triple_barrel: float = 0.0                          # % bet river after betting flop+turn
    delayed_cbet: float = 0.0                           # % bet turn after checking flop (as PF aggressor)
    
    # Positional Betting Tendencies  
    steal_attempt: float = 0.0                          # % raise from late position when folded to
    fold_to_steal: float = 0.0                          # % fold in blinds vs late position raise
    button_isolation: float = 0.0                       # % isolate limpers from button
    
    # Advanced Defensive and Deceptive Lines
    fold_vs_flop_checkraise: float = 0.0                # % fold when c-bet gets check-raised
    limp_reraise: float = 0.0                           # % 3-bet after limping (trapping)
    slowplay_frequency: float = 0.0                     # % check strong hands for deception
    
    # Bet Sizing Tells (Layer 2 exploitative gold)
    river_overbet_frequency: float = 0.0                # % overbet (>100% pot) on river
    value_bet_sizing: float = 0.0                       # Avg bet size with value hands (% pot)
    bluff_bet_sizing: float = 0.0                       # Avg bet size with bluff hands (% pot)
    sizing_tell_strength: float = 0.0                   # |value_size - bluff_size| (exploitability)
    
    # Multi-Street Continuation Patterns
    turn_probe_after_check: float = 0.0                 # % bet turn after checking flop OOP
    river_probe_after_check: float = 0.0                # % bet river after checking turn OOP
    barrel_give_up_turn: float = 0.0                    # % give up (check) turn after c-betting flop
    barrel_give_up_river: float = 0.0                   # % give up river after betting flop+turn
    
    def to_list(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]


# =============================================================================
# MASTER SCHEMA
# =============================================================================


@dataclass
class PokerFeatureSchema:
    """
    Master feature container organized by poker concepts, not calculation methods.
    Makes every feature instantly findable and self-documenting.
    
    Core Features: 188 + 120 = 308 features (MyHandFeatures with full texture analysis)
    Current Street (Hero + Opponent): (11 + 11 + 4) * 2 + 4 + 7 = 63 features (added hand_strength, equity_vs_range, 3 deltas)
    History (Hero + Opponent): (28 + 32) * 2 + 16 = 136 features (added 8 hand strength/equity history features)
    Opponent Model: 56 features (per-street strategic statistics + advanced patterns)
    Total: 308 + 63 + 136 + 56 = 563 features
    """
    # === CORE POKER CONCEPTS ===
    my_hand: MyHandFeatures = field(default_factory=MyHandFeatures)                    # 188 features
    board: BoardFeatures = field(default_factory=BoardFeatures)                       # 120 features
    
    # === CURRENT STREET FEATURES (No history tracking) ===
    # Hero features
    hero_current_sequence: CurrentStreetSequenceFeatures = field(default_factory=CurrentStreetSequenceFeatures)  # 10 features
    hero_current_stack: CurrentStreetStackFeatures = field(default_factory=CurrentStreetStackFeatures)            # 11 features
    hero_current_position: CurrentPositionFeatures = field(default_factory=CurrentPositionFeatures)              # 4 features
    
    # Opponent features (reproducible for opponent analysis)
    opponent_current_sequence: CurrentStreetSequenceFeatures = field(default_factory=CurrentStreetSequenceFeatures)  # 10 features
    opponent_current_stack: CurrentStreetStackFeatures = field(default_factory=CurrentStreetStackFeatures)            # 11 features
    opponent_current_position: CurrentPositionFeatures = field(default_factory=CurrentPositionFeatures)              # 4 features
    
    # Non-seat-specific features
    current_stage: CurrentStageFeatures = field(default_factory=CurrentStageFeatures)                       # 4 features
    current_additional: CurrentAdditionalFeatures = field(default_factory=CurrentAdditionalFeatures)        # 7 features (hero only)
    
    # === HISTORY FEATURES (History tracked) ===
    # Hero history
    hero_sequence_history: SequenceHistoryFeatures = field(default_factory=SequenceHistoryFeatures)             # 28 features (7*4)
    hero_stack_history: StackHistoryFeatures = field(default_factory=StackHistoryFeatures)                      # 32 features (8*4)
    
    # Opponent history (reproducible for opponent analysis)
    opponent_sequence_history: SequenceHistoryFeatures = field(default_factory=SequenceHistoryFeatures)         # 28 features (7*4)
    opponent_stack_history: StackHistoryFeatures = field(default_factory=StackHistoryFeatures)                  # 32 features (8*4)
    
    # Non-seat-specific history
    additional_history: AdditionalHistoryFeatures = field(default_factory=AdditionalHistoryFeatures)       # 16 features (hero only)
    
    # === OPPONENT MODEL FEATURES ===
    # Opponent statistical profile based on observed behavior
    opponent_model: OpponentModelFeatures = field(default_factory=OpponentModelFeatures)                   # 56 features
    
    def to_vector(self) -> List[float]:
        """Flatten the entire nested structure into a single feature vector for ML model."""
        vector = []
        
        # Core poker concepts
        vector.extend(self.my_hand.to_list())                     # 185 features
        vector.extend(self.board.to_list())                       # 120 features
        
        # Current street features (hero + opponent)
        vector.extend(self.hero_current_sequence.to_list())       # 10 features
        vector.extend(self.hero_current_stack.to_list())          # 11 features
        vector.extend(self.hero_current_position.to_list())       # 4 features
        vector.extend(self.opponent_current_sequence.to_list())   # 10 features
        vector.extend(self.opponent_current_stack.to_list())      # 11 features
        vector.extend(self.opponent_current_position.to_list())   # 4 features
        vector.extend(self.current_stage.to_list())               # 4 features
        vector.extend(self.current_additional.to_list())          # 2 features
        
        # History features (hero + opponent)
        vector.extend(self.hero_sequence_history.to_list())       # 28 features
        vector.extend(self.hero_stack_history.to_list())          # 32 features
        vector.extend(self.opponent_sequence_history.to_list())   # 28 features
        vector.extend(self.opponent_stack_history.to_list())      # 32 features
        vector.extend(self.additional_history.to_list())          # 8 features
        vector.extend(self.opponent_model.to_list())              # 56 features
        
        return vector
    
    def find_feature(self, name: str) -> Optional[Tuple[str, int, float]]:
        """
        Find a feature by name and return (concept_group, local_index, value).
        
        Example: schema.find_feature('stack_to_pot_ratio') 
        Returns: ('pot_dynamics', 2, 1.5)
        """
        for group_field in fields(self):
            group_name = group_field.name
            group_obj = getattr(self, group_name)
            
            for feature_field in fields(group_obj):
                if feature_field.name == name:
                    feature_names = [f.name for f in fields(group_obj)]
                    local_index = feature_names.index(name)
                    value = getattr(group_obj, name)
                    return (group_name, local_index, value)
        
        return None
    
    def get_concept_summary(self) -> str:
        """Return a readable summary of all concept groups and their sizes."""
        summary = "Poker Feature Schema Summary:\n"
        total_features = 0
        
        for group_field in fields(self):
            group_name = group_field.name
            group_obj = getattr(self, group_name)
            if hasattr(group_obj, 'to_list'):
                feature_count = len(group_obj.to_list())
            else:
                feature_count = len(fields(group_obj))
            total_features += feature_count
            
            summary += f"  {group_name:25} {feature_count:3d} features\n"
        
        summary += f"  {'TOTAL':25} {total_features:3d} features\n"
        return summary
    
    def validate_feature_count(self, expected: int = 563) -> bool:
        """Validate that we have the expected number of features."""
        actual = len(self.to_vector())
        if actual != expected:
            print(f"WARNING: Expected {expected} features, got {actual}")
            print(f"Feature breakdown:")
            print(self.get_concept_summary())
            return False
        return True

