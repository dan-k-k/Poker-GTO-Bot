# analyzers/board_analyzer.py
# Dedicated analyzer for BoardFeatures schema group
# Single responsibility: Everything about community card texture and threats

from typing import List, Dict
import numpy as np
from dataclasses import fields
from poker_feature_schema import BoardFeatures, TextureFeatureSet


class BoardAnalyzer:
    """
    Dedicated analyzer for BoardFeatures schema group.
    Handles all calculations related to board texture, threats, and community cards.
    
    Schema Responsibility: BoardFeatures (120 features)
    - One-hot card encoding (52 features)  
    - Flush threat map (16 features)
    - Rank count vector (13 features)
    - Straight threat map (30 features)
    - Board composition (4 features)
    - Rank analysis (5 features)
    """
    
    def __init__(self):
        # Define straight patterns once as class attribute
        self.STRAIGHT_PATTERNS = [
            (12, 0, 1, 2, 3),    # A-5 (wheel)
            (0, 1, 2, 3, 4),     # 2-6
            (1, 2, 3, 4, 5),     # 3-7
            (2, 3, 4, 5, 6),     # 4-8
            (3, 4, 5, 6, 7),     # 5-9
            (4, 5, 6, 7, 8),     # 6-T
            (5, 6, 7, 8, 9),     # 7-J
            (6, 7, 8, 9, 10),    # 8-Q
            (7, 8, 9, 10, 11),   # 9-K
            (8, 9, 10, 11, 12)   # T-A (broadway)
        ]
    
    def extract_features(self, community_cards: List[int]) -> BoardFeatures:
        """
        Extract all BoardFeatures for the schema.
        This is the single entry point for board-related features.
        
        Args:
            community_cards: Community cards as integers (0-5 cards)
            
        Returns:
            Complete BoardFeatures dataclass
        """
        if len(community_cards) < 3:
            return BoardFeatures()  # All zeros for preflop
        
        # Calculate all components efficiently without duplication
        board_texture = self.analyze_texture(community_cards)
        one_hot_features = self._calculate_one_hot_encoding(community_cards)
        composition_features = self._analyze_composition(community_cards)
        rank_features = self._analyze_ranks(community_cards)
        coordination = self.calculate_board_coordination(community_cards)
        
        # Create BoardFeatures efficiently
        return BoardFeatures(
            # One-hot card encoding (52 features)
            **{f'card_{i}': one_hot_features[i] for i in range(52)},
            # Board texture (18 features)
            board_texture=board_texture,
            # Board composition (4 features)
            has_one_pair=composition_features[0],
            has_two_pairs=composition_features[1],
            has_trips=composition_features[2],
            has_quads=composition_features[3],
            # Rank analysis (5 features)
            integral_of_highness=rank_features[0],
            highest_rank=rank_features[1],
            lowest_rank=rank_features[2],
            rank_spread=rank_features[3],
            board_coordination=coordination
        )
    
    def analyze_texture(self, cards: List[int]) -> TextureFeatureSet:
        """
        Extract flush and straight texture features from any set of cards.
        Can be used for board analysis OR hand analysis (hole cards + community).
        
        Args:
            cards: List of card integers (can be 2-7 cards)
            
        Returns:
            TextureFeatureSet with flush and straight analysis
        """
        if len(cards) < 2:
            return TextureFeatureSet()  # All zeros if not enough cards
        
        ranks = [c // 4 for c in cards]
        suits = [c % 4 for c in cards]
        
        # === FLUSH THREAT MAP ANALYSIS ===
        
        flush_features = []
        
        # Analyze each suit (0=♣, 1=♦, 2=♥, 3=♠)
        for suit in range(4):
            # Get cards of this suit
            suit_cards = [i for i, s in enumerate(suits) if s == suit]
            suit_ranks = [ranks[i] for i, s in enumerate(suits) if s == suit]
            
            # Feature 1: Cards present in this suit
            cards_present = float(len(suit_cards))
            
            # Feature 2: High card rank (normalized)
            high_card_rank = 0.0
            if suit_ranks:
                high_card_rank = max(suit_ranks) / 12.0
            
            # Features 3 & 4: Best straight-flush potential
            best_sf_cards = 0
            best_sf_gaps = 0
            
            if len(suit_ranks) >= 2:  # Need at least 2 cards for straight-flush analysis
                for pattern in self.STRAIGHT_PATTERNS:
                    # Count cards in this suit that match the straight pattern
                    pattern_cards = [r for r in suit_ranks if r in pattern]
                    
                    if len(pattern_cards) >= 2:  # At least 2 cards for potential
                        # Calculate gaps for this straight-flush pattern
                        if pattern == (12, 0, 1, 2, 3):  # Wheel case
                            # Handle wheel specially for gap calculation
                            wheel_ranks = []
                            for r in pattern_cards:
                                if r == 12:  # Ace
                                    wheel_ranks.append(-1)  # Treat as low ace
                                else:
                                    wheel_ranks.append(r)
                            wheel_ranks.sort()
                            if len(wheel_ranks) >= 2:
                                rank_spread = max(wheel_ranks) - min(wheel_ranks)
                                gaps = rank_spread - (len(pattern_cards) - 1)
                            else:
                                gaps = 0
                        else:
                            sorted_pattern_cards = sorted(pattern_cards)
                            rank_spread = max(sorted_pattern_cards) - min(sorted_pattern_cards)
                            gaps = rank_spread - (len(pattern_cards) - 1)
                        
                        # Track the best straight-flush potential for this suit
                        if len(pattern_cards) > best_sf_cards:
                            best_sf_cards = len(pattern_cards)
                            best_sf_gaps = gaps
                        elif len(pattern_cards) == best_sf_cards and gaps < best_sf_gaps:
                            best_sf_gaps = gaps
            
            # Add features for this suit (normalize count features)
            flush_features.extend([
                cards_present / 5.0,  # Normalize by max cards (0-5)
                high_card_rank,       # Already normalized
                float(best_sf_cards) / 5.0,  # Normalize by max cards
                float(best_sf_gaps) / 3.0  # Gaps
            ])
        
        # === RANK COUNT VECTOR ===
        rank_counts = [0.0] * 13  # Initialize 13 ranks (2 through A)
        for rank in ranks:
            rank_counts[rank] = float(ranks.count(rank))
        
        # === STRAIGHT THREAT MAP ===
        straight_features = []
        
        for pattern in self.STRAIGHT_PATTERNS:
            # Count cards present in this straight pattern
            present_cards = [r for r in ranks if r in pattern]
            cards_present = len(set(present_cards))  # Use set to avoid counting duplicates
            
            # Calculate internal gaps
            if cards_present >= 2:
                sorted_present = sorted(set(present_cards))
                if pattern == (12, 0, 1, 2, 3):  # Special handling for wheel
                    # For wheel, treat A as low (rank -1) for gap calculation
                    wheel_present = []
                    for r in sorted_present:
                        if r == 12:  # Ace
                            wheel_present.append(-1)  # Treat as low ace
                        else:
                            wheel_present.append(r)
                    wheel_present.sort()
                    rank_spread = max(wheel_present) - min(wheel_present)
                else:
                    rank_spread = max(sorted_present) - min(sorted_present)
                
                internal_gaps = rank_spread - (cards_present - 1)
            else:
                internal_gaps = 0
            
            # Determine if this is an open-ended draw
            is_open_ended = 0.0
            if cards_present >= 3:
                # Check if the present cards form the ends of the straight
                sorted_present = sorted(set(present_cards))
                if pattern == (12, 0, 1, 2, 3):  # Wheel case
                    # Wheel is only open-ended on the high side (can't go lower than A-2-3-4-5)
                    is_open_ended = 0.0
                elif pattern == (8, 9, 10, 11, 12):  # Broadway case
                    # Broadway is only open-ended on the low side
                    is_open_ended = 0.0
                else:
                    # For middle straights, check if cards are at both ends
                    pattern_list = list(pattern)
                    has_low_end = any(r in sorted_present for r in pattern_list[:2])
                    has_high_end = any(r in sorted_present for r in pattern_list[-2:])
                    is_open_ended = 1.0 if has_low_end and has_high_end else 0.0
            
            # Add the three features for this straight (normalize count features)
            straight_features.extend([
                float(cards_present) / 5.0,  # Normalize by max cards (0-5)
                float(internal_gaps) / 3.0,        # Gaps 
                is_open_ended                # Already binary (0.0 or 1.0)
            ])
        
        return TextureFeatureSet(
            # Flush Threat Map (16 features)
            spades_cards_present=flush_features[12],  # Suit 3 (♠)
            spades_high_card_rank=flush_features[13],
            spades_straight_flush_cards=flush_features[14],
            spades_straight_flush_gaps=flush_features[15],
            hearts_cards_present=flush_features[8],   # Suit 2 (♥)
            hearts_high_card_rank=flush_features[9],
            hearts_straight_flush_cards=flush_features[10],
            hearts_straight_flush_gaps=flush_features[11],
            clubs_cards_present=flush_features[0],    # Suit 0 (♣)
            clubs_high_card_rank=flush_features[1],
            clubs_straight_flush_cards=flush_features[2],
            clubs_straight_flush_gaps=flush_features[3],
            diamonds_cards_present=flush_features[4], # Suit 1 (♦)
            diamonds_high_card_rank=flush_features[5],
            diamonds_straight_flush_cards=flush_features[6],
            diamonds_straight_flush_gaps=flush_features[7],
            # Rank count vector (normalized by max possible count)
            rank_2_count=rank_counts[0] / 4.0,
            rank_3_count=rank_counts[1] / 4.0,
            rank_4_count=rank_counts[2] / 4.0,
            rank_5_count=rank_counts[3] / 4.0,
            rank_6_count=rank_counts[4] / 4.0,
            rank_7_count=rank_counts[5] / 4.0,
            rank_8_count=rank_counts[6] / 4.0,
            rank_9_count=rank_counts[7] / 4.0,
            rank_T_count=rank_counts[8] / 4.0,
            rank_J_count=rank_counts[9] / 4.0,
            rank_Q_count=rank_counts[10] / 4.0,
            rank_K_count=rank_counts[11] / 4.0,
            rank_A_count=rank_counts[12] / 4.0,
            # Straight threat map (30 features)
            A5_cards_present=straight_features[0],
            A5_internal_gaps=straight_features[1],
            A5_is_open_ended=straight_features[2],
            S26_cards_present=straight_features[3],
            S26_internal_gaps=straight_features[4],
            S26_is_open_ended=straight_features[5],
            S37_cards_present=straight_features[6],
            S37_internal_gaps=straight_features[7],
            S37_is_open_ended=straight_features[8],
            S48_cards_present=straight_features[9],
            S48_internal_gaps=straight_features[10],
            S48_is_open_ended=straight_features[11],
            S59_cards_present=straight_features[12],
            S59_internal_gaps=straight_features[13],
            S59_is_open_ended=straight_features[14],
            S6T_cards_present=straight_features[15],
            S6T_internal_gaps=straight_features[16],
            S6T_is_open_ended=straight_features[17],
            S7J_cards_present=straight_features[18],
            S7J_internal_gaps=straight_features[19],
            S7J_is_open_ended=straight_features[20],
            S8Q_cards_present=straight_features[21],
            S8Q_internal_gaps=straight_features[22],
            S8Q_is_open_ended=straight_features[23],
            S9K_cards_present=straight_features[24],
            S9K_internal_gaps=straight_features[25],
            S9K_is_open_ended=straight_features[26],
            TA_cards_present=straight_features[27],
            TA_internal_gaps=straight_features[28],
            TA_is_open_ended=straight_features[29]
        )
    
    def _calculate_one_hot_encoding(self, community_cards: List[int]) -> List[float]:
        """Calculate one-hot encoding of community cards (52 features)."""
        board_one_hot = [0.0] * 52
        for card in community_cards:
            board_one_hot[card] = 1.0
        return board_one_hot
    
    def _analyze_composition(self, community_cards: List[int]) -> List[float]:
        """Analyze board composition: pairs, trips, quads (4 features)."""
        ranks = [c // 4 for c in community_cards]
        rank_counts = [ranks.count(r) for r in set(ranks)]
        
        # Count different types of multiples
        num_pairs = sum(1 for count in rank_counts if count == 2)
        num_trips = sum(1 for count in rank_counts if count == 3) 
        num_quads = sum(1 for count in rank_counts if count == 4)
        
        # Monotonic board composition features
        has_one_pair = 1.0 if num_pairs >= 1 else 0.0      # Board has at least 1 pair
        has_two_pairs = 1.0 if num_pairs >= 2 else 0.0     # Board has 2 pairs 
        has_trips = 1.0 if num_trips >= 1 else 0.0         # Board has trips
        has_quads = 1.0 if num_quads >= 1 else 0.0         # Board has quads
        
        return [has_one_pair, has_two_pairs, has_trips, has_quads]
    
    def _analyze_ranks(self, community_cards: List[int]) -> List[float]:
        """Analyze rank distribution: integral of highness, high, low, spread (4 features)."""
        ranks = [c // 4 for c in community_cards]
        
        # Integral of highness - sum of all ranks normalized by max possible
        total_rank_value = sum(ranks)
        max_possible = len(community_cards) * 12  # All aces would be max
        integral_of_highness = total_rank_value / max_possible if max_possible > 0 else 0.0
        
        # Board high card rank (normalized)
        highest_rank = max(ranks) / 12.0 if ranks else 0.0  # 0.0=2, 1.0=A
        
        # Board rank distribution analysis
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) > 1:
            lowest_rank = min(ranks) / 12.0  # Lowest rank on board
            rank_spread = (max(ranks) - min(ranks)) / 12.0  # Spread (0=compact, 1=2 vs A)
        else:
            lowest_rank = highest_rank  # Single rank
            rank_spread = 0.0  # No spread with single rank
        
        return [integral_of_highness, highest_rank, lowest_rank, rank_spread]
            
    def calculate_board_coordination(self, community: List[int]) -> float:
        """
        Calculates board coordination based on the consistency of gaps between ranks.
        A lower standard deviation of gaps indicates better coordination.
        """
        if len(community) < 3:
            return 0.0

        # Using a set gets unique ranks, which is better for this calculation
        ranks = sorted(list(set(c // 4 for c in community)))
        
        if len(ranks) < 2:
            return 1.0 # Perfectly coordinated if only one rank

        gaps = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]

        if not gaps:
            return 0.0

        # Standard deviation of the gaps is a measure of how uneven the spacing is.
        std_dev_of_gaps = np.std(gaps)

        # We normalize this score. A std_dev of 0 is perfect coordination (score=1.0).
        # The scaling factor (e.g., 4.0) is empirical to map the score between 0 and 1.
        coordination = max(0.0, 1.0 - (std_dev_of_gaps / 4.0))
        
        return coordination

