# analyzers/hand_analyzer.py
# Dedicated analyzer for MyHandFeatures schema group
# Single responsibility: Everything about the player's hole cards and hand strength

import random
import sys
import os
from typing import List, Dict
from poker_core import HandEvaluator
from poker_feature_schema import MyHandFeatures
from .board_analyzer import BoardAnalyzer

# Equity calculation moved to CurrentStreetAnalyzer


class HandAnalyzer:
    """
    Dedicated analyzer for MyHandFeatures schema group.
    Handles all calculations related to hole card strength and categories.
    
    Schema Responsibility: MyHandFeatures (185 features)
    - Hole card one-hot encoding (52 features)
    - Hole card rank analysis (4 features) 
    - Hole card texture analysis (59 features)
    - Hand strength via Monte Carlo simulation (1 feature)
    - Monotonic strength categories (9 features)
    - Kicker information (5 features)
    - Combined hand texture analysis (59 features)
    """
    
    def __init__(self):
        self.cache = {}  # Cache for hand strength calculations
        self.board_analyzer = BoardAnalyzer()  # For texture analysis
        
        # Preflop equity lookup table (169 unique starting hands)
        # Values represent equity vs random hand (win rate)
        self.PREFLOP_EQUITY = {
            # Premium pairs
            "AA": 0.852, "KK": 0.824, "QQ": 0.796, "JJ": 0.775, "TT": 0.751,
            "99": 0.723, "88": 0.693, "77": 0.661, "66": 0.628, "55": 0.595,
            "44": 0.563, "33": 0.531, "22": 0.499,
            
            # Suited aces
            "AKs": 0.670, "AQs": 0.662, "AJs": 0.654, "ATs": 0.647, "A9s": 0.623,
            "A8s": 0.615, "A7s": 0.608, "A6s": 0.601, "A5s": 0.600, "A4s": 0.593,
            "A3s": 0.586, "A2s": 0.580,
            
            # Offsuit aces
            "AKo": 0.654, "AQo": 0.640, "AJo": 0.626, "ATo": 0.613, "A9o": 0.583,
            "A8o": 0.569, "A7o": 0.556, "A6o": 0.543, "A5o": 0.536, "A4o": 0.523,
            "A3o": 0.510, "A2o": 0.498,
            
            # Suited kings
            "KQs": 0.628, "KJs": 0.620, "KTs": 0.613, "K9s": 0.590, "K8s": 0.583,
            "K7s": 0.576, "K6s": 0.570, "K5s": 0.563, "K4s": 0.557, "K3s": 0.551,
            "K2s": 0.545,
            
            # Offsuit kings
            "KQo": 0.606, "KJo": 0.592, "KTo": 0.578, "K9o": 0.551, "K8o": 0.537,
            "K7o": 0.524, "K6o": 0.511, "K5o": 0.498, "K4o": 0.486, "K3o": 0.474,
            "K2o": 0.462,
            
            # Suited queens
            "QJs": 0.587, "QTs": 0.580, "Q9s": 0.557, "Q8s": 0.550, "Q7s": 0.544,
            "Q6s": 0.537, "Q5s": 0.531, "Q4s": 0.525, "Q3s": 0.519, "Q2s": 0.513,
            
            # Offsuit queens
            "QJo": 0.559, "QTo": 0.545, "Q9o": 0.518, "Q8o": 0.504, "Q7o": 0.491,
            "Q6o": 0.478, "Q5o": 0.466, "Q4o": 0.454, "Q3o": 0.442, "Q2o": 0.430,
            
            # Suited jacks
            "JTs": 0.548, "J9s": 0.525, "J8s": 0.518, "J7s": 0.512, "J6s": 0.505,
            "J5s": 0.499, "J4s": 0.493, "J3s": 0.487, "J2s": 0.481,
            
            # Offsuit jacks
            "JTo": 0.514, "J9o": 0.487, "J8o": 0.473, "J7o": 0.460, "J6o": 0.447,
            "J5o": 0.435, "J4o": 0.423, "J3o": 0.411, "J2o": 0.399,
            
            # Suited tens
            "T9s": 0.493, "T8s": 0.486, "T7s": 0.480, "T6s": 0.474, "T5s": 0.468,
            "T4s": 0.462, "T3s": 0.456, "T2s": 0.450,
            
            # Offsuit tens  
            "T9o": 0.455, "T8o": 0.441, "T7o": 0.428, "T6o": 0.416, "T5o": 0.404,
            "T4o": 0.392, "T3o": 0.381, "T2o": 0.370,
            
            # Suited nines
            "98s": 0.454, "97s": 0.448, "96s": 0.442, "95s": 0.436, "94s": 0.430,
            "93s": 0.424, "92s": 0.418,
            
            # Offsuit nines
            "98o": 0.410, "97o": 0.397, "96o": 0.385, "95o": 0.373, "94o": 0.362,
            "93o": 0.351, "92o": 0.340,
            
            # Suited eights
            "87s": 0.416, "86s": 0.410, "85s": 0.404, "84s": 0.398, "83s": 0.392,
            "82s": 0.386,
            
            # Offsuit eights
            "87o": 0.366, "86o": 0.354, "85o": 0.342, "84o": 0.331, "83o": 0.320,
            "82o": 0.310,
            
            # Suited sevens
            "76s": 0.378, "75s": 0.372, "74s": 0.366, "73s": 0.360, "72s": 0.354,
            
            # Offsuit sevens
            "76o": 0.323, "75o": 0.312, "74o": 0.301, "73o": 0.291, "72o": 0.281,
            
            # Suited sixes
            "65s": 0.340, "64s": 0.334, "63s": 0.328, "62s": 0.322,
            
            # Offsuit sixes
            "65o": 0.281, "64o": 0.271, "63o": 0.261, "62o": 0.252,
            
            # Suited fives
            "54s": 0.302, "53s": 0.296, "52s": 0.290,
            
            # Offsuit fives
            "54o": 0.240, "53o": 0.231, "52o": 0.222,
            
            # Suited fours
            "43s": 0.264, "42s": 0.258,
            
            # Offsuit fours
            "43o": 0.201, "42o": 0.192,
            
            # Suited threes
            "32s": 0.227,
            
            # Offsuit threes
            "32o": 0.162,
        }
    
    def extract_features(self, hole_cards: List[int], community_cards: List[int]) -> MyHandFeatures:
        """
        Extract all MyHandFeatures for the schema.
        This is the single entry point for hand-related features.
        
        Args:
            hole_cards: Player's two hole cards
            community_cards: Community cards (0-5 cards)
            
        Returns:
            Complete MyHandFeatures dataclass (184 features)
        """
        # Calculate hole card one-hot encoding (component analysis)
        hole_one_hot = self._calculate_hole_one_hot(hole_cards)
        
        # Calculate hole card rank analysis (component analysis)
        hole_rank_features = self.board_analyzer._analyze_ranks(hole_cards)
        
        # Calculate hole card texture (component analysis - hole cards only)
        hole_card_texture = self.board_analyzer.analyze_texture(hole_cards)
        
        # Hand strength calculations moved to Additional Features
        
        # Get structured hand data
        category_data = self._categorize_hand_features(hole_cards, community_cards)
        
        # Calculate hand texture (flush and straight potential of my cards + board)
        all_my_cards = hole_cards + community_cards
        hand_texture = self.board_analyzer.analyze_texture(all_my_cards)
        
        return MyHandFeatures(
            # Hole card one-hot encoding (52 features)
            **{f'hole_{i}': hole_one_hot[i] for i in range(52)},
            # Hole card rank analysis (4 features)
            hole_integral_of_highness=hole_rank_features[0],
            hole_highest_rank=hole_rank_features[1],
            hole_lowest_rank=hole_rank_features[2],
            hole_rank_spread=hole_rank_features[3],
            # Hole card texture (59 features)
            hole_card_texture=hole_card_texture,
            # Hand strength - MOVED TO ADDITIONAL FEATURES
            # Monotonic categories
            at_least_pair=category_data['monotonic_flags'][1],
            at_least_two_pair=category_data['monotonic_flags'][2],
            at_least_three_kind=category_data['monotonic_flags'][3],
            at_least_straight=category_data['monotonic_flags'][4],
            at_least_flush=category_data['monotonic_flags'][5],
            at_least_full_house=category_data['monotonic_flags'][6],
            at_least_four_kind=category_data['monotonic_flags'][7],
            straight_flush=category_data['monotonic_flags'][8],
            royal_flush=category_data['monotonic_flags'][9] if len(category_data['monotonic_flags']) > 9 else 0.0,
            # Kickers
            kicker_1=category_data['kickers'][0],
            kicker_2=category_data['kickers'][1],
            kicker_3=category_data['kickers'][2],
            kicker_4=category_data['kickers'][3],
            kicker_5=category_data['kickers'][4],
            # Hand texture
            hand_texture=hand_texture
        )
    
    def _calculate_hand_strength(self, hole: List[int], community: List[int]) -> float:
        """
        ✅ The single, authoritative method for calculating hand strength.
        Uses preflop lookup table for speed and accuracy, Monte Carlo simulation for post-flop.
        """
        # --- PREFLOP: Use the fast and accurate lookup table ---
        if len(community) == 0:
            return self._get_preflop_strength(hole)
        
        # --- POST-FLOP: Use Monte Carlo simulation with caching ---
        cache_key = tuple(sorted(hole + community))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if len(hole) < 2:
            return 0.5  # Default for invalid hands
        
        # Monte Carlo simulation
        wins = 0
        ties = 0
        trials = 500  # Increased for better accuracy
        
        # Create deck without known cards
        known_cards = set(hole + community)
        deck = [i for i in range(52) if i not in known_cards]
        
        for _ in range(trials):
            # Deal random opponent hand
            if len(deck) < 2:
                break
            opp_cards = random.sample(deck, 2)
            remaining_deck = [c for c in deck if c not in opp_cards]
            
            # Complete the board if needed
            cards_needed = 5 - len(community)
            if cards_needed > 0 and len(remaining_deck) >= cards_needed:
                board_completion = random.sample(remaining_deck, cards_needed)
                final_board = community + board_completion
            else:
                final_board = community + [0] * cards_needed  # Pad if needed
            
            # Evaluate hands
            my_strength = self._evaluate_hand_strength(hole, final_board)
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
        self.cache[cache_key] = strength
        return strength
    
    def _get_preflop_strength(self, hole: List[int]) -> float:
        """Get preflop hand strength from lookup table."""
        hand_key = self._cards_to_key(hole)
        return self.PREFLOP_EQUITY.get(hand_key, 0.3)  # Default for very weak hands
    
    def _cards_to_key(self, hole: List[int]) -> str:
        """Convert hole cards to lookup key like 'AKs' or 'T7o'."""
        if len(hole) != 2:
            return "72o"  # Default to weakest hand
        
        # Convert card numbers to ranks and suits
        card1, card2 = hole
        rank1, suit1 = card1 // 4, card1 % 4
        rank2, suit2 = card2 // 4, card2 % 4
        
        # Convert rank numbers to characters
        rank_chars = "23456789TJQKA"
        
        # Order by rank (higher rank first)
        if rank1 >= rank2:
            high_rank, low_rank = rank_chars[rank1], rank_chars[rank2]
            suited = suit1 == suit2
        else:
            high_rank, low_rank = rank_chars[rank2], rank_chars[rank1]
            suited = suit1 == suit2
        
        # Handle pairs (AA, KK, etc.)
        if high_rank == low_rank:
            return high_rank + high_rank
        
        # Handle suited vs offsuit
        suffix = "s" if suited else "o"
        return high_rank + low_rank + suffix
    
    def _calculate_hole_one_hot(self, hole_cards: List[int]) -> List[float]:
        """Calculate one-hot encoding of hole cards (52 features)."""
        hole_one_hot = [0.0] * 52
        for card in hole_cards:
            hole_one_hot[card] = 1.0
        return hole_one_hot
    
    def _categorize_hand_features(self, hole: List[int], community: List[int]) -> dict:
        """
        Extracts hand features for all streets, including detailed pre-flop properties.
        Returns structured dictionary instead of magic indices.
        """
        # Extract rank information for hand evaluation
        rank1 = hole[0] // 4
        rank2 = hole[1] // 4

        # Calculate dynamic hand strength based on current street
        if len(community) == 0:
            # Preflop: Use hole card strength
            monotonic_flags = [1.0] + [0.0] * 9  # Always have at least high card
            
            if rank1 == rank2:
                # Pocket pair
                monotonic_flags[1] = 1.0  # at_least_pair
                kickers = [(rank1 + 1) / 13.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # High cards only
                high_card = max(rank1, rank2)
                low_card = min(rank1, rank2)
                kickers = [(high_card + 1) / 13.0, (low_card + 1) / 13.0, 0.0, 0.0, 0.0]
        else:
            # Post-flop: Use best 5-card hand evaluation
            all_cards = hole + community
            best_rank_tuple = HandEvaluator.best_hand_rank(all_cards)
            
            if best_rank_tuple[0] == -1:
                # Fallback for incomplete evaluation
                monotonic_flags = [0.0] * 10
                kickers = [0.0] * 5
            else:
                hand_type = best_rank_tuple[0]
                
                # Create monotonic categories (at_least_X)
                # Note: hand_type mapping: 0=high_card, 1=pair, 2=two_pair, 3=trips, 4=straight, 5=flush, 6=full_house, 7=quads, 8=straight_flush, 9=royal_flush
                monotonic_flags = [0.0] * 10
                
                # Always have at least high card
                monotonic_flags[0] = 1.0  # at_least_high_card (always true)
                
                # Set monotonic flags based on hand type
                if hand_type >= 1:  # pair or better
                    monotonic_flags[1] = 1.0  # at_least_pair
                if hand_type >= 2:  # two pair or better
                    monotonic_flags[2] = 1.0  # at_least_two_pair
                if hand_type >= 3:  # trips or better
                    monotonic_flags[3] = 1.0  # at_least_three_kind
                if hand_type >= 4:  # straight or better
                    monotonic_flags[4] = 1.0  # at_least_straight
                if hand_type >= 5:  # flush or better
                    monotonic_flags[5] = 1.0  # at_least_flush
                if hand_type >= 6:  # full house or better
                    monotonic_flags[6] = 1.0  # at_least_full_house
                if hand_type >= 7:  # quads or better
                    monotonic_flags[7] = 1.0  # at_least_four_kind
                if hand_type >= 8:  # straight flush or better
                    monotonic_flags[8] = 1.0  # straight_flush
                if hand_type >= 9:  # royal flush
                    monotonic_flags[9] = 1.0  # royal_flush

                # Extract kicker information
                comparison_ranks = []
                for item in best_rank_tuple[1:]:
                    if isinstance(item, (list, tuple)):
                        comparison_ranks.extend(item)
                    else:
                        comparison_ranks.append(item)
                
                while len(comparison_ranks) < 5:
                    comparison_ranks.append(-1)
                comparison_ranks = comparison_ranks[:5]
                
                kickers = [(r + 1) / 13.0 for r in comparison_ranks]

        return {
            # Dynamic hand strength (changes with community cards)
            'monotonic_flags': monotonic_flags,
            'kickers': kickers
        }

    
    # Equity calculation methods moved to CurrentStreetAnalyzer
    
    def _evaluate_hand_strength(self, hole_cards: List[int], board: List[int]) -> float:
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
    
    def clear_cache(self):
        """Clear the hand strength cache."""
        self.cache.clear()

