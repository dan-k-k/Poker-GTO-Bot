# poker_core.py
# Core poker classes and utilities - foundation for the poker AI system

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
from itertools import combinations
import random


class Suit(Enum):
    """Card suits in the order used by the existing integer representation."""
    CLUBS = 0      # ♣
    DIAMONDS = 1   # ♦ 
    HEARTS = 2     # ♥
    SPADES = 3     # ♠
    
    def __str__(self):
        symbols = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}
        return symbols[self.value]


class Rank(Enum):
    """Card ranks in the order used by the existing integer representation."""
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12
    
    def __str__(self):
        symbols = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
        return symbols[self.value]


@dataclass(frozen=True)
class Card:
    """
    Immutable card representation that maintains compatibility with existing integer-based system.
    Cards can be converted to/from integers using the same 0-51 mapping as before.
    """
    rank: Rank
    suit: Suit
    
    def to_int(self) -> int:
        """Convert card to integer representation (0-51) for compatibility."""
        return self.rank.value * 4 + self.suit.value
    
    @classmethod
    def from_int(cls, card_id: int) -> 'Card':
        """Create card from integer representation (0-51) for compatibility."""
        if not 0 <= card_id <= 51:
            raise ValueError(f"Card ID must be 0-51, got {card_id}")
        rank = Rank(card_id // 4)
        suit = Suit(card_id % 4)
        return cls(rank, suit)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return f"Card({self.rank.name}, {self.suit.name})"


class Deck:
    """
    Standard 52-card deck with shuffling and dealing capabilities.
    Maintains compatibility with existing integer-based system.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        """Reset deck to full 52 cards and shuffle."""
        # Create all 52 cards as integers (for compatibility)
        self.cards = list(range(52))
        self.rng.shuffle(self.cards)
    
    def deal(self, num_cards: int = 1) -> List[int]:
        """Deal specified number of cards as integers."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards, only {len(self.cards)} remaining")
        
        dealt = []
        for _ in range(num_cards):
            dealt.append(self.cards.pop())
        return dealt
    
    def deal_card(self) -> int:
        """Deal a single card as integer."""
        return self.deal(1)[0]
    
    def cards_remaining(self) -> int:
        """Number of cards remaining in deck."""
        return len(self.cards)
    
    def peek_next(self, num_cards: int = 1) -> List[int]:
        """Peek at next cards without dealing them."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot peek {num_cards} cards, only {len(self.cards)} remaining")
        return self.cards[-num_cards:]


@dataclass
class GameState:
    """
    Centralized container for all game state variables.
    Separates game data from game logic for cleaner architecture.
    """
    
    # Basic game configuration
    num_players: int
    starting_stack: int
    small_blind: int
    big_blind: int
    
    # Cards and community
    hole_cards: List[List[int]]  # Each player's hole cards as integers
    community: List[int]         # Community cards as integers
    
    # Money and betting
    stacks: List[int]            # Each player's current stack
    current_bets: List[int]      # Current round bets
    pot: int                     # Current pot size
    starting_pot_this_round: int # Pot size at start of current betting round
    starting_stacks_this_hand: List[int] # Each player's stack at start of hand (before blinds)
    
    # Player states
    active: List[bool]           # Is player still in hand
    all_in: List[bool]           # Is player all-in
    acted: List[bool]            # Has player acted this round
    surviving_players: List[int] # List of player IDs still in tournament
    
    # Game flow
    stage: int                   # 0=preflop, 1=flop, 2=turn, 3=river
    dealer_pos: int              # Dealer button position
    sb_pos: int                  # Small blind position
    bb_pos: int                  # Big blind position
    to_move: int                 # Current player to act
    
    # Betting context
    initial_bet: Optional[int]   # Initial bet amount for current round
    last_raise_size: int         # Size of last raise
    last_raiser: Optional[int]   # Player who made last raise
    
    # Game status
    terminal: bool               # Is hand over
    winners: Optional[List[int]] # Winners (if terminal)
    win_reason: Optional[str]    # How hand ended
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            hole_cards=[cards.copy() for cards in self.hole_cards],
            community=self.community.copy(),
            stacks=self.stacks.copy(),
            current_bets=self.current_bets.copy(),
            pot=self.pot,
            starting_pot_this_round=self.starting_pot_this_round,
            active=self.active.copy(),
            all_in=self.all_in.copy(),
            acted=self.acted.copy(),
            stage=self.stage,
            dealer_pos=self.dealer_pos,
            sb_pos=self.sb_pos,
            bb_pos=self.bb_pos,
            to_move=self.to_move,
            initial_bet=self.initial_bet,
            last_raise_size=self.last_raise_size,
            last_raiser=self.last_raiser,
            terminal=self.terminal,
            winners=self.winners.copy() if self.winners else None,
            win_reason=self.win_reason
        )
    
    def get_legal_actions(self) -> List[int]:
        """Get legal actions for current player."""
        if self.terminal or not self.active[self.to_move] or self.all_in[self.to_move]:
            return []
        
        current_max = max(self.current_bets)
        player_bet = self.current_bets[self.to_move]
        to_call = current_max - player_bet
        
        legal = []
        
        # Check/Call (always legal)
        legal.append(1)
        
        # Fold (only if facing a bet)
        if to_call > 0:
            legal.append(0)
        
        # Raise (if player has enough chips)
        if self.stacks[self.to_move] > 0:
            legal.append(2)
        
        return sorted(legal)


# Utility functions for card conversion
def cards_to_ints(cards: List[Card]) -> List[int]:
    """Convert list of Card objects to integers."""
    return [card.to_int() for card in cards]


def ints_to_cards(card_ints: List[int]) -> List[Card]:
    """Convert list of integers to Card objects."""
    return [Card.from_int(card_id) for card_id in card_ints]


def card_to_int(card: Card) -> int:
    """Convert single Card to integer."""
    return card.to_int()


def int_to_card(card_id: int) -> Card:
    """Convert single integer to Card."""
    return Card.from_int(card_id)


class HandEvaluator:
    """
    Centralized hand evaluation logic.
    Single source of truth for poker hand rankings.
    Eliminates duplicate code across TexasHoldem.py, poker_utils.py, and agents.py.
    """
    
    @staticmethod
    def hand_rank(card_list: List[int]) -> Tuple:
        """
        Evaluate the rank of a 5-card poker hand.
        
        Args:
            card_list: List of 5 card integers (0-51)
            
        Returns:
            Tuple representing hand strength (higher = better)
            Format: (hand_type, primary_rank, secondary_rank, ...)
            
        Hand types (0-8):
            0: High card, 1: Pair, 2: Two pair, 3: Three of a kind,
            4: Straight, 5: Flush, 6: Full house, 7: Four of a kind, 8: Straight flush
        """
        if len(card_list) != 5:
            raise ValueError(f"hand_rank expects exactly 5 cards, got {len(card_list)}")
        
        # Convert cards to ranks and suits
        ranks = sorted([c // 4 for c in card_list], reverse=True)
        suits = [c % 4 for c in card_list]
        
        # Count rank frequencies
        counts = {r: ranks.count(r) for r in set(ranks)}
        freq = sorted(counts.values(), reverse=True)
        unique_ranks = sorted(counts.keys(), reverse=True)
        
        # Check for flush and straight
        is_flush = len(set(suits)) == 1
        sorted_ranks = sorted(set(ranks), reverse=True)
        is_straight = len(sorted_ranks) == 5 and (sorted_ranks[0] - sorted_ranks[4] == 4)
        
        # Handle A-2-3-4-5 straight (wheel)
        if set(sorted_ranks) == {12, 3, 2, 1, 0}:  # A, 5, 4, 3, 2
            is_straight = True
            sorted_ranks = [3, 2, 1, 0, -1]  # Treat ace as low (rank -1 for comparison)
        
        # Evaluate hand type
        if is_straight and is_flush:
            return (8, sorted_ranks[0])  # Straight flush
        
        if freq == [4, 1]:  # Four of a kind
            four_kind = [r for r, c in counts.items() if c == 4][0]
            kicker = [r for r, c in counts.items() if c == 1][0]
            return (7, four_kind, kicker)
        
        if freq == [3, 2]:  # Full house
            three_kind = [r for r, c in counts.items() if c == 3][0]
            pair = [r for r, c in counts.items() if c == 2][0]
            return (6, three_kind, pair)
        
        if is_flush:
            return (5, *ranks)  # Flush (all 5 cards matter for comparison)
        
        if is_straight:
            return (4, sorted_ranks[0])  # Straight
        
        if freq == [3, 1, 1]:  # Three of a kind
            three_kind = [r for r, c in counts.items() if c == 3][0]
            kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
            return (3, three_kind, *kickers)
        
        if freq == [2, 2, 1]:  # Two pair
            pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in counts.items() if c == 1][0]
            return (2, pairs[0], pairs[1], kicker)
        
        if freq == [2, 1, 1, 1]:  # One pair
            pair = [r for r, c in counts.items() if c == 2][0]
            kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
            return (1, pair, *kickers)
        
        # High card
        return (0, *ranks)
    
    @staticmethod
    def best_hand_rank(cards: List[int]) -> Tuple:
        """
        Find the best 5-card poker hand from 5-7 cards (flop through to river).
        
        Args:
            cards: List of 5-7 card integers (hole + community)
            
        Returns:
            Tuple representing the best possible hand rank
        """
        if len(cards) < 5 or len(cards) > 7:
            raise ValueError(f"best_hand_rank expects 5-7 cards, got {len(cards)}")
        
        # If exactly 5 cards, evaluate directly
        if len(cards) == 5:
            return HandEvaluator.hand_rank(cards)
        
        best_rank = (-1,)  # Worst possible rank
        
        # Try all combinations of 5 cards from the available cards
        for combo in combinations(cards, 5):
            rank = HandEvaluator.hand_rank(list(combo))
            if rank > best_rank:
                best_rank = rank
        
        return best_rank
    
    @staticmethod
    def best_five_cards(seven_cards: List[int]) -> List[int]:
        """
        Find the actual best 5 cards from 7 cards.
        
        Args:
            seven_cards: List of 7 card integers (hole + community)
            
        Returns:
            List of 5 card integers representing the best hand
        """
        if len(seven_cards) != 7:
            raise ValueError(f"best_five_cards expects exactly 7 cards, got {len(seven_cards)}")
        
        best_rank = (-1,)
        best_cards = []
        
        # Try all combinations of 5 cards from the 7
        for combo in combinations(seven_cards, 5):
            card_list = list(combo)
            rank = HandEvaluator.hand_rank(card_list)
            if rank > best_rank:
                best_rank = rank
                best_cards = card_list
        
        return best_cards
    
    @staticmethod
    def compare_hands(hand1: List[int], hand2: List[int]) -> int:
        """
        Compare two 7-card hands.
        
        Args:
            hand1: First 7-card hand
            hand2: Second 7-card hand
            
        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        rank1 = HandEvaluator.best_hand_rank(hand1)
        rank2 = HandEvaluator.best_hand_rank(hand2)
        
        if rank1 > rank2:
            return 1
        elif rank2 > rank1:
            return -1
        else:
            return 0
    
    @staticmethod
    def hand_type_name(rank_tuple: Tuple) -> str:
        """
        Get human-readable name for a hand rank.
        
        Args:
            rank_tuple: Hand rank tuple from hand_rank()
            
        Returns:
            String description of the hand
        """
        hand_type = rank_tuple[0]
        names = {
            0: "High Card",
            1: "Pair", 
            2: "Two Pair",
            3: "Three of a Kind",
            4: "Straight",
            5: "Flush",
            6: "Full House", 
            7: "Four of a Kind",
            8: "Straight Flush"
        }
        return names.get(hand_type, "Unknown")


def get_betting_order(seat_id: int, dealer_pos: int, num_players: int) -> int:
    """
    Calculates a betting order where higher is better (acts later).
    SB=0, BB=1, ..., Button=num_players-1
    
    This is a centralized utility function used across multiple analyzers
    to ensure consistent betting order calculations.
    
    Args:
        seat_id: Player's seat position
        dealer_pos: Current dealer button position  
        num_players: Total number of players
        
    Returns:
        Betting order (0=early, higher=later)
    """
    relative_position = (seat_id - dealer_pos) % num_players
    return (relative_position - 1 + num_players) % num_players


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def card_to_string(card_id: int) -> str:
    """Convert card ID (0-51) to string representation like '2s'."""
    if card_id < 0 or card_id > 51:
        return 'As'  # Fallback
        
    rank_id = card_id // 4
    suit_id = card_id % 4
    
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    return ranks[rank_id] + suits[suit_id]


def string_to_card_id(card_str: str) -> int:
    """Convert card string like '2s' to card ID (0-51)."""
    if len(card_str) < 2:
        return 0
    rank_char = card_str[0]
    suit_char = card_str[1]
    
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
               '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    rank = rank_map.get(rank_char, 0)
    suit = suit_map.get(suit_char, 0)
    
    return rank * 4 + suit


def get_street_name(stage: int) -> str:
    """Convert stage number to street name."""
    stage_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
    return stage_map.get(stage, 'preflop')

