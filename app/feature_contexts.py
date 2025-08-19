# app/feature_contexts.py
# Explicit context objects for clean feature extraction architecture

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from poker_core import GameState


@dataclass
class StaticContext:
    """
    Container for stateless information derivable purely from current GameState.
    Used by feature extraction methods that don't need historical data.
    """
    game_state: GameState
    seat_id: int
    
    @property
    def hole_cards(self) -> List[int]:
        """Player's hole cards."""
        return self.game_state.hole_cards[self.seat_id]
    
    @property
    def community(self) -> List[int]:
        """Community cards."""
        return self.game_state.community
    
    @property
    def stack(self) -> int:
        """Player's current stack."""
        return self.game_state.stacks[self.seat_id]
    
    @property
    def current_bet(self) -> int:
        """Player's current bet."""
        return self.game_state.current_bets[self.seat_id]
    
    @property
    def pot(self) -> int:
        """Current pot size."""
        return self.game_state.pot
    
    @property
    def to_call(self) -> int:
        """Amount needed to call."""
        max_bet = max(self.game_state.current_bets)
        return max_bet - self.current_bet
    
    @property
    def stage(self) -> int:
        """Current game stage (0=preflop, 1=flop, 2=turn, 3=river)."""
        return self.game_state.stage
    
    @property
    def num_players(self) -> int:
        """Number of players in game."""
        return self.game_state.num_players
    
    @property
    def legal_actions(self) -> List[int]:
        """Legal actions for current player."""
        return self.game_state.get_legal_actions()


@dataclass
class DynamicContext:
    """
    Container for stateful, historical information that requires tracking.
    Used by feature extraction methods that need betting history, patterns, etc.
    """
    history_tracker: Any  # HistoryTracker instance
    opponent_model: Optional[Any] = None  # Future: OpponentModel
    range_manager: Optional[Any] = None   # Future: RangeManager

@dataclass
class FeatureCalculationRequest:
    """
    Request object that bundles everything needed for feature calculation.
    Makes it clear what dependencies each feature extraction method needs.
    """
    static_ctx: StaticContext
    dynamic_ctx: Optional[DynamicContext] = None  # Only for stateful methods
    
    @classmethod
    def stateless(cls, game_state: GameState, seat_id: int) -> 'FeatureCalculationRequest':
        """Create request for stateless feature calculation."""
        return cls(static_ctx=StaticContext(game_state, seat_id))
    
    @classmethod
    def stateful(cls, game_state: GameState, seat_id: int, history_tracker: Any) -> 'FeatureCalculationRequest':
        """Create request for stateful feature calculation."""
        static_ctx = StaticContext(game_state, seat_id)
        dynamic_ctx = DynamicContext(history_tracker=history_tracker)
        return cls(static_ctx=static_ctx, dynamic_ctx=dynamic_ctx)
    
