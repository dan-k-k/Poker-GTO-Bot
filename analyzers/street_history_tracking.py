# analyzers/street_history_tracking.py
# for history analyzer
# Flexible generic historical state tracking for any feature
# Decoupled from specific feature logic - stores any key-value data

from typing import Dict, List, Any, Optional
import copy


class HistoryTracker:
    """
    Generic flexible tracker for any historical game state information.
    
    Core Philosophy: Store any feature data as key-value pairs.
    No rigid schemas - analyzers can store whatever they need.
    
    Key Methods:
    - save_snapshot(hand_key, street, data_dict): Store any feature dict
    - get_snapshot(hand_key, street): Retrieve stored feature dict
    
    This decouples history storage from feature logic, creating a scalable
    backend that allows new features to be tracked over time with no
    modifications needed to the HistoryTracker itself.
    """
    
    def __init__(self):
        # === CORE GENERIC STORAGE ===
        # Flexible storage: Dict[hand_key, Dict[street, Dict[feature_name, value]]]
        self.snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # === BACKWARD COMPATIBILITY ===
        # Keep some existing functionality for gradual migration
        self.player_total_invested: List[int] = []
        self.num_players: int = 2  # Default heads-up
        
        # === HAND TRACKING ===
        self.current_hand_number: int = 0
        self.hands_tracked: int = 0
        
        # === ENVIRONMENT TRACKING ===
        self.starting_stack: int = 200
        self.big_blind: int = 2
        self.small_blind: int = 1
    
    def new_hand(self, hand_number: int, starting_stack: int = 200, big_blind: int = 2, small_blind: int = 1, num_players: int = 2):
        """Initialize tracking for a new hand."""
        self.current_hand_number = hand_number
        self.starting_stack = starting_stack
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.num_players = num_players
        self.hands_tracked += 1
        
        # Reset player investment tracking for new hand
        self.player_total_invested = [0] * num_players
        
        # Cleanup old data to prevent memory bloat
        self._cleanup_old_data()
    
    def save_snapshot(self, hand_key: str, street: str, data_dict: Dict[str, Any]) -> None:
        """
        Save a snapshot of any feature data for a specific hand and street.
        
        Args:
            hand_key: Unique identifier for the hand (e.g., "hand_123")
            street: Street name ("preflop", "flop", "turn", "river")
            data_dict: Dictionary of feature_name -> value pairs
            
        Example:
            tracker.save_snapshot("hand_42", "flop", {
                "pot_size": 100.0,
                "betting_intensity": 0.75,
                "hand_strength": 0.85,
                "board_coordination": 0.6
            })
        """
        if hand_key not in self.snapshots:
            self.snapshots[hand_key] = {}
        
        # Store a copy to prevent external modifications
        self.snapshots[hand_key][street] = data_dict.copy()
        
        # Cleanup to prevent memory bloat
        self._cleanup_old_snapshots()
    
    def _cleanup_old_snapshots(self):
        """Clean up old snapshot data to prevent memory bloat."""
        # Keep only the most recent 50 hands
        if len(self.snapshots) > 50:
            # Sort by hand number if possible, otherwise by key
            try:
                sorted_hands = sorted(self.snapshots.keys(), key=lambda x: int(x.split('_')[-1]) if '_' in x else int(x))
            except (ValueError, IndexError):
                sorted_hands = sorted(self.snapshots.keys())
            
            # Remove oldest hands
            hands_to_remove = sorted_hands[:-25]  # Keep most recent 25
            for hand_key in hands_to_remove:
                del self.snapshots[hand_key]
    
    def get_snapshot(self, hand_key: str, street: str) -> Dict[str, Any]:
        """
        Retrieve a snapshot of feature data for a specific hand and street.
        
        Args:
            hand_key: Unique identifier for the hand
            street: Street name
            
        Returns:
            Dictionary of feature_name -> value pairs, or empty dict if not found
            
        Example:
            data = tracker.get_snapshot("hand_42", "flop")
            pot_size = data.get("pot_size", 0.0)
        """
        return self.snapshots.get(hand_key, {}).get(street, {})
    
    def get_feature_value(self, hand_key: str, street: str, feature_name: str, default: float = 0.0) -> float:
        """
        Get a specific feature value from a snapshot.
        
        Args:
            hand_key: Unique identifier for the hand
            street: Street name
            feature_name: Name of the feature to retrieve
            default: Default value if feature not found
            
        Returns:
            The feature value or default if not found
        """
        snapshot = self.get_snapshot(hand_key, street)
        return snapshot.get(feature_name, default)
    
    def calculate_delta(self, hand_key: str, current_street: str, previous_street: str, feature_name: str) -> float:
        """
        Calculate the change in a feature value between two streets.
        
        Args:
            hand_key: Unique identifier for the hand
            current_street: Current street name
            previous_street: Previous street name  
            feature_name: Name of the feature to compare
            
        Returns:
            The delta (current - previous), or 0.0 if data not available
        """
        current_value = self.get_feature_value(hand_key, current_street, feature_name, 0.0)
        previous_value = self.get_feature_value(hand_key, previous_street, feature_name, 0.0)
        return current_value - previous_value
    
    def get_all_streets_for_hand(self, hand_key: str) -> List[str]:
        """
        Get all streets that have snapshots for a given hand.
        
        Args:
            hand_key: Unique identifier for the hand
            
        Returns:
            List of street names that have data
        """
        return list(self.snapshots.get(hand_key, {}).keys())
    
    def get_latest_snapshot(self, hand_key: str) -> Dict[str, Any]:
        """
        Get the most recent snapshot for a hand (latest street).
        
        Args:
            hand_key: Unique identifier for the hand
            
        Returns:
            Most recent snapshot data, or empty dict if none found
        """
        hand_data = self.snapshots.get(hand_key, {})
        if not hand_data:
            return {}
        
        # Streets in chronological order
        street_order = ["preflop", "flop", "turn", "river"]
        for street in reversed(street_order):
            if street in hand_data:
                return hand_data[street]
        
        # Fallback: return last street alphabetically
        latest_street = max(hand_data.keys())
        return hand_data[latest_street]
    
    def initialize_hand_with_blinds(self, game_state, hand_number: int = None):
        """
        Initialize a new hand and track initial blind posts.
        
        Args:
            game_state: GameState after blinds are posted
            hand_number: Optional hand number override
        """
        if hand_number is None:
            hand_number = self.current_hand_number + 1
        
        # Initialize hand tracking
        self.new_hand(
            hand_number=hand_number,
            starting_stack=game_state.starting_stack,
            big_blind=game_state.big_blind,
            small_blind=game_state.small_blind,
            num_players=game_state.num_players
        )
        
        # Track blind investments
        self.record_blind_posts(
            game_state.sb_pos, 
            game_state.bb_pos,
            game_state.small_blind,
            game_state.big_blind
        )
    
    def reset_for_new_hand(self):
        """Reset for new hand - alias for new_hand with incremented hand number."""
        self.new_hand(self.current_hand_number + 1, num_players=self.num_players)
    
    def set_starting_stacks(self, starting_stacks: List[int]):
        """Set starting stacks for all players."""
        if starting_stacks and len(starting_stacks) > 0:
            self.starting_stack = starting_stacks[0]  # Use first player's stack as reference
    
    # === MISSING METHODS REQUIRED BY ANALYZERS ===
    
    def get_hand_number(self) -> int:
        """Get the current hand number."""
        return self.current_hand_number
    
    def get_big_blind(self) -> int:
        """Get the big blind amount."""
        return self.big_blind
    
    def get_player_investment_all(self) -> List[int]:
        """Get total investment for all players in current hand."""
        return self.player_total_invested.copy()
    
    def record_blind_posts(self, sb_pos: int, bb_pos: int, small_blind: int, big_blind: int):
        """Record blind posts for hand initialization."""
        if sb_pos < len(self.player_total_invested):
            self.player_total_invested[sb_pos] += small_blind
        if bb_pos < len(self.player_total_invested):
            self.player_total_invested[bb_pos] += big_blind
    
    def update_investment(self, player_id: int, amount: int):
        """Update a player's total investment for the current hand."""
        if player_id < len(self.player_total_invested):
            self.player_total_invested[player_id] += amount
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat (placeholder for now)."""
        # For now, rely on _cleanup_old_snapshots() in save_snapshot()
        # Could add additional cleanup logic here if needed
        pass


# Global instance for backward compatibility
history_tracker = HistoryTracker()


# === USAGE EXAMPLES ===
"""
Example Usage of the New Generic HistoryTracker:

# Save any feature data
tracker.save_snapshot("hand_42", "flop", {
    "pot_size": 100.0,
    "betting_intensity": 0.75,
    "hand_strength": 0.85,
    "board_coordination": 0.6,
    "stack_to_pot_ratio": 2.5
})

# Retrieve specific features
pot_size = tracker.get_feature_value("hand_42", "flop", "pot_size")
hand_strength = tracker.get_feature_value("hand_42", "flop", "hand_strength")

# Calculate deltas between streets
strength_delta = tracker.calculate_delta("hand_42", "turn", "flop", "hand_strength")

# Get full snapshot
flop_data = tracker.get_snapshot("hand_42", "flop")

# This system is completely flexible - any analyzer can store any features:
# - PotDynamicsAnalyzer can store all its financial metrics
# - ActionAnalyzer can store betting patterns and aggression levels  
# - BoardAnalyzer can store texture and threat analysis
# - HandAnalyzer can store strength evolution

# Delta calculations become trivial:
spr_delta = tracker.calculate_delta("hand_42", "river", "turn", "stack_to_pot_ratio")
"""

