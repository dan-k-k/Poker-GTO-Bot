# feature_extractor.py
# Schema-aligned FeatureExtractor with new analyzer architecture
# Uses CurrentStreetAnalyzer and HistoryAnalyzer to eliminate overlap

import numpy as np
from typing import List, Dict, Optional, Tuple

from poker_core import GameState
from feature_contexts import StaticContext, DynamicContext

# Import the master schema
from poker_feature_schema import PokerFeatureSchema

# Import new unified analyzers
from analyzers.history_tracking import HistoryTracker
from analyzers.hand_analyzer import HandAnalyzer
from analyzers.board_analyzer import BoardAnalyzer
from analyzers.current_street_analyzer import CurrentStreetAnalyzer
from analyzers.history_analyzer import HistoryAnalyzer
from analyzers.action_sequencer import ActionSequencer


class FeatureExtractor:
    """
    Schema-aligned FeatureExtractor with new unified analyzer architecture.
    
    Eliminates overlap between old analyzers using:
    - CurrentStreetAnalyzer: Non-history tracked, opponent reproducible features
    - HistoryAnalyzer: History tracked features with proper HistoryTracker integration
    
    Schema Mapping:
    - MyHandFeatures                    → HandAnalyzer
    - BoardFeatures                     → BoardAnalyzer  
    - CurrentStreetSequenceFeatures     → CurrentStreetAnalyzer
    - CurrentStreetStackFeatures        → CurrentStreetAnalyzer
    - CurrentPositionFeatures           → CurrentStreetAnalyzer
    - CurrentStageFeatures              → CurrentStreetAnalyzer
    - CurrentAdditionalFeatures         → CurrentStreetAnalyzer
    - SequenceHistoryFeatures           → HistoryAnalyzer
    - StackHistoryFeatures              → HistoryAnalyzer
    - AdditionalHistoryFeatures         → HistoryAnalyzer
    """
    
    def __init__(self, num_players: int = 2):
        # Core tracking components
        self.history_tracker = HistoryTracker()
        self.action_sequencer = ActionSequencer()
        
        # New unified analyzers
        self.hand_analyzer = HandAnalyzer()
        self.board_analyzer = BoardAnalyzer()
        self.current_street_analyzer = CurrentStreetAnalyzer()
        self.history_analyzer = HistoryAnalyzer()
    
    def extract_features(self, game_state: GameState, seat_id: int, 
                        opponent_stats: Dict = None, opponent_action_history: List[Dict] = None) -> Tuple[np.ndarray, PokerFeatureSchema]:
        """
        Main orchestration method - returns both feature vector and structured schema.
        
        Args:
            game_state: Current game state
            seat_id: Player seat ID
            opponent_stats: Optional dict with opponent's poker statistics
            opponent_action_history: Optional list of opponent's actions this hand
            
        Returns:
            Tuple of (507-element feature vector, structured schema) for ML training and debugging
        """
        # Build explicit contexts
        static_ctx = StaticContext(game_state=game_state, seat_id=seat_id)
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        
        # Create the master schema object to be populated
        schema = PokerFeatureSchema()
        
        # === CORE POKER CONCEPTS ===
        schema.my_hand = self.hand_analyzer.extract_features(
            static_ctx.hole_cards, static_ctx.community
        )
        
        schema.board = self.board_analyzer.extract_features(
            static_ctx.community
        )
        
        # === CURRENT STREET FEATURES (No history tracking) ===
        # Get opponent seat_id (in heads-up, it's the other player)
        opponent_seat_id = 1 - seat_id if static_ctx.num_players == 2 else (seat_id + 1) % static_ctx.num_players
        
        # HERO current street features
        hero_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.hero_current_sequence = self._dict_to_current_sequence_features(hero_current_sequence_data)
        
        hero_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_current_stack = self._dict_to_current_stack_features(hero_current_stack_data)
        
        hero_current_position_data = self.current_street_analyzer.calculate_current_position(
            seat_id, static_ctx
        )
        schema.hero_current_position = self._dict_to_current_position_features(hero_current_position_data)
        
        # OPPONENT current street features
        opponent_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_sequence = self._dict_to_current_sequence_features(opponent_current_sequence_data)
        
        opponent_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_current_stack = self._dict_to_current_stack_features(opponent_current_stack_data)
        
        opponent_current_position_data = self.current_street_analyzer.calculate_current_position(
            opponent_seat_id, static_ctx
        )
        schema.opponent_current_position = self._dict_to_current_position_features(opponent_current_position_data)
        
        # NON-SEAT-SPECIFIC current street features
        current_stage_data = self.current_street_analyzer.calculate_current_stage(static_ctx)
        schema.current_stage = self._dict_to_current_stage_features(current_stage_data)
        
        current_additional_data = self.current_street_analyzer.calculate_current_street_additional(
            static_ctx, dynamic_ctx, opponent_stats, opponent_action_history
        )
        schema.current_additional = self._dict_to_current_additional_features(current_additional_data)
        
        # === HISTORY FEATURES (History tracked) ===
        # HERO history features
        hero_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_sequence_history = self._dict_to_sequence_history_features(hero_sequence_history_data)
        
        hero_stack_history_data = self.history_analyzer.calculate_stack_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_stack_history = self._dict_to_stack_history_features(hero_stack_history_data)
        
        # OPPONENT history features
        opponent_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_sequence_history = self._dict_to_sequence_history_features(opponent_sequence_history_data)
        
        opponent_stack_history_data = self.history_analyzer.calculate_stack_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_stack_history = self._dict_to_stack_history_features(opponent_stack_history_data)
        
        # NON-SEAT-SPECIFIC history features
        additional_history_data = self.history_analyzer.calculate_additional_history(
            static_ctx, dynamic_ctx
        )
        schema.additional_history = self._dict_to_additional_history_features(additional_history_data)
        
        # === FLATTEN TO VECTOR FOR ML MODEL ===
        feature_vector = schema.to_vector()
        
        return np.array(feature_vector), schema
    
    # === PUBLIC INTERFACE FOR HISTORY MANAGEMENT ===
    
    def new_hand(self, starting_stacks: List[int]):
        """Reset for new hand."""
        self.hand_analyzer.clear_cache()
        self.history_tracker.reset_for_new_hand()
        self.history_tracker.set_starting_stacks(starting_stacks)
        self.action_sequencer.new_street()  # Clear action sequence
    
    def new_street(self):
        """Called when a new street begins."""
        self.action_sequencer.new_street()
    
    def record_action(self, seat_id: int, action_type: str, amount: int = 0):
        """Record an action for current street tracking."""
        self.action_sequencer.record_action(seat_id, action_type, amount)
    
    def save_street_snapshot(self, game_state: GameState, street: str = None):
        """Save street snapshot for history tracking using ActionSequencer data."""
        static_ctx = StaticContext(game_state=game_state, seat_id=0)  # seat_id not used for snapshot
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        
        # Get the complete action log from ActionSequencer for accurate historical data
        action_log = self.action_sequencer.get_live_action_sequence()
        
        # Save accurate snapshot using ActionSequencer data
        self.history_analyzer.save_street_snapshot(action_log, static_ctx, dynamic_ctx, street)
    
    # === HELPER METHODS FOR CONVERTING DICT TO FEATURE OBJECTS ===
    
    def _dict_to_current_sequence_features(self, data: dict):
        from poker_feature_schema import CurrentStreetSequenceFeatures
        return CurrentStreetSequenceFeatures(**data)
    
    def _dict_to_current_stack_features(self, data: dict):
        from poker_feature_schema import CurrentStreetStackFeatures
        return CurrentStreetStackFeatures(**data)
    
    def _dict_to_current_position_features(self, data: dict):
        from poker_feature_schema import CurrentPositionFeatures
        return CurrentPositionFeatures(**data)
    
    def _dict_to_current_stage_features(self, data: dict):
        from poker_feature_schema import CurrentStageFeatures
        return CurrentStageFeatures(**data)
    
    def _dict_to_current_additional_features(self, data: dict):
        from poker_feature_schema import CurrentAdditionalFeatures
        return CurrentAdditionalFeatures(**data)
    
    def _dict_to_sequence_history_features(self, data: dict):
        from poker_feature_schema import SequenceHistoryFeatures
        return SequenceHistoryFeatures(**data)
    
    def _dict_to_stack_history_features(self, data: dict):
        from poker_feature_schema import StackHistoryFeatures
        return StackHistoryFeatures(**data)
    
    def _dict_to_additional_history_features(self, data: dict):
        from poker_feature_schema import AdditionalHistoryFeatures
        return AdditionalHistoryFeatures(**data)
    
    # === DEBUGGING AND INTROSPECTION ===
    
    def get_feature_schema(self, game_state: GameState, seat_id: int, 
                          opponent_stats: Dict = None, opponent_action_history: List[Dict] = None) -> PokerFeatureSchema:
        """Get the full structured schema for debugging and introspection."""
        # Build contexts
        static_ctx = StaticContext(game_state=game_state, seat_id=seat_id)
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        
        # Create the master schema object to be populated
        schema = PokerFeatureSchema()
        
        # === CORE POKER CONCEPTS ===
        schema.my_hand = self.hand_analyzer.extract_features(
            static_ctx.hole_cards, static_ctx.community
        )
        
        schema.board = self.board_analyzer.extract_features(
            static_ctx.community
        )
        
        # === CURRENT STREET FEATURES (No history tracking) ===
        # Get opponent seat_id (in heads-up, it's the other player)
        opponent_seat_id = 1 - seat_id if static_ctx.num_players == 2 else (seat_id + 1) % static_ctx.num_players
        
        # HERO current street features
        hero_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.hero_current_sequence = self._dict_to_current_sequence_features(hero_current_sequence_data)
        
        hero_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_current_stack = self._dict_to_current_stack_features(hero_current_stack_data)
        
        hero_current_position_data = self.current_street_analyzer.calculate_current_position(
            seat_id, static_ctx
        )
        schema.hero_current_position = self._dict_to_current_position_features(hero_current_position_data)
        
        # OPPONENT current street features
        opponent_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_sequence = self._dict_to_current_sequence_features(opponent_current_sequence_data)
        
        opponent_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_current_stack = self._dict_to_current_stack_features(opponent_current_stack_data)
        
        opponent_current_position_data = self.current_street_analyzer.calculate_current_position(
            opponent_seat_id, static_ctx
        )
        schema.opponent_current_position = self._dict_to_current_position_features(opponent_current_position_data)
        
        # NON-SEAT-SPECIFIC current street features
        current_stage_data = self.current_street_analyzer.calculate_current_stage(static_ctx)
        schema.current_stage = self._dict_to_current_stage_features(current_stage_data)
        
        current_additional_data = self.current_street_analyzer.calculate_current_street_additional(
            static_ctx, dynamic_ctx, opponent_stats, opponent_action_history
        )
        schema.current_additional = self._dict_to_current_additional_features(current_additional_data)
        
        # === HISTORY FEATURES (History tracked) ===
        # HERO history features
        hero_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_sequence_history = self._dict_to_sequence_history_features(hero_sequence_history_data)
        
        hero_stack_history_data = self.history_analyzer.calculate_stack_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.hero_stack_history = self._dict_to_stack_history_features(hero_stack_history_data)
        
        # OPPONENT history features
        opponent_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_sequence_history = self._dict_to_sequence_history_features(opponent_sequence_history_data)
        
        opponent_stack_history_data = self.history_analyzer.calculate_stack_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_stack_history = self._dict_to_stack_history_features(opponent_stack_history_data)
        
        # NON-SEAT-SPECIFIC history features
        additional_history_data = self.history_analyzer.calculate_additional_history(
            static_ctx, dynamic_ctx
        )
        schema.additional_history = self._dict_to_additional_history_features(additional_history_data)
        
        return schema
    
    def find_feature(self, feature_name: str, game_state: GameState, seat_id: int) -> Optional[tuple]:
        """Find a feature by name and return its location and value."""
        schema = self.get_feature_schema(game_state, seat_id)
        return schema.find_feature(feature_name)
    
    def find_analyzer_for_feature(self, feature_name: str) -> Optional[str]:
        """
        Find which analyzer is responsible for a given feature.
        
        Example: extractor.find_analyzer_for_feature('stack_in_bb')
        Returns: 'CurrentStreetAnalyzer'
        """
        # Create empty schema to check field ownership
        schema = PokerFeatureSchema()
        
        # Check each concept group for the feature
        analyzer_mapping = {
            'my_hand': 'HandAnalyzer',
            'board': 'BoardAnalyzer',
            'hero_current_sequence': 'CurrentStreetAnalyzer',
            'hero_current_stack': 'CurrentStreetAnalyzer', 
            'hero_current_position': 'CurrentStreetAnalyzer',
            'opponent_current_sequence': 'CurrentStreetAnalyzer',
            'opponent_current_stack': 'CurrentStreetAnalyzer', 
            'opponent_current_position': 'CurrentStreetAnalyzer',
            'current_stage': 'CurrentStreetAnalyzer',
            'current_additional': 'CurrentStreetAnalyzer',
            'hero_sequence_history': 'HistoryAnalyzer',
            'hero_stack_history': 'HistoryAnalyzer',
            'opponent_sequence_history': 'HistoryAnalyzer',
            'opponent_stack_history': 'HistoryAnalyzer',
            'additional_history': 'HistoryAnalyzer'
        }
        
        result = schema.find_feature(feature_name)
        if result:
            group_name, _, _ = result
            return analyzer_mapping.get(group_name)
        
        return None
    
    def print_feature_summary(self):
        """Print a summary of the feature schema organization."""
        schema = PokerFeatureSchema()
        print(schema.get_concept_summary())
        
        print("\nSchema → Analyzer Mapping:")
        print("-" * 50)
        print("  my_hand                       → HandAnalyzer")
        print("  board                         → BoardAnalyzer")
        print("  hero_current_sequence         → CurrentStreetAnalyzer")
        print("  hero_current_stack            → CurrentStreetAnalyzer")
        print("  hero_current_position         → CurrentStreetAnalyzer")
        print("  opponent_current_sequence     → CurrentStreetAnalyzer")
        print("  opponent_current_stack        → CurrentStreetAnalyzer")
        print("  opponent_current_position     → CurrentStreetAnalyzer")
        print("  current_stage                 → CurrentStreetAnalyzer")
        print("  current_additional            → CurrentStreetAnalyzer")
        print("  hero_sequence_history         → HistoryAnalyzer")
        print("  hero_stack_history            → HistoryAnalyzer")
        print("  opponent_sequence_history     → HistoryAnalyzer")
        print("  opponent_stack_history        → HistoryAnalyzer")
        print("  additional_history            → HistoryAnalyzer")
        
    def get_analyzer_responsibilities(self) -> Dict[str, List[str]]:
        """Get a map of analyzer responsibilities for documentation."""
        return {
            'HandAnalyzer': [
                'Hand strength via Monte Carlo simulation',
                'Monotonic strength categories (pair+, two_pair+, etc.)',
                'Kicker information for tie-breaking',
                'One-hot hole card encoding (52 features)',
                'Hole card texture analysis (59 features)'
            ],
            'BoardAnalyzer': [
                'One-hot community card encoding (52 features)',
                'Board texture analysis (59 features)',
                'Board composition (pairs, trips, quads)',
                'Rank analysis and board coordination'
            ],
            'CurrentStreetAnalyzer': [
                'Current street sequence features (opponent reproducible)',
                'Current street stack/commitment features', 
                'Current position and stage indicators',
                'Current street additional features (SPR, implied odds)',
                'All features calculable without history tracking'
            ],
            'HistoryAnalyzer': [
                'Sequence history across all streets (28 features)',
                'Stack history across all streets (32 features)',
                'Additional history features (8 features)',
                'Integrates with HistoryTracker for cross-street data'
            ],
            'ActionSequencer': [
                'Current street action logging',
                'Real-time action sequence tracking',
                'Supports CurrentStreetAnalyzer calculations'
            ],
            'HistoryTracker': [
                'Generic key-value historical storage',
                'Cross-street feature persistence',
                'Snapshot-based data management'
            ]
        }
    
    def validate_feature_extraction(self, game_state: GameState, seat_id: int) -> bool:
        """Validate that feature extraction produces expected vector size."""
        try:
            features, _ = self.extract_features(game_state, seat_id)
            expected_size = 507
            actual_size = len(features)
            
            if actual_size != expected_size:
                print(f"Feature extraction validation FAILED:")
                print(f"Expected: {expected_size} features")
                print(f"Actual: {actual_size} features")
                return False
            
            print(f"Feature extraction validation PASSED: {actual_size} features")
            return True
            
        except Exception as e:
            print(f"Feature extraction validation ERROR: {e}")
            return False
        
