# feature_extractor.py
# Schema-aligned FeatureExtractor with new analyzer architecture
# Uses CurrentStreetAnalyzer and HistoryAnalyzer to eliminate overlap

import numpy as np
import copy
from typing import List, Dict, Optional, Tuple

from poker_core import GameState
from feature_contexts import StaticContext, DynamicContext

# Import the master schema
from poker_feature_schema import PokerFeatureSchema, CurrentStrategicFeatures, MyHandFeatures

# Import new unified analyzers
from analyzers.street_history_tracking import StreetHistoryTracker
from analyzers.hand_analyzer import HandAnalyzer
from analyzers.board_analyzer import BoardAnalyzer
from analyzers.current_street_analyzer import CurrentStreetAnalyzer
from analyzers.street_history_analyzer import StreetHistoryAnalyzer
from analyzers.strategic_analyzer import StrategicAnalyzer
from analyzers.action_sequencer import ActionSequencer
from trainingL1.equity_calculator import EquityCalculator, RangeConstructorNN


class FeatureExtractor:
    """
    Schema-aligned FeatureExtractor with new unified analyzer architecture.
    
    Eliminates overlap between old analyzers using:
    - CurrentStreetAnalyzer: Non-history tracked, opponent reproducible features
    - StreetHistoryAnalyzer: History tracked features with proper StreetHistoryTracker integration
    
    Schema Mapping:
    - MyHandFeatures                    → HandAnalyzer
    - BoardFeatures                     → BoardAnalyzer  
    - CurrentStreetSequenceFeatures     → CurrentStreetAnalyzer
    - CurrentStreetStackFeatures        → CurrentStreetAnalyzer
    - CurrentPositionFeatures           → CurrentStreetAnalyzer
    - CurrentStageFeatures              → CurrentStreetAnalyzer
    - CurrentAdditionalFeatures         → CurrentStreetAnalyzer
    - SequenceHistoryFeatures           → StreetHistoryAnalyzer
    - StackHistoryFeatures              → StreetHistoryAnalyzer
    - AdditionalHistoryFeatures         → StreetHistoryAnalyzer
    """
    
    def __init__(self, num_players: int = 2):
        # Core tracking components
        self.history_tracker = StreetHistoryTracker()
        self.action_sequencer = ActionSequencer()
        
        # New unified analyzers
        self.hand_analyzer = HandAnalyzer()
        self.board_analyzer = BoardAnalyzer()
        self.current_street_analyzer = CurrentStreetAnalyzer()
        self.history_analyzer = StreetHistoryAnalyzer()
        
        # Strategic analysis components
        self.range_constructor = RangeConstructorNN()
        self.equity_calculator = EquityCalculator()
        self.strategic_analyzer = StrategicAnalyzer(self.range_constructor, self.equity_calculator)
    
    def _gather_context(self, game_state: GameState, seat_id: int, schema_obj: PokerFeatureSchema, 
                       action_history: Dict = None, opponent_stats: Dict = None) -> dict:
        """Organizational helper to bundle a player's full context into a single dictionary."""
        # Convert card IDs to strings
        hand_strings = []
        if hasattr(game_state, 'hole_cards') and seat_id < len(game_state.hole_cards):
            hand_strings = [self._card_id_to_string(c) for c in game_state.hole_cards[seat_id]]
        
        board_strings = []
        if hasattr(game_state, 'community'):
            board_strings = [self._card_id_to_string(c) for c in game_state.community]
        
        return {
            'features': schema_obj.to_vector(),
            'static_ctx': StaticContext(game_state, seat_id),
            'stats': opponent_stats or {},
            'action_history': action_history or [],
            'hand_strings': hand_strings,
            'board_strings': board_strings,
            'pot': getattr(game_state, 'pot', 100)
        }
    
    def _card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to string representation like '2s'."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _extract_base_schema(self, game_state: GameState, seat_id: int, 
                             opponent_stats: dict = None) -> PokerFeatureSchema:
        """
        Helper to build a complete schema of all non-strategic features
        from a specific player's perspective.
        """
        schema = PokerFeatureSchema()
        static_ctx = StaticContext(game_state=game_state, seat_id=seat_id)
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        opponent_seat_id = 1 - seat_id if static_ctx.num_players == 2 else (seat_id + 1) % static_ctx.num_players

        # Only populate hand features if we know the cards (naturally empty for opponent from self's perspective)
        if hasattr(game_state, 'hole_cards') and seat_id < len(game_state.hole_cards) and game_state.hole_cards[seat_id]:
            schema.my_hand = self.hand_analyzer.extract_features(static_ctx.hole_cards, static_ctx.community)
        
        schema.board = self.board_analyzer.extract_features(static_ctx.community)
        
        # Current street features
        self_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.self_current_sequence = self._dict_to_current_sequence_features(self_current_sequence_data)
        
        self_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.self_current_stack = self._dict_to_current_stack_features(self_current_stack_data)
        
        self_current_position_data = self.current_street_analyzer.calculate_current_position(
            seat_id, static_ctx
        )
        schema.self_current_position = self._dict_to_current_position_features(self_current_position_data)
        
        # Opponent features
        opponent_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_sequence = self._dict_to_current_sequence_features(opponent_current_sequence_data)
        
        opponent_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_stack = self._dict_to_current_stack_features(opponent_current_stack_data)
        
        opponent_current_position_data = self.current_street_analyzer.calculate_current_position(
            opponent_seat_id, static_ctx
        )
        schema.opponent_current_position = self._dict_to_current_position_features(opponent_current_position_data)
        
        # Non-seat-specific features
        current_stage_data = self.current_street_analyzer.calculate_current_stage(static_ctx)
        schema.current_stage = self._dict_to_current_stage_features(current_stage_data)
        
        current_additional_data = self.current_street_analyzer.calculate_current_street_additional(
            static_ctx, dynamic_ctx, self.action_sequencer, opponent_stats, None
        )
        schema.current_additional = self._dict_to_current_additional_features(current_additional_data)
        
        # History features
        self_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_sequence_history = self._dict_to_sequence_history_features(self_sequence_history_data)
        
        self_stack_history_data = self.history_analyzer.calculate_stack_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_stack_history = self._dict_to_stack_history_features(self_stack_history_data)
        
        opponent_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_sequence_history = self._dict_to_sequence_history_features(opponent_sequence_history_data)
        
        opponent_stack_history_data = self.history_analyzer.calculate_stack_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_stack_history = self._dict_to_stack_history_features(opponent_stack_history_data)
        
        additional_history_data = self.history_analyzer.calculate_additional_history(
            static_ctx, dynamic_ctx
        )
        schema.additional_history = self._dict_to_additional_history_features(additional_history_data)
        
        # Opponent model features
        if opponent_stats:
            schema.opponent_model = self._dict_to_opponent_model_features(opponent_stats)

        return schema

    def extract_features(self, game_state: GameState, seat_id: int, 
                        opponent_stats: Dict = None, full_hand_action_history: Dict = None) -> Tuple[np.ndarray, PokerFeatureSchema]:
        """
        Main orchestration method. Follows a clean, non-recursive 3-phase process.
        
        Args:
            game_state: Current game state
            seat_id: Player seat ID (self)
            opponent_stats: Optional dict with opponent's poker statistics
            full_hand_action_history: Optional dict of full hand action history
            
        Returns:
            Tuple of (feature vector, structured schema) for ML training and debugging
        """
        self_seat_id = seat_id
        opponent_seat_id = 1 - seat_id if hasattr(game_state, 'num_players') and game_state.num_players == 2 else (seat_id + 1) % 2
        
        # === PHASE 1: Base Feature Extraction ===
        # Create the full base feature set from the self's perspective
        self_schema = self._extract_base_schema(game_state, self_seat_id, opponent_stats)
        
        # === PHASE 2: Strategic Analysis ===
        # Only run if the RangePredictorNN is trained and loaded
        if hasattr(self.range_constructor, 'model') and self.range_constructor.model is not None:
            # 2.1 Get the opponent's feature vector using the same universal helper
            # This vector will naturally lack the opponent's private hand info
            opp_schema = self._extract_base_schema(game_state, opponent_seat_id, opponent_stats)
            
            # 2.2 Gather the contexts for the StrategicAnalyzer
            self_context = self._gather_context(game_state, self_seat_id, self_schema, full_hand_action_history, opponent_stats)
            opponent_context = self._gather_context(game_state, opponent_seat_id, opp_schema, full_hand_action_history, opponent_stats)
            
            # 2.3 Make one clean call to get all strategic features
            strategic_features = self.strategic_analyzer.calculate_features(self_context, opponent_context)
            
            # 2.4 Populate the self's schema with the results
            self_schema.current_strategic = self._dict_to_current_strategic_features(strategic_features)
        
        # === PHASE 3: Final History Population and Return ===
        # The history calculated here now correctly reflects the final state
        static_ctx = StaticContext(game_state, self_seat_id)
        dynamic_ctx = DynamicContext(self.history_tracker)
        self_schema.strategic_history = self._dict_to_strategic_history_features(
            self.history_analyzer.calculate_strategic_history(static_ctx, dynamic_ctx)
        )
        
        return self_schema.to_vector(), self_schema
    
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
    
    def _dict_to_opponent_model_features(self, data: dict):
        from poker_feature_schema import OpponentModelFeatures
        return OpponentModelFeatures(**data)
    
    def _dict_to_current_strategic_features(self, data: dict):
        from poker_feature_schema import CurrentStrategicFeatures
        return CurrentStrategicFeatures(**data)
    
    def _dict_to_strategic_history_features(self, data: dict):
        from poker_feature_schema import StrategicHistoryFeatures
        return StrategicHistoryFeatures(**data)
    
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
        
        # self current street features
        self_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.self_current_sequence = self._dict_to_current_sequence_features(self_current_sequence_data)
        
        self_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.self_current_stack = self._dict_to_current_stack_features(self_current_stack_data)
        
        self_current_position_data = self.current_street_analyzer.calculate_current_position(
            seat_id, static_ctx
        )
        schema.self_current_position = self._dict_to_current_position_features(self_current_position_data)
        
        # OPPONENT current street features
        opponent_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_sequence = self._dict_to_current_sequence_features(opponent_current_sequence_data)
        
        opponent_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
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
            static_ctx, dynamic_ctx, self.action_sequencer, opponent_stats, opponent_action_history
        )
        schema.current_additional = self._dict_to_current_additional_features(current_additional_data)
        
        # === HISTORY FEATURES (History tracked) ===
        # self history features
        self_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_sequence_history = self._dict_to_sequence_history_features(self_sequence_history_data)
        
        self_stack_history_data = self.history_analyzer.calculate_stack_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_stack_history = self._dict_to_stack_history_features(self_stack_history_data)
        
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
            'self_current_sequence': 'CurrentStreetAnalyzer',
            'self_current_stack': 'CurrentStreetAnalyzer', 
            'self_current_position': 'CurrentStreetAnalyzer',
            'opponent_current_sequence': 'CurrentStreetAnalyzer',
            'opponent_current_stack': 'CurrentStreetAnalyzer', 
            'opponent_current_position': 'CurrentStreetAnalyzer',
            'current_stage': 'CurrentStreetAnalyzer',
            'current_additional': 'CurrentStreetAnalyzer',
            'self_sequence_history': 'StreetHistoryAnalyzer',
            'self_stack_history': 'StreetHistoryAnalyzer',
            'opponent_sequence_history': 'StreetHistoryAnalyzer',
            'opponent_stack_history': 'StreetHistoryAnalyzer',
            'additional_history': 'StreetHistoryAnalyzer'
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
        print("  self_current_sequence         → CurrentStreetAnalyzer")
        print("  self_current_stack            → CurrentStreetAnalyzer")
        print("  self_current_position         → CurrentStreetAnalyzer")
        print("  opponent_current_sequence     → CurrentStreetAnalyzer")
        print("  opponent_current_stack        → CurrentStreetAnalyzer")
        print("  opponent_current_position     → CurrentStreetAnalyzer")
        print("  current_stage                 → CurrentStreetAnalyzer")
        print("  current_additional            → CurrentStreetAnalyzer")
        print("  self_sequence_history         → StreetHistoryAnalyzer")
        print("  self_stack_history            → StreetHistoryAnalyzer")
        print("  opponent_sequence_history     → StreetHistoryAnalyzer")
        print("  opponent_stack_history        → StreetHistoryAnalyzer")
        print("  additional_history            → StreetHistoryAnalyzer")
        
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
            'StreetHistoryAnalyzer': [
                'Sequence history across all streets (28 features)',
                'Stack history across all streets (32 features)',
                'Additional history features (8 features)',
                'Integrates with StreetHistoryTracker for cross-street data'
            ],
            'ActionSequencer': [
                'Current street action logging',
                'Real-time action sequence tracking',
                'Supports CurrentStreetAnalyzer calculations'
            ],
            'StreetHistoryTracker': [
                'Generic key-value historical storage',
                'Cross-street feature persistence',
                'Snapshot-based data management'
            ]
        }
    
    def validate_feature_extraction(self, game_state: GameState, seat_id: int) -> bool:
        """Validate that feature extraction produces expected vector size."""
        try:
            features, _ = self.extract_features(game_state, seat_id)
            expected_size = 694
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
        
