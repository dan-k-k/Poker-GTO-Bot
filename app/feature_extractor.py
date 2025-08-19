# app/feature_extractor.py
# Schema-aligned FeatureExtractor with new analyzer architecture
# Uses CurrentStreetAnalyzer and HistoryAnalyzer to eliminate overlap

import numpy as np
import copy
from typing import List, Dict, Optional, Tuple

from app.poker_core import GameState
from app.feature_contexts import StaticContext, DynamicContext

# Import the master schema
from app.poker_feature_schema import PokerFeatureSchema, CurrentStrategicFeatures, MyHandFeatures, CurrentAdditionalFeatures, AdditionalHistoryFeatures

# Import new unified analyzers
from app.analyzers.street_history_tracking import StreetHistoryTracker
from app.analyzers.hand_analyzer import HandAnalyzer
from app.analyzers.board_analyzer import BoardAnalyzer
from app.analyzers.current_street_analyzer import CurrentStreetAnalyzer
from app.analyzers.street_history_analyzer import StreetHistoryAnalyzer
from app.analyzers.strategic_analyzer import StrategicAnalyzer
from app.analyzers.action_sequencer import ActionSequencer

# Direct imports - no fallback to avoid circular imports
from app.trainingL1.equity_calculator import EquityCalculator
from app.trainingL1.range_constructors import RangeConstructorNN


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
    
    def __init__(self, num_players: int = 2, range_constructor=None, equity_calculator=None):
        # Core tracking components
        self.history_tracker = StreetHistoryTracker()
        self.action_sequencer = ActionSequencer()
        
        # New unified analyzers
        self.hand_analyzer = HandAnalyzer()
        self.board_analyzer = BoardAnalyzer()
        self.current_street_analyzer = CurrentStreetAnalyzer()
        self.history_analyzer = StreetHistoryAnalyzer()
        
        # --- START REFACTOR ---
        # It no longer creates its own instances. It receives the shared ones.
        # Import both fallback options
        from trainingL1.range_constructors import RangeConstructor
        self.range_constructor = range_constructor or RangeConstructor()
        self.equity_calculator = equity_calculator or EquityCalculator()
        self.strategic_analyzer = StrategicAnalyzer(self.range_constructor, self.equity_calculator)
        # --- END REFACTOR ---
        
    def _card_id_to_string(self, card_id: int) -> str:
        """Convert card ID (0-51) to string representation like '2s'."""
        if card_id < 0 or card_id > 51:
            return 'As'  # Fallback
            
        rank_id = card_id // 4
        suit_id = card_id % 4
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        
        return ranks[rank_id] + suits[suit_id]
    
    def _create_initial_schema(self, game_state: GameState, seat_id: int, 
                               self_stats: dict = None,  # ADD this
                               opponent_stats: dict = None) -> PokerFeatureSchema:
        """
        The SINGLE core helper to build a complete schema of all non-leaky
        features (both public and private) from a player's perspective.
        The metadata-driven _schema_to_public_vector will filter out private features as needed.
        """
        schema = PokerFeatureSchema()
        static_ctx = StaticContext(game_state=game_state, seat_id=seat_id)
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        opponent_seat_id = 1 - seat_id if hasattr(game_state, 'num_players') and game_state.num_players == 2 else (seat_id + 1) % 2

        # Board is always public
        schema.board = self.board_analyzer.extract_features(static_ctx.community)
        
        # Current street features are public (action sequences, stacks, positions)
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
        
        # Opponent current street features (also public)
        opp_current_sequence_data = self.current_street_analyzer.calculate_current_street_sequence(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.opponent_current_sequence = self._dict_to_current_sequence_features(opp_current_sequence_data)
        
        opp_current_stack_data = self.current_street_analyzer.calculate_current_street_stack(
            opponent_seat_id, static_ctx, dynamic_ctx, self.action_sequencer
        )  
        schema.opponent_current_stack = self._dict_to_current_stack_features(opp_current_stack_data)
        
        opp_current_position_data = self.current_street_analyzer.calculate_current_position(
            opponent_seat_id, static_ctx
        )
        schema.opponent_current_position = self._dict_to_current_position_features(opp_current_position_data)
                
        # Non-seat-specific features are public
        current_stage_data = self.current_street_analyzer.calculate_current_stage(static_ctx)
        schema.current_stage = self._dict_to_current_stage_features(current_stage_data)
        
        # Current additional features (includes both public and private - metadata will filter)
        current_additional_data = self.current_street_analyzer.calculate_current_street_additional(
            static_ctx, dynamic_ctx, self.action_sequencer
        )
        schema.current_additional = self._dict_to_current_additional_features(current_additional_data)
        
        # History features (non-strategic ones are public)
        self_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_sequence_history = self._dict_to_sequence_history_features(self_sequence_history_data)
        
        opp_sequence_history_data = self.history_analyzer.calculate_sequence_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_sequence_history = self._dict_to_sequence_history_features(opp_sequence_history_data)
        
        self_stack_history_data = self.history_analyzer.calculate_stack_history(
            seat_id, static_ctx, dynamic_ctx
        )
        schema.self_stack_history = self._dict_to_stack_history_features(self_stack_history_data)
        
        opp_stack_history_data = self.history_analyzer.calculate_stack_history(
            opponent_seat_id, static_ctx, dynamic_ctx
        )
        schema.opponent_stack_history = self._dict_to_stack_history_features(opp_stack_history_data)
        
        # Additional history features (includes both public and private - metadata will filter)
        additional_history_data = self.history_analyzer.calculate_additional_history(
            static_ctx, dynamic_ctx
        )
        schema.additional_history = self._dict_to_additional_history_features(additional_history_data)
        
        # === PRIVATE FEATURES ===
        # Private hand features
        if hasattr(game_state, 'hole_cards') and game_state.hole_cards[seat_id]:
            schema.my_hand = self.hand_analyzer.extract_features(static_ctx.hole_cards, static_ctx.community)
        
        # Opponent model features are public
        # --- START CHANGES ---
        # Populate self and opponent models
        if self_stats:
            schema.self_model = self._dict_to_self_model_features(self_stats)  # Can reuse the helper
        if opponent_stats:
            schema.opponent_model = self._dict_to_opponent_model_features(opponent_stats)
        # --- END CHANGES ---

        return schema

    def extract_public_features(self, game_state: GameState, seat_id: int, 
                                self_stats: dict = None, # ADD THIS
                                opponent_stats: dict = None) -> List[float]:
        """
        Extracts a SHORTER feature vector containing ONLY public information...
        """
        # Build the complete schema first, passing both sets of stats
        complete_schema = self._create_initial_schema(game_state, seat_id, self_stats, opponent_stats)
        
        # Filter to public-only using smart metadata-driven function
        return self._schema_to_public_vector(complete_schema)

    
    def _schema_to_public_vector(self, schema: PokerFeatureSchema) -> List[float]:
        """
        Automatically builds the public feature vector by iterating through the
        schema and including only fields NOT marked as 'private' or 'leaky'.
        This makes the schema the single source of truth for what's public vs private.
        """
        from dataclasses import fields
        public_vector = []
        
        # Iterate through the main groups in the schema (my_hand, board, etc.)
        for group_field in fields(PokerFeatureSchema):
            group_obj = getattr(schema, group_field.name)
            
            # Check metadata on the group itself (e.g., the entire MyHandFeatures might be private)
            is_group_private = group_field.metadata.get('private', False)
            is_group_leaky = group_field.metadata.get('leaky', False)

            if is_group_private or is_group_leaky:
                continue  # Skip this entire group

            # Check if this group has individual features we can inspect
            if hasattr(group_obj, '__dataclass_fields__'):
                # This group has individual fields - check each one
                group_vector = []
                for feature_field in fields(group_obj):
                    is_feature_private = feature_field.metadata.get('private', False)
                    is_feature_leaky = feature_field.metadata.get('leaky', False)

                    if not is_feature_private and not is_feature_leaky:
                        value = getattr(group_obj, feature_field.name)
                        # Handle nested dataclass features (like TextureFeatureSet)
                        if hasattr(value, 'to_list'):
                            group_vector.extend(value.to_list())
                        else:
                            group_vector.append(value)
                
                public_vector.extend(group_vector)
            elif hasattr(group_obj, 'to_list'):
                # This group doesn't have individual field metadata, so include the whole group
                # (e.g., board features, position features that are all public)
                public_vector.extend(group_obj.to_list())
        
        return public_vector
    
    def _extract_opponent_public_features_from_schema(self, schema: PokerFeatureSchema) -> List[float]:
        """
        Efficiently extract opponent public features from an existing schema.
        Reuses opponent features already computed in the schema instead of rebuilding.
        """
        from dataclasses import fields
        opponent_vector = []
        
        # Extract opponent-specific features that are public
        opponent_groups = ['opponent_current_sequence', 'opponent_current_stack', 'opponent_current_position', 
                          'opponent_sequence_history', 'opponent_stack_history']
        
        for group_name in opponent_groups:
            if hasattr(schema, group_name):
                group_obj = getattr(schema, group_name)
                if hasattr(group_obj, 'to_list'):
                    opponent_vector.extend(group_obj.to_list())
        
        # Add shared public features (board, stage, etc.)
        shared_groups = ['board', 'current_stage']
        for group_name in shared_groups:
            if hasattr(schema, group_name):
                group_obj = getattr(schema, group_name)
                if hasattr(group_obj, 'to_list'):
                    opponent_vector.extend(group_obj.to_list())
        
        # Add public parts of current_additional (filtering out private features)
        if hasattr(schema, 'current_additional'):
            group_obj = schema.current_additional
            if hasattr(group_obj, '__dataclass_fields__'):
                for feature_field in fields(group_obj):
                    is_feature_private = feature_field.metadata.get('private', False)
                    is_feature_leaky = feature_field.metadata.get('leaky', False)
                    
                    if not is_feature_private and not is_feature_leaky:
                        value = getattr(group_obj, feature_field.name)
                        if hasattr(value, 'to_list'):
                            opponent_vector.extend(value.to_list())
                        else:
                            opponent_vector.append(value)
        
        # Add public parts of additional_history
        if hasattr(schema, 'additional_history'):
            group_obj = schema.additional_history
            if hasattr(group_obj, '__dataclass_fields__'):
                for feature_field in fields(group_obj):
                    is_feature_private = feature_field.metadata.get('private', False)
                    is_feature_leaky = feature_field.metadata.get('leaky', False)
                    
                    if not is_feature_private and not is_feature_leaky:
                        value = getattr(group_obj, feature_field.name)
                        if hasattr(value, 'to_list'):
                            opponent_vector.extend(value.to_list())
                        else:
                            opponent_vector.append(value)
        
        return opponent_vector
    
    def _create_strategic_context(self, game_state: GameState, seat_id: int, public_features: List[float],
                                  self_stats: Dict = None, opponent_stats: Dict = None) -> dict:
        """Create context dictionary for strategic analysis using public features."""
        # Convert card IDs to strings
        hand_strings = []
        if hasattr(game_state, 'hole_cards') and seat_id < len(game_state.hole_cards):
            hand_strings = [self._card_id_to_string(c) for c in game_state.hole_cards[seat_id]]
        
        board_strings = []
        if hasattr(game_state, 'community'):
            board_strings = [self._card_id_to_string(c) for c in game_state.community]
        
        return {
            'public_features': public_features,
            'static_ctx': StaticContext(game_state, seat_id),
            'self_stats': self_stats or {}, # ADD this
            'opponent_stats': opponent_stats or {}, # RENAME this key for clarity
            'hand_strings': hand_strings,
            'board_strings': board_strings,
            'pot': getattr(game_state, 'pot', 100),
            'history_tracker': self.history_tracker,
            'action_sequencer': self.action_sequencer # Add this line
        }

    def extract_features(self, game_state: GameState, seat_id: int, role: str,
                        self_stats: Dict = None,
                        opponent_stats: Dict = None) -> Tuple[np.ndarray, PokerFeatureSchema]:
        """
        Main orchestration method. Follows a clean, direct 3-phase process.
        
        Args:
            game_state: Current game state
            seat_id: Player seat ID (self)
            self_stats: Optional dict with self's poker statistics
            opponent_stats: Optional dict with opponent's poker statistics
            
        Returns:
            Tuple of (feature vector, structured schema) for ML training and debugging
        """
        self_seat_id = seat_id
        opponent_seat_id = 1 - seat_id if hasattr(game_state, 'num_players') and game_state.num_players == 2 else (seat_id + 1) % 2
        if role == "AS":
            opponent_stats = None

        # === PHASE 1: Build the Complete Non-Leaky Schema ===
        # This schema contains all non-leaky features (both public and private)
        # 2. Update the schema creation call
        final_schema = self._create_initial_schema(game_state, self_seat_id, self_stats, opponent_stats)
        
        # === PHASE 2: Strategic Analysis (Leaky Features) ===
        # Always run strategic analysis - it will use NN if available, otherwise heuristic fallback
        # 2.1 Get public-only features efficiently from the already-built schema
        self_public_features = self._schema_to_public_vector(final_schema)
        
        # 2.2 For opponent, reuse opponent features already computed in Phase 1
        # Extract opponent public features from the existing schema instead of rebuilding
        opp_public_features = self._extract_opponent_public_features_from_schema(final_schema)
        
        # 2.3 Create contexts with public features for clean range prediction
        self_context = self._create_strategic_context(game_state, self_seat_id, self_public_features, self_stats, opponent_stats)
        opponent_context = self._create_strategic_context(game_state, opponent_seat_id, opp_public_features, opponent_stats, self_stats)
        
        # 2.4 Calculate all strategic (leaky) features
        strategic_features = self.strategic_analyzer.calculate_features(self_context, opponent_context)
        
        # 2.5 Populate the schema with strategic results
        final_schema.current_strategic = self._dict_to_current_strategic_features(strategic_features)
        
        # === PHASE 3: Add Strategic History ===
        # Add strategic history features (leaky, depends on strategic analysis from previous streets)
        static_ctx = StaticContext(game_state=game_state, seat_id=self_seat_id)
        dynamic_ctx = DynamicContext(self.history_tracker)
        
        final_schema.strategic_history = self._dict_to_strategic_history_features(
            self.history_analyzer.calculate_strategic_history(static_ctx, dynamic_ctx)
        )
        
        # And update the call to to_vector() in feature_extractor.py
        # At the very end of the function, before the return
        return final_schema.to_vector(role=role), final_schema
    
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
    
    def save_street_snapshot(self, game_state: GameState, strategic_features: dict = None, street: str = None):
        """Save street snapshot for history tracking using ActionSequencer data."""
        static_ctx = StaticContext(game_state=game_state, seat_id=0)  # seat_id not used for snapshot
        dynamic_ctx = DynamicContext(history_tracker=self.history_tracker)
        
        # Get the complete action log from ActionSequencer for accurate historical data
        action_log = self.action_sequencer.get_live_action_sequence()
        
        # Save accurate snapshot using ActionSequencer data with strategic features
        self.history_analyzer.save_street_snapshot(action_log, static_ctx, dynamic_ctx, strategic_features, street)
    
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
    
    def _dict_to_self_model_features(self, data: dict):
        from poker_feature_schema import SelfModelFeatures
        return SelfModelFeatures(**data)
    
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
            expected_size = 790
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
        
