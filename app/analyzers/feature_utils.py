# app/analyzers/feature_utils.py
"""
Utility functions for feature processing and sanitization.
"""

from poker_feature_schema import PokerFeatureSchema
from dataclasses import fields


def sanitize_features(features: list, purpose: str) -> list:
    """
    Universal sanitization function using metadata tags from the schema.
    Can be imported and used throughout the codebase.
    
    Args:
        features: The feature vector to sanitize
        purpose: 'training' (removes private & leaky) or 'perceived_range' (removes private only)
        
    Returns:
        Sanitized feature vector with appropriate features zeroed out
    """
    sanitized = list(features)
    schema = PokerFeatureSchema()
    current_index = 0

    # Iterate through the main feature groups (my_hand, board, etc.)
    for group_field in fields(schema):
        group_obj = getattr(schema, group_field.name)
        
        # Iterate through the actual features within each group
        for feature_field in fields(group_obj):
            value = getattr(group_obj, feature_field.name)
            meta = feature_field.metadata or {}  # Handle None metadata
            
            # Determine the size of this feature in the vector
            feature_size = 1
            if hasattr(value, 'to_list'):  # It's a nested dataclass like TextureFeatureSet
                feature_size = len(value.to_list())

            # Check metadata tags and sanitize accordingly
            should_sanitize = False
            if purpose == 'training' and (meta.get('private') or meta.get('leaky')):
                should_sanitize = True
            elif purpose == 'perceived_range' and meta.get('private'):
                should_sanitize = True

            if should_sanitize:
                # Zero out the entire block for this feature
                for i in range(current_index, current_index + feature_size):
                    if i < len(sanitized):
                        sanitized[i] = 0.0
            
            # Advance the index by the size of the feature we just processed
            current_index += feature_size
            
    return sanitized

