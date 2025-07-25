"""
Range Predictor Module

Neural network-based opponent range reconstruction system.
Uses game features to predict opponent hand properties and build accurate ranges.
"""

from .range_network import RangeNetwork
from .range_dataset import RangeDataset

__all__ = ['RangeNetwork', 'RangeDataset']