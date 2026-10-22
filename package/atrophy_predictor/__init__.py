"""
Atrophy Predictor - A tool for predicting brain atrophy patterns from clinical notes.
"""

__version__ = "0.1.0"

from .core.predictor import AtrophyPredictor
from .models.schema import AtrophyPredictions, Hemisphere
from .utils.comparator import ZScoreComparator

__all__ = [
    'AtrophyPredictor',
    'AtrophyPredictions',
    'Hemisphere',
    'ZScoreComparator',
] 