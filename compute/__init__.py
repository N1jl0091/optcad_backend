# Expose main functions for easy import
from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    cadence_binning,
    optimal_cadence
)
from .config import *
