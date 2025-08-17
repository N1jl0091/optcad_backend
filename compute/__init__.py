# Expose the main wrapper and other functions for easy imports
from .optcad_compute import process_activity_stream
from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    cadence_binning,
    optimal_cadence,
)
from .config import *
__all__ = [
    "process_activity_stream",
    "prepare_data",
    "calculate_gradient",
    "segment_ride",
    "calculate_cadence_elevation",
    "aggregate_segments",
    "compute_scores",
    "cadence_binning",
    "optimal_cadence",
]
