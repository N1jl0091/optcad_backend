from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    optimal_cadence
)
from .config import TIME_LIMIT_SEC, GRAD_LIMIT_PCT, GRADIENT_WINDOW, BIN_SIZE, EXERTION_A, EXERTION_B, EXERTION_C
import pandas as pd

def process_activity_stream(streams: dict) -> dict:
    """
    Process Strava activity streams and compute OptCad segments & performance.

    streams: dict containing Strava stream data ('time', 'cadence', 'distance', 'altitude', 'moving', etc.)
    Returns: dict with segments, performance score, and optimal cadence
    """
    # Convert streams to DataFrame
    df = pd.DataFrame({
        'time': streams.get('time', []),
        'cadence': streams.get('cadence', []),
        'speed': streams.get('velocity_smooth', []),
        'Distance': streams.get('distance', []),
        'Altitude': streams.get('altitude', []),
        'moving': streams.get('moving', [])
    })

    # 1. Prepare data
    df = prepare_data(df)

    # 2. Calculate gradient
    df = calculate_gradient(df, window=GRADIENT_WINDOW)

    # 3. Segment ride
    df = segment_ride(df, time_limit=TIME_LIMIT_SEC, grad_limit=GRAD_LIMIT_PCT)

    # 4. Compute cadence and elevation
    df = calculate_cadence_elevation(df)

    # 5. Aggregate segments
    agg_df = aggregate_segments(df)

    # 6. Compute scores
    agg_df = compute_scores(agg_df, a=EXERTION_A, b=EXERTION_B, c=EXERTION_C)

    # 7. Compute optimal cadence
    opt_cadence = optimal_cadence(agg_df)

    # Convert all DataFrames to JSON-serializable lists of dicts
    segments = agg_df.to_dict(orient='records')

    # If optimal_cadence includes DataFrames internally, convert them too
    if isinstance(opt_cadence.get("details"), pd.DataFrame):
        opt_cadence["details"] = opt_cadence["details"].to_dict(orient='records')

    return {
        "segments": segments,
        "optimal_cadence": opt_cadence
    }
