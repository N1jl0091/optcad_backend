import pandas as pd

def lag_column(df: pd.DataFrame, col: str, n: int = 1) -> pd.Series:
    """Return lagged column by n steps."""
    return df[col].shift(n)

def compute_diff(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute difference between consecutive values."""
    return df[col].diff().fillna(0)

def filter_rows(df: pd.DataFrame, condition) -> pd.DataFrame:
    """Filter DataFrame by condition (boolean series)."""
    return df[condition].copy()
