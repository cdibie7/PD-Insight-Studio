"""
standardise.py
Handles column normalisation, type inference, and time-axis parsing.
"""
import re
import pandas as pd
import numpy as np
from typing import Optional


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, replace spaces/special chars with underscores, limit to 64 chars."""
    new_cols = {}
    for col in df.columns:
        clean = col.strip().lower()
        clean = re.sub(r'[^a-z0-9]+', '_', clean)
        clean = re.sub(r'_+', '_', clean).strip('_')
        clean = clean[:64]
        new_cols[col] = clean
    df = df.rename(columns=new_cols)
    # Handle duplicate column names after normalisation
    seen = {}
    final_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)
    df.columns = final_cols
    return df


def infer_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce columns to numeric or datetime where appropriate."""
    for col in df.columns:
        # Try numeric
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > 0.7:
                df[col] = converted
                continue
        # Try datetime
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], errors='coerce', infer_format=True)
                if converted.notna().mean() > 0.5:
                    df[col] = converted
            except Exception:
                pass
    return df


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Parse a time column. Detects epoch ms, epoch s, or ISO datetime.
    Returns a Series of datetime or None if parsing fails.
    """
    col = df[time_col]
    # Epoch milliseconds (values > 1e10)
    numeric = pd.to_numeric(col, errors='coerce')
    if numeric.notna().mean() > 0.5:
        if numeric.median() > 1e10:
            return pd.to_datetime(numeric, unit='ms', errors='coerce')
        elif numeric.median() > 1e6:
            return pd.to_datetime(numeric, unit='s', errors='coerce')
    # ISO string
    return pd.to_datetime(col, errors='coerce', infer_format=True)


def compute_months_since_baseline(df: pd.DataFrame, entity_id_col: str, time_col: str) -> pd.Series:
    """
    Given a parsed datetime column and entity id, compute months since first visit per subject.
    Returns a float Series.
    """
    df = df.copy()
    df['_time_parsed'] = parse_time_column(df, time_col)
    baseline = df.groupby(entity_id_col)['_time_parsed'].transform('min')
    delta_days = (df['_time_parsed'] - baseline).dt.total_seconds() / 86400
    return (delta_days / 30.44).round(2)


def merge_datasets(dfs: list, how: str = 'outer') -> pd.DataFrame:
    """Concatenate a list of DataFrames (outer join by default)."""
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    return result
