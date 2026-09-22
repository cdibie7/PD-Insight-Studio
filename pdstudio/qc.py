"""
qc.py
Quality control module — transparent flagging, no silent deletions.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def run_qc(
    df: pd.DataFrame,
    required_cols: List[str] = None,
    col_missingness_threshold: float = 0.8,
    row_missingness_threshold: float = 0.8,
    duplicate_keys: List[str] = None,
    outlier_method: str = 'iqr',      # 'iqr' or 'zscore'
    outlier_threshold: float = 3.0,   # z-score OR IQR multiplier
    numeric_cols: List[str] = None,   # cols to check for outliers
    time_col: str = None,
) -> pd.DataFrame:
    """
    Adds qc_reason (str) and qc_keep (bool) columns to the dataframe.
    Never deletes rows. Returns the annotated dataframe.
    """
    df = df.copy()
    reasons = pd.Series([''] * len(df), index=df.index, dtype=object)

    # 1. Missing required columns (mark all rows)
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                _append_reason(reasons, df.index, f"missing_required_col:{col}")
            else:
                missing_mask = df[col].isna()
                _append_reason_mask(reasons, missing_mask, f"null_required:{col}")

    # 2. High-missingness columns flagged as informational
    col_miss = df.isnull().mean()
    high_miss_cols = col_miss[col_miss > col_missingness_threshold].index.tolist()
    # We don't flag rows for this, just report it in summary

    # 3. Row-level missingness
    row_miss = df.isnull().mean(axis=1)
    _append_reason_mask(reasons, row_miss > row_missingness_threshold, "high_row_missingness")

    # 4. Duplicates
    if duplicate_keys:
        valid_keys = [k for k in duplicate_keys if k in df.columns]
        if valid_keys:
            dup_mask = df.duplicated(subset=valid_keys, keep='first')
            _append_reason_mask(reasons, dup_mask, "duplicate_key")

    # 5. Outliers in numeric columns
    if numeric_cols:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() < 10:
                continue
            if outlier_method == 'iqr':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lo = q1 - outlier_threshold * iqr
                hi = q3 + outlier_threshold * iqr
                out_mask = series.notna() & ((series < lo) | (series > hi))
            else:  # zscore
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue
                z = (series - mean) / std
                out_mask = series.notna() & (z.abs() > outlier_threshold)
            _append_reason_mask(reasons, out_mask, f"outlier:{col}")

    df['qc_reason'] = reasons
    df['qc_keep'] = df['qc_reason'] == ''
    return df


def _append_reason(reasons: pd.Series, idx, msg: str):
    for i in idx:
        if reasons[i]:
            reasons[i] += ',' + msg
        else:
            reasons[i] = msg


def _append_reason_mask(reasons: pd.Series, mask: pd.Series, msg: str):
    for i in reasons.index[mask]:
        if reasons[i]:
            reasons[i] += ',' + msg
        else:
            reasons[i] = msg


def qc_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a summary dict for display."""
    total = len(df)
    kept = df['qc_keep'].sum()
    removed = total - kept

    # Count by reason
    reason_counts = {}
    for reasons in df.loc[~df['qc_keep'], 'qc_reason']:
        for r in str(reasons).split(','):
            r = r.strip()
            if r:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    # High-missingness columns
    col_miss = df.drop(columns=['qc_reason', 'qc_keep'], errors='ignore').isnull().mean()
    high_miss = col_miss[col_miss > 0.2].sort_values(ascending=False).head(10).to_dict()

    return {
        'total': total,
        'kept': int(kept),
        'removed': int(removed),
        'kept_pct': round(100 * kept / total, 1) if total else 0,
        'reason_counts': dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
        'high_missingness_cols': {k: round(v * 100, 1) for k, v in high_miss.items()},
    }
