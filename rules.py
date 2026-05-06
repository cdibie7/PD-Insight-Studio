"""
rules.py
Applies user-defined rules to derive new columns without coding.
Each rule is a dict describing how to create a new variable.
"""
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any


def apply_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a list of rule dicts to the dataframe.
    Returns df with new derived columns added.
    """
    df = df.copy()
    for rule in rules:
        try:
            rule_type = rule.get('type')
            if rule_type == 'keyword_map':
                df = _apply_keyword_map(df, rule)
            elif rule_type == 'months_since_baseline':
                df = _apply_months_since_baseline(df, rule)
            elif rule_type == 'numeric_bin':
                df = _apply_numeric_bin(df, rule)
            elif rule_type == 'proxy_label':
                df = _apply_proxy_label(df, rule)
            elif rule_type == 'formula':
                df = _apply_formula(df, rule)
        except Exception as e:
            print(f"[Rule error: {rule.get('output_col', '?')}] {e}")
    return df


def _apply_keyword_map(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    Map keywords in a source column to category labels.
    rule = {
        'type': 'keyword_map',
        'source_col': 'medication_status',
        'output_col': 'med_state',
        'mappings': [
            {'keywords': ['before', 'pre', 'off'], 'label': 'BEFORE'},
            {'keywords': ['after', 'post', 'on'],  'label': 'AFTER'},
            {'keywords': ['none', 'no med', 'control'], 'label': 'NO_MEDS'},
        ],
        'default': 'OTHER'
    }
    """
    src = rule['source_col']
    out = rule['output_col']
    mappings = rule.get('mappings', [])
    default = rule.get('default', 'OTHER')

    if src not in df.columns:
        df[out] = default
        return df

    result = pd.Series([default] * len(df), index=df.index, dtype=object)
    col_lower = df[src].astype(str).str.lower()

    for mapping in reversed(mappings):  # later rules win
        pattern = '|'.join(re.escape(k.lower()) for k in mapping['keywords'])
        if pattern:
            mask = col_lower.str.contains(pattern, na=False)
            result[mask] = mapping['label']

    df[out] = result
    return df


def _apply_months_since_baseline(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    rule = {
        'type': 'months_since_baseline',
        'time_col': 'visit_date',
        'entity_id_col': 'patient_id',
        'output_col': 'months_since_baseline'
    }
    """
    from standardise import compute_months_since_baseline
    tc = rule['time_col']
    eid = rule['entity_id_col']
    out = rule.get('output_col', 'months_since_baseline')
    if tc in df.columns and eid in df.columns:
        df[out] = compute_months_since_baseline(df, eid, tc)
    return df


def _apply_numeric_bin(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    rule = {
        'type': 'numeric_bin',
        'source_col': 'age',
        'output_col': 'age_group',
        'bins': [0, 50, 65, 999],
        'labels': ['<50', '50-65', '65+']
    }
    """
    src = rule['source_col']
    out = rule['output_col']
    if src not in df.columns:
        return df
    df[out] = pd.cut(pd.to_numeric(df[src], errors='coerce'),
                     bins=rule['bins'], labels=rule['labels'], right=True)
    return df


def _apply_proxy_label(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    Create a diagnosis proxy from another column.
    rule = {
        'type': 'proxy_label',
        'source_col': 'med_state',
        'output_col': 'dx_proxy',
        'mapping': {'NO_MEDS': 'Control', '__default__': 'PD'}
    }
    """
    src = rule['source_col']
    out = rule['output_col']
    mapping = rule.get('mapping', {})
    default = mapping.get('__default__', 'Unknown')
    if src not in df.columns:
        df[out] = default
        return df
    df[out] = df[src].astype(str).map(lambda x: mapping.get(x, default))
    return df


def _apply_formula(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    Simple arithmetic formula using column names.
    rule = {
        'type': 'formula',
        'output_col': 'tap_rate',
        'expression': 'tap_count / duration_seconds'
    }
    Only supports basic arithmetic between column references and constants.
    """
    out = rule['output_col']
    expr = rule.get('expression', '')
    # Safe evaluation: only allow col references, operators, numbers
    allowed_pattern = r'^[\w\s\+\-\*\/\(\)\.]+$'
    if not re.match(allowed_pattern, expr):
        return df
    local_vars = {col: pd.to_numeric(df[col], errors='coerce') for col in df.columns if col in expr}
    try:
        df[out] = eval(expr, {"__builtins__": {}}, local_vars)
    except Exception as e:
        print(f"[Formula error] {e}")
    return df


def validate_rules(rules: List[Dict]) -> List[str]:
    """Return a list of warning messages for invalid rules."""
    warnings = []
    for i, r in enumerate(rules):
        if 'type' not in r:
            warnings.append(f"Rule {i}: missing 'type' field.")
        if 'output_col' not in r:
            warnings.append(f"Rule {i}: missing 'output_col' field.")
    return warnings
