"""
insights.py
Generates descriptive statistics, group comparisons, time trends, and narrative text.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Dict, Any


# ─── Descriptive Statistics ─────────────────────────────────────────────────

def describe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a summary DataFrame for numeric columns."""
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            continue
        rows.append({
            'column': col,
            'n': len(s),
            'missing_pct': round(100 * df[col].isna().mean(), 1),
            'mean': round(s.mean(), 4),
            'sd': round(s.std(), 4),
            'median': round(s.median(), 4),
            'iqr': round(s.quantile(0.75) - s.quantile(0.25), 4),
            'min': round(s.min(), 4),
            'max': round(s.max(), 4),
        })
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Return {col: counts_df} for categorical columns."""
    result = {}
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        pct = (vc / len(df) * 100).round(1)
        result[col] = pd.DataFrame({'count': vc, 'pct': pct}).reset_index().rename(columns={'index': col})
    return result


# ─── Group Comparisons ───────────────────────────────────────────────────────

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d between two groups."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt(((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2) / (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def eta_squared(df: pd.DataFrame, feature_col: str, group_col: str) -> float:
    """One-way ANOVA eta-squared."""
    groups = [g.dropna() for _, g in df.groupby(group_col)[feature_col].apply(lambda x: pd.to_numeric(x, errors='coerce'))]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    grand_mean = df[feature_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
    return float(ss_between / ss_total) if ss_total else np.nan


def compare_groups(df: pd.DataFrame, feature_cols: List[str], group_col: str) -> pd.DataFrame:
    """
    Compare numeric features across groups.
    Returns a DataFrame with stats, test result, and effect size.
    """
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []
    groups = df[group_col].dropna().unique()

    for col in feature_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        group_data = {g: numeric[df[group_col] == g].dropna() for g in groups}
        group_data = {g: v for g, v in group_data.items() if len(v) >= 2}
        if len(group_data) < 2:
            continue

        row = {'feature': col, 'groups_compared': str(list(group_data.keys()))}
        for g, vals in group_data.items():
            row[f'mean_{g}'] = round(vals.mean(), 4)
            row[f'sd_{g}'] = round(vals.std(), 4)
            row[f'n_{g}'] = len(vals)

        # Statistical test
        if len(group_data) == 2:
            a, b = list(group_data.values())
            _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            row['test'] = 'Mann-Whitney U'
            row['p_value'] = round(p, 4)
            row['effect_size'] = round(cohens_d(a, b), 4)
            row['effect_size_type'] = "Cohen's d"
        else:
            group_arrays = list(group_data.values())
            _, p = stats.kruskal(*group_arrays)
            row['test'] = 'Kruskal-Wallis'
            row['p_value'] = round(p, 4)
            es = eta_squared(df, col, group_col)
            row['effect_size'] = round(es, 4) if not np.isnan(es) else None
            row['effect_size_type'] = 'eta-squared'

        rows.append(row)

    return pd.DataFrame(rows)


def within_subject_comparison(df: pd.DataFrame, feature_cols: List[str],
                               entity_id_col: str, condition_col: str,
                               condition_a: str, condition_b: str) -> pd.DataFrame:
    """
    Paired within-subject comparison between two conditions.
    Returns summary with paired t-test or Wilcoxon results.
    """
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        df_a = df[df[condition_col] == condition_a][[entity_id_col]].copy()
        df_a['val_a'] = numeric[df[condition_col] == condition_a].values
        df_b = df[df[condition_col] == condition_b][[entity_id_col]].copy()
        df_b['val_b'] = numeric[df[condition_col] == condition_b].values

        merged = pd.merge(df_a, df_b, on=entity_id_col).dropna()
        if len(merged) < 5:
            continue

        diff = merged['val_a'] - merged['val_b']
        try:
            _, p = stats.wilcoxon(merged['val_a'], merged['val_b'])
            test = 'Wilcoxon'
        except Exception:
            _, p = stats.ttest_rel(merged['val_a'], merged['val_b'])
            test = 'Paired t-test'

        rows.append({
            'feature': col,
            f'mean_{condition_a}': round(merged['val_a'].mean(), 4),
            f'mean_{condition_b}': round(merged['val_b'].mean(), 4),
            'mean_diff': round(diff.mean(), 4),
            'n_pairs': len(merged),
            'test': test,
            'p_value': round(p, 4),
            'effect_size_d': round(cohens_d(merged['val_a'], merged['val_b']), 4),
        })

    return pd.DataFrame(rows)


# ─── Time Trends / Progression ───────────────────────────────────────────────

def compute_slopes(df: pd.DataFrame, entity_id_col: str, time_col: str,
                   outcome_col: str) -> pd.DataFrame:
    """
    Compute per-subject linear slope of outcome vs time.
    Returns a DataFrame with entity_id, slope, intercept, r2, n_visits.
    """
    results = []
    time_num = pd.to_numeric(df[time_col], errors='coerce')
    outcome_num = pd.to_numeric(df[outcome_col], errors='coerce')
    df_work = df[[entity_id_col]].copy()
    df_work['_t'] = time_num
    df_work['_y'] = outcome_num

    for subject, grp in df_work.groupby(entity_id_col):
        grp = grp.dropna(subset=['_t', '_y'])
        if len(grp) < 3:
            continue
        slope, intercept, r, p, _ = stats.linregress(grp['_t'], grp['_y'])
        results.append({
            entity_id_col: subject,
            'slope': round(slope, 6),
            'intercept': round(intercept, 4),
            'r2': round(r**2, 4),
            'p_value': round(p, 4),
            'n_visits': len(grp),
        })

    return pd.DataFrame(results)


def progression_summary(slopes_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarise slope distribution and identify fast progressors."""
    if slopes_df.empty or 'slope' not in slopes_df.columns:
        return {}
    s = slopes_df['slope']
    q75 = s.quantile(0.75)
    fast = slopes_df[s >= q75]
    return {
        'n_subjects': len(slopes_df),
        'mean_slope': round(s.mean(), 6),
        'median_slope': round(s.median(), 6),
        'sd_slope': round(s.std(), 6),
        'fast_progressor_threshold': round(q75, 6),
        'n_fast_progressors': len(fast),
        'pct_fast': round(100 * len(fast) / len(slopes_df), 1),
    }


# ─── PD Templates ────────────────────────────────────────────────────────────

def mpower_tap_metrics(df: pd.DataFrame, tap_count_col: str = None,
                       duration_col: str = None, timestamps_col: str = None) -> pd.DataFrame:
    """
    Compute tap_rate, variability, and fatigue_slope if possible.
    Works with either: tap timestamps list OR tap_count + duration.
    """
    df = df.copy()
    if tap_count_col and duration_col and tap_count_col in df.columns and duration_col in df.columns:
        tap_count = pd.to_numeric(df[tap_count_col], errors='coerce')
        duration = pd.to_numeric(df[duration_col], errors='coerce')
        df['tap_rate'] = (tap_count / duration.replace(0, np.nan)).round(4)

    if timestamps_col and timestamps_col in df.columns:
        def calc_variability(ts_str):
            try:
                ts = [float(x) for x in str(ts_str).strip('[]').split(',') if x.strip()]
                if len(ts) < 3:
                    return np.nan
                ibi = np.diff(sorted(ts))
                return float(np.std(ibi))
            except Exception:
                return np.nan

        def calc_fatigue(ts_str):
            try:
                ts = sorted([float(x) for x in str(ts_str).strip('[]').split(',') if x.strip()])
                if len(ts) < 6:
                    return np.nan
                ibi = np.diff(ts)
                half = len(ibi) // 2
                first_rate = 1.0 / np.mean(ibi[:half]) if np.mean(ibi[:half]) else np.nan
                second_rate = 1.0 / np.mean(ibi[half:]) if np.mean(ibi[half:]) else np.nan
                return round(second_rate - first_rate, 6) if not np.isnan(first_rate) and not np.isnan(second_rate) else np.nan
            except Exception:
                return np.nan

        df['tap_variability'] = df[timestamps_col].apply(calc_variability).round(6)
        df['tap_fatigue_slope'] = df[timestamps_col].apply(calc_fatigue)

    return df


# ─── Narrative Report ────────────────────────────────────────────────────────

def generate_narrative(
    dataset_name: str,
    shape: tuple,
    qc_summary: Dict,
    descriptive_df: Optional[pd.DataFrame],
    group_comparison_df: Optional[pd.DataFrame],
    group_col: Optional[str],
    progression_summary_dict: Optional[Dict],
    outcome_col: Optional[str],
) -> str:
    """Generate a Markdown narrative report string."""
    lines = []
    lines.append(f"# PD Insight Studio — Analysis Report\n")
    lines.append(f"**Dataset:** {dataset_name}  ")
    lines.append(f"**Shape:** {shape[0]:,} rows × {shape[1]} columns\n")

    lines.append("## 1. Quality Control Summary\n")
    lines.append(f"- Total rows: **{qc_summary.get('total', '?'):,}**")
    lines.append(f"- Rows kept (passed QC): **{qc_summary.get('kept', '?'):,}** ({qc_summary.get('kept_pct', '?')}%)")
    lines.append(f"- Rows flagged: **{qc_summary.get('removed', '?'):,}**")
    if qc_summary.get('reason_counts'):
        lines.append("\n**Top removal reasons:**")
        for reason, cnt in list(qc_summary['reason_counts'].items())[:5]:
            lines.append(f"  - `{reason}`: {cnt} rows")
    if qc_summary.get('high_missingness_cols'):
        lines.append("\n**Columns with >20% missingness:**")
        for col, pct in list(qc_summary['high_missingness_cols'].items())[:8]:
            lines.append(f"  - `{col}`: {pct}% missing")

    if descriptive_df is not None and not descriptive_df.empty:
        lines.append("\n## 2. Descriptive Statistics\n")
        lines.append(descriptive_df.to_markdown(index=False))

    if group_comparison_df is not None and not group_comparison_df.empty and group_col:
        lines.append(f"\n## 3. Group Comparisons — by `{group_col}`\n")
        # Top 5 most different features by effect size
        eff_col = 'effect_size' if 'effect_size' in group_comparison_df.columns else None
        if eff_col:
            top = group_comparison_df.dropna(subset=[eff_col]).nlargest(5, eff_col)
        else:
            top = group_comparison_df.head(5)
        lines.append(top.to_markdown(index=False))
        lines.append("\n> **Interpretation note:** Effect sizes indicate practical difference magnitude.")
        lines.append("> Cohen's d: 0.2=small, 0.5=medium, 0.8=large. Eta-squared: 0.01=small, 0.06=medium, 0.14=large.")

    if progression_summary_dict:
        lines.append(f"\n## 4. Progression Analysis — `{outcome_col}`\n")
        for k, v in progression_summary_dict.items():
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

    lines.append("\n## 5. Limitations\n")
    lines.append("- This analysis is **exploratory and observational**. Associations do not imply causation.")
    lines.append("- Results may be affected by missing data, selection bias, and unmeasured confounders.")
    lines.append("- Predictions (if generated) are for research purposes only and have **no clinical validity**.")
    lines.append("- Any medication-effect analysis reflects group differences; individual responses may vary significantly.")

    return '\n'.join(lines)
