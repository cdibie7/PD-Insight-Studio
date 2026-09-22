"""
plotting.py
Creates matplotlib figures for embedding in the Tkinter UI and PNG export.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional, Dict
import io

# Style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#4a90d9',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'grid.color': '#2a3a5a',
    'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.titlecolor': '#7ec8e3',
})

PALETTE = ['#4a90d9', '#e85d04', '#7ec8e3', '#f48c06', '#90e0ef', '#d62828']


def make_histogram(df: pd.DataFrame, col: str, group_col: str = None,
                   bins: int = 30, figsize=(7, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    data = pd.to_numeric(df[col], errors='coerce').dropna()

    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        for i, grp in enumerate(groups):
            g_data = pd.to_numeric(df.loc[df[group_col] == grp, col], errors='coerce').dropna()
            ax.hist(g_data, bins=bins, alpha=0.65, label=str(grp),
                    color=PALETTE[i % len(PALETTE)], edgecolor='none')
        ax.legend(framealpha=0.3)
    else:
        ax.hist(data, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor='none')

    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {col}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def make_boxplot(df: pd.DataFrame, feature_cols: List[str], group_col: str = None,
                 figsize=(9, 5)) -> plt.Figure:
    n = len(feature_cols)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')

        if group_col and group_col in df.columns:
            data_by_group = [numeric[df[group_col] == g].dropna()
                             for g in df[group_col].dropna().unique()]
            labels = list(df[group_col].dropna().unique())
            bp = ax.boxplot(data_by_group, labels=labels, patch_artist=True, notch=False)
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ['whiskers', 'caps', 'fliers', 'medians']:
                plt.setp(bp[element], color='#aaaaaa')
            plt.setp(bp['medians'], color='white', linewidth=2)
        else:
            bp = ax.boxplot(numeric.dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor(PALETTE[0])
            bp['boxes'][0].set_alpha(0.7)

        ax.set_title(col, fontsize=9)
        ax.grid(axis='y', alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


def make_bar_chart(counts_df: pd.DataFrame, col: str, figsize=(7, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    # counts_df has columns: col, count, pct
    labels_col = counts_df.columns[0]
    counts = counts_df['count']
    labels = counts_df[labels_col].astype(str)

    bars = ax.barh(labels, counts, color=PALETTE[:len(labels)], alpha=0.85, edgecolor='none')
    ax.set_xlabel('Count')
    ax.set_title(f'Value counts: {col}')
    ax.grid(axis='x', alpha=0.3)

    for bar, cnt, pct in zip(bars, counts, counts_df['pct']):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{cnt} ({pct}%)', va='center', fontsize=8, color='#cccccc')
    fig.tight_layout()
    return fig


def make_group_effect_chart(comparison_df: pd.DataFrame, figsize=(8, 5)) -> plt.Figure:
    """Bar chart of effect sizes for all compared features."""
    if comparison_df.empty or 'effect_size' not in comparison_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No effect size data', ha='center', va='center', transform=ax.transAxes)
        return fig

    df = comparison_df.dropna(subset=['effect_size']).copy()
    df = df.sort_values('effect_size', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in df['effect_size']]
    ax.barh(df['feature'], df['effect_size'].abs(), color=colors, alpha=0.85, edgecolor='none')
    ax.set_xlabel('|Effect Size|')
    ax.set_title("Feature Effect Sizes Across Groups")
    ax.axvline(0.2, linestyle='--', color='#aaaaaa', linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, linestyle='--', color='#ffd700', linewidth=0.8, alpha=0.6)
    ax.axvline(0.8, linestyle='--', color='#ff6b6b', linewidth=0.8, alpha=0.6)
    ax.text(0.21, ax.get_ylim()[1] * 0.98, 'small', fontsize=7, color='#aaaaaa')
    ax.text(0.51, ax.get_ylim()[1] * 0.98, 'medium', fontsize=7, color='#ffd700')
    ax.text(0.81, ax.get_ylim()[1] * 0.98, 'large', fontsize=7, color='#ff6b6b')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return fig


def make_time_trend(df: pd.DataFrame, entity_id_col: str, time_col: str,
                    outcome_col: str, group_col: str = None,
                    max_subjects: int = 30, figsize=(9, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    time_num = pd.to_numeric(df[time_col], errors='coerce')
    outcome_num = pd.to_numeric(df[outcome_col], errors='coerce')

    subjects = df[entity_id_col].dropna().unique()[:max_subjects]
    for subj in subjects:
        mask = df[entity_id_col] == subj
        t = time_num[mask]
        y = outcome_num[mask]
        valid = t.notna() & y.notna()
        if valid.sum() < 2:
            continue

        color = PALETTE[0]
        if group_col and group_col in df.columns:
            g = df.loc[mask & df[group_col].notna(), group_col]
            if not g.empty:
                grps = list(df[group_col].dropna().unique())
                color = PALETTE[grps.index(g.iloc[0]) % len(PALETTE)] if g.iloc[0] in grps else PALETTE[0]

        ax.plot(t[valid], y[valid], color=color, alpha=0.35, linewidth=1.2)

    ax.set_xlabel(time_col)
    ax.set_ylabel(outcome_col)
    ax.set_title(f'{outcome_col} over {time_col} (n={len(subjects)} subjects shown)')
    ax.grid(alpha=0.25)

    if group_col and group_col in df.columns:
        grps = df[group_col].dropna().unique()
        patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=str(g)) for i, g in enumerate(grps)]
        ax.legend(handles=patches, framealpha=0.3)

    fig.tight_layout()
    return fig


def make_slope_distribution(slopes_df: pd.DataFrame, slope_col: str = 'slope',
                              group_data: pd.DataFrame = None, entity_id_col: str = None,
                              group_col: str = None, figsize=(8, 4)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: histogram of slopes
    ax = axes[0]
    slopes = slopes_df[slope_col].dropna()
    ax.hist(slopes, bins=25, color=PALETTE[0], alpha=0.85, edgecolor='none')
    q75 = slopes.quantile(0.75)
    ax.axvline(q75, color=PALETTE[1], linewidth=2, linestyle='--', label=f'Q75 (fast: >{q75:.4f})')
    ax.axvline(slopes.median(), color='white', linewidth=1.5, linestyle=':', label=f'Median')
    ax.set_xlabel('Slope')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Progression Slopes')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(alpha=0.25)

    # Right: slopes by group if available
    ax2 = axes[1]
    if group_data is not None and entity_id_col and group_col:
        merged = pd.merge(slopes_df, group_data[[entity_id_col, group_col]].drop_duplicates(),
                          on=entity_id_col, how='left')
        grps = merged[group_col].dropna().unique()
        data_by_grp = [merged.loc[merged[group_col] == g, slope_col].dropna() for g in grps]
        if any(len(d) > 0 for d in data_by_grp):
            bp = ax2.boxplot(data_by_grp, labels=[str(g) for g in grps], patch_artist=True)
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            plt.setp(bp['medians'], color='white', linewidth=2)
            ax2.set_title('Slopes by Group')
            ax2.grid(axis='y', alpha=0.25)
    else:
        ax2.text(0.5, 0.5, 'No group data', ha='center', va='center', transform=ax2.transAxes, color='#888')

    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert figure to PNG bytes for saving."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def save_fig(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
