import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dirs(out_dir: Path):
    (out_dir / 'artifacts' / 'figures' / 'eda').mkdir(parents=True, exist_ok=True)
    (out_dir / 'reports' / 'analysis').mkdir(parents=True, exist_ok=True)


def _load_processed(out_root: Path) -> pd.DataFrame:
    p = out_root / 'data' / 'processed'
    # support parquet or CSV processed artifacts
    if (p / 'train.parquet').exists():
        return pd.read_parquet(p / 'train.parquet')
    files = list(p.glob('*.parquet'))
    if files:
        return pd.read_parquet(files[0])
    # fallback to dataset_joined.csv or X.csv + y.csv
    if (p / 'dataset_joined.csv').exists():
        return pd.read_csv(p / 'dataset_joined.csv')
    if (p / 'X.csv').exists():
        X = pd.read_csv(p / 'X.csv')
        y_path = p / 'y.csv'
        if y_path.exists():
            try:
                y = pd.read_csv(y_path)
            except Exception:
                y = None
        else:
            y = None
        if y is not None and not y.empty:
            # align shapes: concat horizontally
            return pd.concat([X, y], axis=1)
        return X
    raise FileNotFoundError(f'No processed parquet or csv files found under {p}')


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 200):
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def plot_target_prevalence(df: pd.DataFrame, targets: List[str], out_path: Path):
    counts = {}
    for t in targets:
        if t in df.columns:
            counts[t] = int(df[t].sum())
        else:
            counts[t] = 0
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, vals, color='C0')
    ax.set_ylabel('Counts')
    ax.set_title('Target Prevalence')
    for i,v in enumerate(vals):
        ax.text(i, v, str(v), ha='center', va='bottom')
    save_fig(fig, out_path)


def plot_user_activity(df: pd.DataFrame, user_col: str, out_path: Path):
    if user_col not in df.columns:
        return
    counts = df[user_col].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(counts.values, bins=50, color='C1')
    ax.set_xlabel('Impressions per user')
    ax.set_ylabel('Number of users')
    ax.set_title('User Activity Distribution')
    save_fig(fig, out_path)


def plot_item_popularity(df: pd.DataFrame, item_col: str, out_path: Path):
    if item_col not in df.columns:
        return
    counts = df[item_col].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(counts.values, bins=50, color='C2')
    ax.set_xlabel('Impressions per item')
    ax.set_ylabel('Number of items')
    ax.set_title('Item Popularity Distribution')
    save_fig(fig, out_path)


def plot_temporal_distribution(df: pd.DataFrame, ts_col: str, out_path: Path):
    if ts_col not in df.columns:
        return
    ts = pd.to_datetime(df[ts_col], errors='coerce')
    if ts.isna().all():
        return
    counts = ts.dt.floor('D').value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(counts.index, counts.values, '-o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Impressions')
    ax.set_title('Temporal Distribution of Interactions')
    fig.autofmt_xdate()
    save_fig(fig, out_path)


def plot_top_categorical(df: pd.DataFrame, cat_cols: List[str], out_path: Path, top_k: int = 10):
    # create one combined figure with subplots
    n = min(len(cat_cols), 4)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols[:n]):
        vc = df[col].fillna('__MISSING__').value_counts().head(top_k)
        ax.barh(vc.index[::-1], vc.values[::-1], color='C3')
        ax.set_title(f'Top {top_k} values: {col}')
    save_fig(fig, out_path)


def plot_missingness_summary(df: pd.DataFrame, out_path: Path):
    null_pct = df.isna().mean() * 100
    null_pct = null_pct.sort_values(ascending=False).head(50)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(null_pct.index[::-1], null_pct.values[::-1], color='C4')
    ax.set_xlabel('Percent missing')
    ax.set_title('Top 50 columns by missingness')
    save_fig(fig, out_path)


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str], out_path: Path):
    cols = [c for c in numeric_cols if c in df.columns]
    n = min(len(cols), 6)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.5*n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols[:n]):
        ser = pd.to_numeric(df[col], errors='coerce').dropna()
        ax.hist(ser.values, bins=50, color='C5')
        ax.set_title(col)
    save_fig(fig, out_path)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], out_path: Path):
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    ax.set_title('Correlation heatmap')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out_path)


def plot_target_interactions(df: pd.DataFrame, t1: str, t2: str, out_path: Path):
    if t1 not in df.columns or t2 not in df.columns:
        return
    x = df[t1].fillna(0)
    y = df[t2].fillna(0)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist2d(x, y, bins=50, cmap='Blues')
    ax.set_xlabel(t1)
    ax.set_ylabel(t2)
    ax.set_title(f'Interaction: {t1} vs {t2}')
    save_fig(fig, out_path)


def run_eda(out_root: Path = None):
    repo_root = Path(__file__).resolve().parents[3]
    out_root = Path(out_root) if out_root else repo_root
    _ensure_dirs(out_root)

    df = _load_processed(out_root)

    fig_dir = out_root / 'artifacts' / 'figures' / 'eda'

    # determine targeted columns using feature registry artifact when available
    candidate_targets = ['is_like', 'long_view', 'is_follow', 'creator_interest_proxy']
    targets = []
    try:
        fr_path = out_root / 'artifacts' / 'feature_metadata' / 'feature_registry.json'
        if fr_path.exists():
            with fr_path.open('r', encoding='utf-8') as f:
                fr = json.load(f)
                # prefer explicit TARGET_COLUMNS from artifact
                targets = [t for t in fr.get('TARGET_COLUMNS', []) if t in df.columns]
    except Exception:
        targets = []
    # fallback to scanning dataframe columns
    if not targets:
        targets = [c for c in candidate_targets if c in df.columns]

    # Try to load feature registry to get numeric/categorical lists
    numeric = []
    categorical = []
    try:
        fr_path = out_root / 'artifacts' / 'feature_metadata' / 'feature_registry.json'
        if fr_path.exists():
            with fr_path.open('r', encoding='utf-8') as f:
                fr = json.load(f)
                numeric = fr.get('NUMERIC_COLUMNS', [])
                categorical = fr.get('CATEGORICAL_COLUMNS', [])
    except Exception:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Generate figures
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Generating EDA figures...')
    plot_target_prevalence(df, targets, fig_dir / 'target_prevalence.png')
    plot_user_activity(df, 'user_id', fig_dir / 'user_activity.png')
    plot_item_popularity(df, 'video_id', fig_dir / 'item_popularity.png')
    # try common temporal columns
    for ts_candidate in ['_impression_ts', 'time_ms', 'date', 'timestamp']:
        if ts_candidate in df.columns:
            plot_temporal_distribution(df, ts_candidate, fig_dir / 'temporal_distribution.png')
            break
    plot_top_categorical(df, categorical, fig_dir / 'top_categorical.png')
    plot_missingness_summary(df, fig_dir / 'missingness_summary.png')
    plot_numeric_histograms(df, numeric, fig_dir / 'numeric_histograms.png')
    plot_correlation_heatmap(df, numeric[:20], fig_dir / 'correlation_heatmap.png')
    # pairwise target interactions
    if len(targets) >= 2:
        for i in range(len(targets)):
            for j in range(i+1, len(targets)):
                t1, t2 = targets[i], targets[j]
                plot_target_interactions(df, t1, t2, fig_dir / f'{t1}_vs_{t2}.png')

    # write markdown summary
    md_lines = ["# EDA Summary", ""]
    md_lines.append("## Key figures")
    md_lines.append("Figures saved to `artifacts/figures/eda/`")
    md_lines.append("")
    # Add short interpretation for each figure
    md_lines.append("## Target Prevalence")
    if targets:
        for t in targets:
            if t in df.columns:
                n_pos = int(df[t].sum())
                n_tot = int(df[t].count())
                md_lines.append(f"- `{t}`: {n_pos}/{n_tot} positive ({n_pos/n_tot:.4%} positive)")
    else:
        md_lines.append("- No target columns detected in processed data.")
    md_lines.append("")
    md_lines.append("## User Activity")
    md_lines.append("- `user_activity.png` shows distribution of impressions per user; heavy right-skew indicates many one-off users and a small set of heavy users.")
    md_lines.append("")
    md_lines.append("## Item Popularity")
    md_lines.append("- `item_popularity.png` shows per-item impression counts; long-tail popularity may require frequency-based handling in modeling.")
    md_lines.append("")
    md_lines.append("## Temporal Distribution")
    md_lines.append("- `temporal_distribution.png` shows interaction volume over time; useful to verify stationarity and choose split strategy.")
    md_lines.append("")
    md_lines.append("## Missingness Summary")
    md_lines.append("- `missingness_summary.png` highlights columns with missing data; consider imputation or feature removal for high-missing columns.")
    md_lines.append("")
    md_lines.append("## Numeric Feature Distributions")
    md_lines.append("- `numeric_histograms.png` and `correlation_heatmap.png` provide quick checks for scaling needs and multicollinearity.")
    md_lines.append("")
    md_lines.append("## Target Interactions")
    if len(targets) >= 2:
        md_lines.append("- Pairwise 2D histograms (`<t1>_vs_<t2>.png`) show joint sparsity and co-occurrence patterns between targets.")
    else:
        md_lines.append("- Not enough targets present to show pairwise interactions.")
    md_lines.append("")
    md_lines.append("## Notes and next steps")
    md_lines.append("- All detected targets are treated as labels and excluded from model inputs by the preprocessing pipeline.")
    md_lines.append("- Extremely imbalanced targets (low positive rate) may require class weighting, sampling, or alternative metrics (NDCG/PR-AUC).")

    write_text = None
    try:
        from src.utils.io import write_text as _w
        write_text = _w
    except Exception:
        def write_text(p, s):
            Path(p).write_text(s, encoding='utf-8')

    write_text(out_root / 'reports' / 'analysis' / 'eda_summary.md', '\n'.join(md_lines))
    logger.info('EDA complete. Reports written.')
