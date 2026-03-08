"""High-quality report plotting utilities.

Functions produce presentation-ready PNG figures and return the file paths.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dirs(out_root: Path):
    (out_root / 'artifacts' / 'figures' / 'final').mkdir(parents=True, exist_ok=True)
    (out_root / 'artifacts' / 'tables' / 'final').mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, dpi: int = 200):
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def plot_target_distributions(df: pd.DataFrame, out_root: Path):
    """Bar chart of target prevalences for presentation."""
    _ensure_dirs(out_root)
    targets = [c for c in ['is_like', 'long_view', 'creator_interest_proxy'] if c in df.columns]
    counts = {t: int(df[t].sum()) for t in targets}
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, vals, color=['#2E86AB','#F6C85F','#6FB07F'])
    ax.set_ylabel('Count')
    ax.set_title('Target Prevalence')
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + max(vals)*0.01, f'{int(h):,}', ha='center', va='bottom')

    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'target_distributions.png'
    return _save_fig(fig, out_path)


def plot_training_curves(history: list, out_root: Path):
    """Plot training and validation loss curves from history list of dicts."""
    _ensure_dirs(out_root)
    if not history:
        return None
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df['epoch'], df['train_loss'], label='train_loss', color='#2E86AB')
    ax.plot(df['epoch'], df['val_loss'], label='val_loss', color='#F26419')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'training_curves.png'
    return _save_fig(fig, out_path)


def plot_per_task_performance(metrics_dict: dict, out_root: Path):
    """Plot per-task NDCG or metric summaries. Expects metrics_dict: {task: {metric: value}}."""
    _ensure_dirs(out_root)
    if not metrics_dict:
        return None
    tasks = list(metrics_dict.keys())
    # prefer ndcg@10 if available
    metric_key = None
    for k in ['ndcg@10', 'ndcg@5', 'ndcg@20']:
        if any(k in metrics_dict[t] for t in tasks):
            metric_key = k
            break
    if not metric_key:
        metric_key = next(iter(next(iter(metrics_dict.values())).keys()))

    vals = [metrics_dict[t].get(metric_key, 0.0) for t in tasks]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(tasks, vals, color='#6FB07F')
    ax.set_ylabel(metric_key)
    ax.set_title('Per-task performance')
    for i, v in enumerate(vals):
        ax.text(i, v + 1e-6, f'{v:.3f}', ha='center', va='bottom')
    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'per_task_performance.png'
    return _save_fig(fig, out_path)


def plot_ndcg_comparison(df: pd.DataFrame, out_root: Path):
    """Create a small table-like bar chart comparing NDCG@K across baselines.

    Expects a DataFrame with columns: baseline, task, ndcg@5, ndcg@10, ndcg@20
    """
    _ensure_dirs(out_root)
    if df is None or df.empty:
        return None
    # aggregate by baseline and ndcg@10 if present
    metric_col = 'ndcg@10' if 'ndcg@10' in df.columns else [c for c in df.columns if c.startswith('ndcg@')][0]
    pivot = df.pivot(index='baseline', columns='task', values=metric_col)
    pivot = pivot.fillna(0)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(pivot.index))))
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel(metric_col)
    ax.set_title('NDCG Comparison (@10)')
    ax.legend(title='task')
    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'ndcg_comparison.png'
    return _save_fig(fig, out_path)


def plot_weight_sweep(sweep_dict: dict, out_root: Path):
    """Plot sweep results.

    sweep_dict expected: {weight_tuple_str: {'agg_ndcg@k': value, 'metrics': {...}}}
    """
    _ensure_dirs(out_root)
    if not sweep_dict:
        return None
    # convert keys and values
    rows = []
    for k, v in sweep_dict.items():
        try:
            weights = json.loads(k.replace("'", '"')) if isinstance(k, str) else k
        except Exception:
            weights = eval(k)
        agg = v if isinstance(v, (int, float)) else v.get('agg_ndcg@k', None) or v.get('agg_ndcg', None)
        rows.append({'weights': str(weights), 'agg': agg})
    df = pd.DataFrame(rows)
    df = df.sort_values('agg', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(range(len(df)), df['agg'].values, color='#2E86AB')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['weights'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Aggregate NDCG')
    ax.set_title('Top weight sweep results')
    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'weight_sweep.png'
    return _save_fig(fig, out_path)


def plot_pareto_frontier_sample(frontier_df: pd.DataFrame, score_cols: list, out_root: Path, sample_n: int = 3):
    """Plot sample Pareto frontiers (for a few users) showing tradeoffs between two objectives.

    Picks two primary score columns from score_cols.
    """
    _ensure_dirs(out_root)
    if frontier_df is None or frontier_df.empty or len(score_cols) < 2:
        return None
    s1, s2 = score_cols[0], score_cols[1]
    # select sample users
    users = frontier_df['user_id'].dropna().unique()[:sample_n]
    fig, axes = plt.subplots(sample_n, 1, figsize=(6, 3*sample_n))
    if sample_n == 1:
        axes = [axes]
    for ax, u in zip(axes, users):
        sub = frontier_df[frontier_df['user_id'] == u]
        ax.scatter(sub[s1], sub[s2], c='C0')
        ax.set_xlabel(s1)
        ax.set_ylabel(s2)
        ax.set_title(f'User {u} frontier ({len(sub)} points)')
    out_path = out_root / 'artifacts' / 'figures' / 'final' / 'pareto_frontier_sample.png'
    return _save_fig(fig, out_path)


def write_final_tables(out_root: Path, model_cfg: dict = None, dataset_summary: dict = None, baseline_df: pd.DataFrame = None, final_metrics: dict = None):
    tdir = out_root / 'artifacts' / 'tables' / 'final'
    tdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if model_cfg:
        p = tdir / 'model_configuration.json'
        p.write_text(json.dumps(model_cfg, indent=2), encoding='utf-8')
        paths['model_configuration'] = p
    if dataset_summary:
        p = tdir / 'dataset_summary.json'
        p.write_text(json.dumps(dataset_summary, indent=2), encoding='utf-8')
        paths['dataset_summary'] = p
    if baseline_df is not None:
        p = tdir / 'baseline_comparison.csv'
        baseline_df.to_csv(p, index=False)
        paths['baseline_comparison'] = p
    if final_metrics:
        p = tdir / 'final_ranking_performance.json'
        p.write_text(json.dumps(final_metrics, indent=2), encoding='utf-8')
        paths['final_ranking_performance'] = p
    return paths
