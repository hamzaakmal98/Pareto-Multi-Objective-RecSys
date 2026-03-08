"""Generate final report assets: figures and final tables for presentation.

Saves outputs under `artifacts/figures/final/` and `artifacts/tables/final/` and
writes an index markdown file describing each asset.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

from src.utils.runner import setup_run, handle_exceptions
from src.utils.io import ensure_dir, write_text
from src.visualization.report_plots import (
    plot_target_distributions,
    plot_training_curves,
    plot_per_task_performance,
    plot_ndcg_comparison,
    plot_weight_sweep,
    plot_pareto_frontier_sample,
    write_final_tables,
)


def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return None


def load_csv(p: Path):
    if p.exists():
        return pd.read_csv(p)
    return None


def main(out_root: str = '.', logger=None):
    out_root = Path(out_root)
    logger.info('Generating report assets under %s', out_root)

    figs_dir = ensure_dir(out_root / 'artifacts' / 'figures' / 'final')
    tabs_dir = ensure_dir(out_root / 'artifacts' / 'tables' / 'final')

    # Load available data
    processed_train = None
    try:
        processed_train = pd.read_parquet(out_root / 'data' / 'processed' / 'train.parquet')
    except Exception:
        try:
            # try CSV
            processed_train = load_csv(out_root / 'data' / 'processed' / 'train.csv')
        except Exception:
            processed_train = None

    # training history
    training_history = load_json(out_root / 'artifacts' / 'training_history.json') or load_json(out_root / 'artifacts' / 'training_history.json')

    # baseline comparison CSV
    baseline_df = load_csv(out_root / 'artifacts' / 'tables' / 'baseline_comparison.csv')

    # ndcg/metric summaries
    metric_summary = load_json(out_root / 'reports' / 'metrics' / 'ranking_metrics.json')
    pareto_metrics = load_json(out_root / 'reports' / 'metrics' / 'pareto_metrics.json')

    # weight sweep
    sweep = None
    try:
        sweep = json.loads((out_root / 'artifacts' / 'tables' / 'baseline_weight_sweep.json').read_text(encoding='utf-8'))
    except Exception:
        sweep = None

    # Pareto frontier sample
    frontier_df = load_csv(out_root / 'artifacts' / 'tables' / 'pareto_frontier.csv')

    generated = {}

    # Figures
    if processed_train is not None:
        p = plot_target_distributions(processed_train, out_root)
        if p:
            generated['target_distributions'] = p

    if training_history:
        hist = training_history if isinstance(training_history, list) else training_history.get('history', training_history)
        p = plot_training_curves(hist, out_root)
        if p:
            generated['training_curves'] = p

    if metric_summary:
        # metric_summary expected to be dict of per-task metrics
        p = plot_per_task_performance(metric_summary, out_root)
        if p:
            generated['per_task_performance'] = p

    if baseline_df is not None:
        p = plot_ndcg_comparison(baseline_df, out_root)
        if p:
            generated['ndcg_comparison'] = p

    if sweep:
        p = plot_weight_sweep(sweep, out_root)
        if p:
            generated['weight_sweep'] = p

    if frontier_df is not None:
        score_cols = [c for c in ['like_score', 'longview_score', 'creator_score'] if c in frontier_df.columns]
        p = plot_pareto_frontier_sample(frontier_df, score_cols, out_root, sample_n=3)
        if p:
            generated['pareto_frontier_sample'] = p

    # Final tables
    model_cfg = load_json(out_root / 'artifacts' / 'models' / 'model_config.json') or {}
    dataset_summary = {}
    try:
        # attempt to synthesize dataset summary
        if processed_train is not None:
            dataset_summary = {'train_rows': int(len(processed_train)), 'columns': processed_train.columns.tolist()}
    except Exception:
        dataset_summary = {}

    final_metrics = metric_summary or {}
    tbl_paths = write_final_tables(out_root, model_cfg=model_cfg, dataset_summary=dataset_summary, baseline_df=baseline_df, final_metrics=final_metrics)

    # write index markdown with captions
    md_lines = ['# Final Report Assets', '']
    md_lines.append('This folder contains presentation-ready figures and final tables produced by the pipeline. Files are stored under `artifacts/figures/final/` and `artifacts/tables/final/`.')
    md_lines.append('')
    md_lines.append('## Figures')
    captions = {
        'target_distributions': 'Target prevalence counts (likes, long views, creator interest proxy).',
        'training_curves': 'Training and validation loss curves over epochs.',
        'per_task_performance': 'Per-task NDCG (preferred K) comparison.',
        'ndcg_comparison': 'NDCG@10 comparison across baseline methods.',
        'weight_sweep': 'Top weight combinations from scalarization sweep (aggregate NDCG).',
        'pareto_frontier_sample': 'Sample Pareto frontier plots showing objective tradeoffs for example users.'
    }
    for k, cap in captions.items():
        if k in generated:
            md_lines.append(f'- **{k}**: {generated[k].name} — {cap}')
    md_lines.append('')
    md_lines.append('## Tables')
    for name, p in tbl_paths.items():
        md_lines.append(f'- **{name}**: {p.name}')

    idx_path = ensure_dir(out_root / 'artifacts' / 'tables') / 'final' / 'report_assets_index.md'
    write_text(idx_path, '\n'.join(md_lines))

    logger.info('Generated assets: %s', ', '.join([p.name for p in generated.values()]))
    logger.info('Saved final tables: %s', ', '.join([str(v.name) for v in tbl_paths.values()]))


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--out-root', default='.')
        args = parser.parse_args()
        main(args.out_root, logger=logger)

    _cli()
