"""Run baseline recommenders and produce comparison report.

Usage:
  python scripts/run_baselines.py --predictions path/to/predictions.csv --out-root .
"""
import argparse
import json
from pathlib import Path
import itertools
import pandas as pd

from src.models.baselines import timestamp_baseline, score_baseline, weighted_scalar_baseline, sweep_weights, evaluate_baseline
from src.utils.io import ensure_dir
from src.utils.runner import setup_run, handle_exceptions


def generate_weight_grid(values=[0.0, 0.5, 1.0]):
    grid = []
    for w in itertools.product(values, repeat=3):
        grid.append(list(w))
    return grid


def main(predictions: str, out_root: str = '.', ks=None, logger=None):
    if ks is None:
        ks = [5, 10, 20]
    preds = pd.read_csv(predictions)
    out_root = Path(out_root)
    artifacts = ensure_dir(out_root / 'artifacts' / 'tables')
    reports = ensure_dir(out_root / 'reports' / 'analysis')

    # Baselines
    summaries = []

    logger.info('Running timestamp baseline...')
    ts_df = timestamp_baseline(preds)
    ts_path = artifacts / 'baseline_timestamp_ranked.csv'
    ts_df.to_csv(ts_path, index=False)
    ts_metrics = evaluate_baseline(preds, ts_df, ks)
    summaries.append(('timestamp', ts_metrics))

    # single-objective baselines
    for sc, label, target in [('like_score', 'like', 'is_like'), ('longview_score', 'longview', 'long_view'), ('creator_score', 'creator', 'creator_interest_proxy')]:
        if sc in preds.columns:
            logger.info('Running %s-only baseline...', label)
            df_ranked = score_baseline(preds, sc)
            out_path = artifacts / f'baseline_{label}_ranked.csv'
            df_ranked.to_csv(out_path, index=False)
            metrics = evaluate_baseline(preds, df_ranked, ks)
            summaries.append((label, metrics))

    # weighted scalar baseline: default weights equal
    score_cols = [c for c in ['like_score','longview_score','creator_score'] if c in preds.columns]
    if score_cols:
        base_weights = [1.0] * len(score_cols)
        logger.info('Running weighted scalar baseline (equal weights)...')
        scalar_df = weighted_scalar_baseline(preds, score_cols, base_weights)
        scalar_path = artifacts / 'baseline_scalar_equal_ranked.csv'
        scalar_df.to_csv(scalar_path, index=False)
        scalar_metrics = evaluate_baseline(preds, scalar_df, ks)
        summaries.append(('scalar_equal', scalar_metrics))

    # sweep weights
    logger.info('Sweeping weights for scalar baseline (coarse grid)...')
    grid = generate_weight_grid(values=[0.0,0.5,1.0])
    sweep_results, best_weights = sweep_weights(preds, score_cols, grid, ks)
    # save sweep summary
    sweep_path = artifacts / 'baseline_weight_sweep.json'
    sweep_path.write_text(json.dumps({str(k): v['agg_ndcg@k'] for k, v in sweep_results.items()}), encoding='utf-8')
    logger.info('Best normalized weights: %s', best_weights)
    # record best
    if best_weights:
        best_df = weighted_scalar_baseline(preds, score_cols, list(best_weights))
        best_df.to_csv(artifacts / 'baseline_scalar_best_ranked.csv', index=False)
        best_metrics = sweep_results[tuple(best_weights)]['metrics']
        summaries.append(('scalar_best', best_metrics))

    # write summary markdown and CSV
    md_lines = ['# Baseline Comparison', '']
    rows = []
    for name, metrics in summaries:
        md_lines.append(f'## {name}')
        if not metrics:
            md_lines.append('- No metrics available')
            continue
        for task, m in metrics.items():
            # format ndcg@k entries
            ndcg_entries = {k:v for k,v in m.items() if k.startswith('ndcg@')}
            md_lines.append(f'- Task: {task}')
            for k, v in ndcg_entries.items():
                md_lines.append(f'  - {k}: {v:.4f}')
            # add row to CSV summary
            row = {'baseline': name, 'task': task}
            row.update({k: v for k, v in m.items()})
            rows.append(row)
        md_lines.append('')

    df_summary = pd.DataFrame(rows)
    csv_path = artifacts / 'baseline_comparison.csv'
    df_summary.to_csv(csv_path, index=False)
    (reports / 'baseline_comparison.md').write_text('\n'.join(md_lines), encoding='utf-8')

    logger.info('Baselines complete. Summary:')
    logger.info(' - CSV: %s', csv_path)
    logger.info(' - MD: %s', reports / 'baseline_comparison.md')


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--predictions', required=True)
        parser.add_argument('--out-root', default='.')
        parser.add_argument('--ks', nargs='+', type=int, default=[5,10,20])
        args = parser.parse_args()
        main(args.predictions, out_root=args.out_root, ks=args.ks, logger=logger)

    _cli()
