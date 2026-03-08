"""Run Pareto reranking on prediction CSVs.

Usage:
  python scripts/run_pareto_rerank.py --predictions path/to/predictions.csv --top-n 50 --out-root .
"""
import argparse
from pathlib import Path
import json

import pandas as pd

from src.rerank.candidate_generation import union_candidates, top_n_per_objective
from src.rerank.scalarization import scalarize_df
from src.rerank.pareto import extract_frontier
from src.evaluation.ranking import evaluate_ranking
from src.utils.io import ensure_dir, write_json
from src.utils.runner import setup_run, handle_exceptions


def main(predictions: str, top_n: int = 50, weights_json: str = '{}', out_root: str = '.', ks=None, logger=None):
    if ks is None:
        ks = [5, 10, 20]
    preds = pd.read_csv(predictions)
    out_root = Path(out_root)
    artifacts = ensure_dir(out_root / 'artifacts' / 'tables')
    reports = ensure_dir(out_root / 'reports' / 'analysis')

    score_cols = [c for c in ['like_score', 'longview_score', 'creator_score'] if c in preds.columns]
    if not score_cols:
        raise ValueError('No score columns found in predictions')

    logger.info('Generating candidate pool (top-N per objective)')
    candidate_pool = union_candidates(preds, 'user_id', 'video_id', score_cols, top_n)
    candidate_path = artifacts / 'pareto_candidate_pool.csv'
    candidate_pool.to_csv(candidate_path, index=False)
    logger.info('Wrote: %s', candidate_path)

    # scalar baseline
    weights = json.loads(weights_json)
    scalar_df = None
    if weights:
        weight_list = [weights.get('like', 0), weights.get('longview', 0), weights.get('creator', 0)]
        scalar_df = scalarize_df(candidate_pool.copy(), score_cols, weight_list, out_col='scalar_score')
        scalar_out = artifacts / 'scalarized_candidates.csv'
        scalar_df.to_csv(scalar_out, index=False)
        logger.info('Wrote scalarized candidates: %s', scalar_out)

    # Pareto frontier extraction
    logger.info('Extracting Pareto frontier per user')
    frontier = extract_frontier(candidate_pool, score_cols, secondary_score=score_cols[0] if score_cols else None)
    frontier_out = artifacts / 'pareto_frontier.csv'
    frontier.to_csv(frontier_out, index=False)
    logger.info('Wrote frontier: %s', frontier_out)

    # Reranked lists (simple: frontier ordered by secondary score)
    reranked = frontier.copy()
    reranked_path = artifacts / 'pareto_reranked.csv'
    reranked.to_csv(reranked_path, index=False)
    logger.info('Wrote reranked lists: %s', reranked_path)

    # Evaluate NDCG@K for each method where applicable
    logger.info('Evaluating NDCG@K for methods...')
    metrics = {}
    # need original targets in preds; join candidate pools with original preds
    joined = candidate_pool.merge(preds, on=['user_id','video_id'], how='left', suffixes=('','_orig'))
    # single-objective metrics (per score)
    for sc in score_cols:
        m = evaluate_ranking(joined, 'user_id', sc, 'is_like' if 'like' in sc else 'long_view', ks)
        metrics[f'single_{sc}'] = m

    # scalarized evaluation
    if scalar_df is not None:
        scalar_eval_df = scalar_df.merge(preds, on=['user_id','video_id'], how='left')
        scalar_eval = evaluate_ranking(scalar_eval_df, 'user_id', 'scalar_score', 'is_like', ks)
        metrics['scalarized'] = scalar_eval

    # pareto evaluation
    frontier_eval = frontier.merge(preds, on=['user_id','video_id'], how='left')
    pareto_eval = evaluate_ranking(frontier_eval, 'user_id', score_cols[0] if score_cols else score_cols[0], 'is_like', ks)
    metrics['pareto_frontier'] = pareto_eval

    write_json(ensure_dir(out_root / 'reports' / 'metrics') / 'pareto_metrics.json', metrics)
    # write a short markdown report
    md_lines = ['# Pareto Rerank Analysis', '']
    md_lines.append('Candidate pool and frontier saved to artifacts/tables/')
    md_lines.append('')
    md_lines.append('## Summary metrics')
    md_lines.append('')
    for k, v in metrics.items():
        md_lines.append(f'### {k}')
        for sk, sv in v.items():
            md_lines.append(f'- {sk}: {sv:.4f}')
        md_lines.append('')

    write_json(ensure_dir(out_root / 'reports' / 'analysis') / 'pareto_metrics.json', metrics)
    (ensure_dir(out_root / 'reports' / 'analysis') / 'pareto_analysis.md').write_text('\n'.join(md_lines), encoding='utf-8')
    logger.info('Pareto reranking complete. Reports written to reports/analysis/')


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--predictions', required=True)
        parser.add_argument('--top-n', type=int, default=50, help='Top-N candidates per objective')
        parser.add_argument('--weights', help='JSON weights for scalarization e.g. "{\"like\":1,\"longview\":1}"', default='{}')
        parser.add_argument('--out-root', help='Output root', default='.')
        parser.add_argument('--ks', nargs='+', type=int, default=[5,10,20])
        args = parser.parse_args()
        main(args.predictions, top_n=args.top_n, weights_json=args.weights, out_root=args.out_root, ks=args.ks, logger=logger)

    _cli()
