"""CLI to evaluate predictions produced by a model.

Usage:
  python scripts/evaluate_model.py --predictions path/to/predictions.csv

Predictions CSV should contain at least: `user_id`, `video_id`, target columns (`is_like`, `long_view`, ...)
and predicted score columns (`like_score`, `longview_score`, `creator_score`).
"""
import argparse
from pathlib import Path
from src.evaluation.evaluate_predictions import evaluate_predictions
from src.utils.runner import setup_run, handle_exceptions


def main(predictions: str, ks=None, scalar_weights=None, logger=None):
    if ks is None:
        ks = [5, 10, 20]
    res = evaluate_predictions(Path(predictions), ks=ks, scalar_weights=scalar_weights)
    logger.info('Evaluation complete. Summary written to reports/analysis/evaluation_summary.md')


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--predictions', required=True, help='Path to predictions CSV')
        parser.add_argument('--ks', nargs='+', type=int, default=[5,10,20], help='K values for ranking metrics')
        parser.add_argument('--scalar-weights', help='JSON string of weights for scalarization, e.g. "{\"is_like\":1,\"long_view\":1}"')
        args = parser.parse_args()

        scalar_weights = None
        if args.scalar_weights:
            import json
            scalar_weights = json.loads(args.scalar_weights)

        main(args.predictions, ks=args.ks, scalar_weights=scalar_weights, logger=logger)

    _cli()
