"""Minimal prediction wrapper (placeholder).

If your project implements a prediction entrypoint, replace this script to call it.
This helper provides a consistent CLI and logging behavior.
"""
import argparse
from pathlib import Path

from src.utils.runner import setup_run, handle_exceptions


def main(model_path: str = None, data_path: str = None, out_root: str = '.', logger=None):
    logger.info('Prediction step (placeholder).')
    logger.info('Model: %s', model_path)
    logger.info('Data: %s', data_path)
    # Check basic files
    if model_path:
        if not Path(model_path).exists():
            logger.error('Model file not found: %s', model_path)
            raise FileNotFoundError(model_path)
    if data_path:
        if not Path(data_path).exists():
            logger.error('Data file not found: %s', data_path)
            raise FileNotFoundError(data_path)

    logger.info('No prediction implementation found. Implement your scorer in src/models/predict.py and update this wrapper.')


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='Path to model file', default=None)
        parser.add_argument('--data', help='Path to data to score', default=None)
        parser.add_argument('--out-root', default='.')
        args = parser.parse_args()
        main(args.model, args.data, out_root=args.out_root, logger=logger)

    _cli()
