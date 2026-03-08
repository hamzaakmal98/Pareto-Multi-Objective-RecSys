"""Run data preprocessing pipeline (wrapper).

This script calls `src.data.preprocess.run_pipeline` and ensures logging and
output directories are created. Use `--data-root` to point to raw KuaiRand data.
"""
import argparse
from pathlib import Path

from src.utils.runner import setup_run, handle_exceptions
from src.data.preprocess import run_pipeline


def main(data_root: str = None, out_root: str = '.', logger=None):
    logger.info('Starting preprocessing: data_root=%s out_root=%s', data_root, out_root)
    run_pipeline(data_root, out_root)
    logger.info('Preprocessing complete')


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-root', default=None)
        parser.add_argument('--out-root', default='.')
        args = parser.parse_args()
        main(args.data_root, out_root=args.out_root, logger=logger)

    _cli()
