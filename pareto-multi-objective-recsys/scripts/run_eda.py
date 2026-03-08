"""Run EDA pipeline for the processed KuaiRand dataset.

Usage (from repo root):
    python scripts/run_eda.py
or
    python scripts/run_eda.py --out-root <path>
"""
from pathlib import Path
import argparse
from src.visualization.eda import run_eda
from src.utils.runner import setup_run, handle_exceptions


def main(out_root: str = '.', logger=None):
    out_root = Path(out_root) if out_root else Path('.')
    logger.info('Running EDA -> out_root=%s', out_root)
    run_eda(out_root)


if __name__ == '__main__':
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument("--out-root", help="Repo root or output root", default='.')
        args = parser.parse_args()
        main(args.out_root, logger=logger)

    _cli()
