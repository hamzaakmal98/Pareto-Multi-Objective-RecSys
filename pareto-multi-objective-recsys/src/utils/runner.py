"""Helpers to initialize logging, config, and safe script execution."""
from pathlib import Path
import argparse
import logging
import sys
import traceback
import warnings
from logging.handlers import RotatingFileHandler

from src.utils.io import ensure_dir
from src.config import load_config


def _silence_noisy_libs():
    # Silence common noisy warnings that are safe to ignore in experiments
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # Reduce logging from noisy libraries
    for name in ['matplotlib', 'numba', 'urllib3', 'botocore']:
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_run(argv=None):
    """Parse standard args, setup logging, load config, and return (cfg, out_root, logger).

    Standard CLI flags supported: --config, --out-root, --log-level, --run-name
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=None)
    parser.add_argument('--out-root', default='.')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--run-name', default=None)
    # allow scripts to accept extra args
    known, _ = parser.parse_known_args(argv)

    cfg = {}
    try:
        cfg = load_config(known.config)
    except Exception:
        # don't fail here; scripts may not require a config file
        cfg = {}

    out_root = Path(known.out_root)
    ensure_dir(out_root)
    log_dir = ensure_dir(out_root / 'logs')

    # configure logging
    logger = logging.getLogger(known.run_name or 'kuairand')
    logger.setLevel(logging.DEBUG)
    # console handler (high-level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, known.log_level.upper(), logging.INFO))
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    # remove existing handlers to avoid duplicate logs
    logger.handlers = []
    logger.addHandler(ch)

    # file handler (detailed)
    fh = RotatingFileHandler(log_dir / f'{known.run_name or "run"}.log', maxBytes=10_000_00, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(module)s:%(lineno)d - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    _silence_noisy_libs()

    logger.debug('Logger initialized')
    return cfg, out_root, logger


def handle_exceptions(logger, out_root: Path):
    """Return a decorator that wraps `main()` functions to catch exceptions and write tracebacks to file."""

    def _decorator(func):
        def _wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception('Unhandled exception: %s', e)
                # write traceback to file for debugging
                tb_path = ensure_dir(out_root / 'logs') / 'last_traceback.txt'
                with tb_path.open('w', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
                logger.error('Traceback written to %s', tb_path)
                sys.exit(1)
        return _wrapped

    return _decorator
