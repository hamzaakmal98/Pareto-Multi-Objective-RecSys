"""Run quick smoke checks across the repository.

Checks performed:
- import key modules
- verify essential script files exist
- verify processed/raw data paths for KuaiRand-Pure are discoverable
- verify outputs directories can be created
"""
from pathlib import Path
import importlib
import sys

from src.utils.runner import setup_run
from src.utils.io import ensure_dir


def check_imports(mods):
    failures = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            failures.append((m, str(e)))
    return failures


def main():
    cfg, repo_root, logger = setup_run()
    logger.info('Running smoke checks...')

    modules = [
        'src.config',
        'src.utils.io',
        'src.utils.runner',
        'src.data.load_raw',
        'src.data.preprocess',
        'src.models.train',
        'src.models.custom_mmoe',
        'src.rerank.pareto',
        'src.evaluation.ranking',
        'src.visualization.report_plots',
    ]

    failures = check_imports(modules)
    if failures:
        logger.error('Import failures:')
        for m, e in failures:
            logger.error(' - %s : %s', m, e)
        sys.exit(2)
    else:
        logger.info('Imports OK')

    # check scripts exist
    scripts = ['scripts/run_preprocess.py', 'scripts/run_train.py', 'scripts/run_eda.py', 'scripts/run_baselines.py', 'scripts/run_pareto_rerank.py', 'scripts/evaluate_model.py', 'scripts/generate_report_assets.py']
    for s in scripts:
        p = repo_root / s
        if not p.exists():
            logger.warning('Missing script: %s', s)
        else:
            logger.info('Found script: %s', s)

    # check KuaiRand-Pure data discovery
    # try to locate KuaiRand-Pure under repo_root and parent
    possible = list(repo_root.rglob('KuaiRand-Pure'))
    found = False
    for p in possible:
        d = p / 'data'
        if d.exists():
            logger.info('Located KuaiRand-Pure data at: %s', d)
            found = True
            break
    if not found:
        # try sibling folder
        parent = repo_root.parent
        for p in parent.rglob('KuaiRand-Pure'):
            d = p / 'data'
            if d.exists():
                logger.info('Located KuaiRand-Pure data at: %s', d)
                found = True
                break
    if not found:
        logger.warning('KuaiRand-Pure data directory not found under repo; some pipelines may fail without raw data. Pass --out-root or --data-root to scripts to point to data.')

    # ensure outputs can be created
    try:
        ensure_dir(repo_root / 'artifacts' / 'tables')
        ensure_dir(repo_root / 'artifacts' / 'figures')
        ensure_dir(repo_root / 'reports' / 'analysis')
        logger.info('Output directories created/available')
    except Exception as e:
        logger.error('Failed to create output directories: %s', e)
        sys.exit(3)

    logger.info('Smoke checks passed (basic).')


if __name__ == '__main__':
    main()
