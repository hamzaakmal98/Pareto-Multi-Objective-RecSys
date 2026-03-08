"""Simple entrypoint to run training (placeholder).

This script should be replaced/extended with a config-driven trainer.
"""
import argparse
from src.utils.runner import setup_run, handle_exceptions
from src.models.train import Trainer
from src.config import load_config


def main(config_path: str = None, repo_root=None, logger=None):
    cfg = {}
    if config_path:
        cfg = load_config(config_path)
    # ensure data_root points to repo_root by default
    if repo_root and not cfg.get('data_root'):
        cfg['data_root'] = str(repo_root)
    # propagate top-level training epochs if nested under 'training'
    training_cfg = cfg.get('training', {}) if isinstance(cfg.get('training', {}), dict) else {}
    epochs = training_cfg.get('epochs', cfg.get('epochs', 10))

    trainer = Trainer(cfg)
    logger.info('Starting training')
    trainer.fit(epochs=epochs)


if __name__ == "__main__":
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser(description="Run training")
        parser.add_argument("--config", help="Path to YAML config", default=None)
        args = parser.parse_args()
        main(args.config, repo_root=repo_root, logger=logger)

    _cli()
