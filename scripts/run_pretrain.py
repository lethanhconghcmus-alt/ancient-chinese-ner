"""Chạy continued pretraining trên corpus Hán văn.

Cách dùng::

    python scripts/run_pretrain.py --config configs/e4_sft_1024.yaml
    python scripts/run_pretrain.py --config configs/e4_sft_1024.yaml --resume
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.logger import Logger
from src.model_utils import load_model_for_training
from src.train import pretrain_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='File YAML config')
    parser.add_argument('--resume', action='store_true',
                        help='Resume từ pretrain_final checkpoint nếu có')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.paths.resolve_env()
    config.paths.makedirs()

    logger = Logger(os.path.join(config.paths.log_dir, 'pretrain.log'))
    logger.log(f'Experiment: {config.name} | Config: {args.config}')

    checkpoint = None
    if args.resume:
        candidate = os.path.join(config.paths.ckpt_dir, 'pretrain_final')
        if os.path.exists(candidate):
            checkpoint = candidate
            logger.log(f'Resuming from checkpoint: {checkpoint}')

    model, tokenizer = load_model_for_training(config, checkpoint)
    final_path = pretrain_loop(model, tokenizer, config, logger)
    logger.log(f'Done. Final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
