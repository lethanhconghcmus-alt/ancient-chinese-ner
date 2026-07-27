"""Chạy SFT trên data NER, khởi đầu từ pretrain checkpoint.

Cách dùng::

    python scripts/run_sft.py --config configs/e4_sft_1024.yaml
    python scripts/run_sft.py --config configs/e4_sft_1024.yaml --resume
    python scripts/run_sft.py --config configs/e4_sft_1024.yaml --from-checkpoint path/to/ckpt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.logger import Logger
from src.model_utils import load_model_for_training
from src.train import sft_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='File YAML config')
    parser.add_argument('--resume', action='store_true',
                        help='Resume theo sft_state.json (checkpoint gần nhất)')
    parser.add_argument('--from-checkpoint', default=None,
                        help='Checkpoint khởi đầu (mặc định: pretrain_final nếu có)')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.paths.resolve_env()
    config.paths.makedirs()

    logger = Logger(os.path.join(config.paths.log_dir, 'sft.log'))
    logger.log(f'Experiment: {config.name} | Config: {args.config}')

    checkpoint = args.from_checkpoint
    if checkpoint is None and args.resume:
        state = logger.load_state('sft_state')
        if state and state.get('ckpt') and os.path.exists(state['ckpt']):
            checkpoint = state['ckpt']
    if checkpoint is None:
        candidate = os.path.join(config.paths.ckpt_dir, 'pretrain_final')
        if os.path.exists(candidate):
            checkpoint = candidate
        else:
            logger.log('Pretrain checkpoint not found — starting from base model.')

    logger.log(f'Init model from: {checkpoint or config.model.base_model}')
    model, tokenizer = load_model_for_training(config, checkpoint)
    best_path = sft_loop(model, tokenizer, config, logger)
    logger.log(f'Done. Best checkpoint: {best_path}')


if __name__ == '__main__':
    main()
