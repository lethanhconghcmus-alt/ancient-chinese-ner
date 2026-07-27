"""Evaluate model trên test set + in error analysis.

Cách dùng::

    python scripts/run_evaluate.py --config configs/e4_sft_1024.yaml \\
        --checkpoint /path/to/sft_best
    python scripts/run_evaluate.py --config configs/e1_zero_shot.yaml --checkpoint base
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.data_utils import load_jsonl
from src.evaluate import evaluate_dataset, print_results
from src.logger import Logger
from src.model_utils import load_model_for_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='File YAML config')
    parser.add_argument('--checkpoint', default=None,
                        help="Checkpoint để eval; 'base' = base model (zero-shot); "
                             'mặc định: <ckpt_dir>/sft_best')
    parser.add_argument('--adapter', default=None,
                        help='LoRA adapter rời (vd checkpoint tải từ Kaggle)')
    parser.add_argument('--split', default='test', choices=['dev', 'test'],
                        help='Split để eval (mặc định: test)')
    parser.add_argument('--baseline-f1', type=float, default=0.82,
                        help='F1 baseline để so sánh (Paper 1: 0.82)')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.paths.resolve_env()
    config.paths.makedirs()

    logger = Logger(os.path.join(config.paths.log_dir, 'evaluate.log'))
    logger.log(f'Experiment: {config.name} | Config: {args.config}')

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(config.paths.ckpt_dir, 'sft_best')
    elif checkpoint == 'base':
        checkpoint = None  # zero-shot: base model thuần
    logger.log(f'Loading model: {checkpoint or config.model.base_model}')
    model, tokenizer = load_model_for_inference(config, checkpoint, args.adapter)

    test_path = os.path.join(config.paths.sft_data_dir, f'{args.split}.jsonl')
    test_data = load_jsonl(test_path)
    logger.log(f'{args.split} records: {len(test_data):,}')

    results = evaluate_dataset(model, tokenizer, test_data, config, logger)
    print_results(results, baseline_f1=args.baseline_f1)


if __name__ == '__main__':
    main()
