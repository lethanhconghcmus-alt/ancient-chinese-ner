"""Evaluate với RAG few-shot retrieval (TF-IDF hoặc BGE-M3).

Cách dùng::

    python scripts/run_rag.py --config configs/e3_rag_tfidf.yaml \\
        --checkpoint /path/to/sft_best --retriever tfidf --shots 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.data_utils import load_jsonl
from src.evaluate import print_results
from src.logger import Logger
from src.model_utils import load_model_for_inference
from src.rag import build_retriever, evaluate_with_rag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='File YAML config')
    parser.add_argument('--checkpoint', default=None,
                        help="Checkpoint để eval; 'base' = base model; "
                             'mặc định: <ckpt_dir>/sft_best')
    parser.add_argument('--adapter', default=None,
                        help='LoRA adapter rời (vd checkpoint tải từ Kaggle)')
    parser.add_argument('--retriever', default=None, choices=['tfidf', 'bge'],
                        help='Loại retriever (mặc định: theo config)')
    parser.add_argument('--shots', type=int, default=None,
                        help='Số few-shot examples (mặc định: theo config)')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.paths.resolve_env()
    config.paths.makedirs()
    if args.retriever:
        config.rag.retriever = args.retriever
    if args.shots is not None:
        config.rag.num_shots = args.shots

    logger = Logger(os.path.join(config.paths.log_dir, 'rag.log'))
    logger.log(f'Experiment: {config.name} | retriever={config.rag.retriever} '
               f'| shots={config.rag.num_shots}')

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(config.paths.ckpt_dir, 'sft_best')
    elif checkpoint == 'base':
        checkpoint = None
    logger.log(f'Loading model: {checkpoint or config.model.base_model}')
    model, tokenizer = load_model_for_inference(config, checkpoint, args.adapter)

    train_data = load_jsonl(os.path.join(config.paths.sft_data_dir, 'train.jsonl'))
    test_data = load_jsonl(os.path.join(config.paths.sft_data_dir, 'test.jsonl'))
    logger.log(f'Train (index): {len(train_data):,} | Test: {len(test_data):,}')

    retriever = build_retriever(config.rag.retriever, train_data, config)
    results = evaluate_with_rag(model, tokenizer, retriever, test_data,
                                config, logger)
    print_results(results)


if __name__ == '__main__':
    main()
