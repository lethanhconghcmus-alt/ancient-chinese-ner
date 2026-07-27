"""Evaluation pipeline: batch inference, metrics, error analysis.

Loop chính :func:`evaluate_dataset` hỗ trợ resume (Colab hay chết session
giữa chừng) và auto-save kết quả từng phần về ``result_dir``.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import ExperimentConfig
from .data_utils import compute_prf, make_prompt_prefix, parse_entities
from .logger import Logger

PromptFn = Callable[[Dict[str, Any]], str]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_batch(model: Any, tokenizer: Any,
                   records: Sequence[Dict[str, Any]],
                   config: ExperimentConfig,
                   prompt_fn: Optional[PromptFn] = None) -> List[str]:
    """Generate output cho một batch records.

    Args:
        model: model đã bật inference mode (padding trái).
        tokenizer: tokenizer đi kèm.
        records: list record (dict có ``instruction``/``input``).
        config: dùng ``model.max_seq_len`` + ``eval.max_new_tokens``.
        prompt_fn: hàm build prompt từ record; mặc định prompt SFT chuẩn.
            RAG truyền hàm riêng để chèn few-shot examples.

    Returns:
        List text output (đã strip, bỏ phần prompt).
    """
    import torch  # import tại chỗ để module test được trên máy không có GPU stack

    prompt_fn = prompt_fn or make_prompt_prefix
    prompts = [prompt_fn(r) for r in records]
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        truncation=True,
        max_length=config.model.max_seq_len,
        padding=True,
    ).to(model.device)
    input_len = inputs['input_ids'].shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    return [
        tokenizer.decode(o[input_len:], skip_special_tokens=True).strip()
        for o in outputs
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_texts: Sequence[str], gold_texts: Sequence[str],
                    entity_types: Sequence[str] = ('PER', 'LOC', 'ORG', 'DTM', 'TITLE'),
                    ) -> Dict[str, Any]:
    """Tính P/R/F1 overall + per-type từ cặp text (pred, gold).

    Entity so khớp exact-match trên cặp ``(surface, type)``.

    Returns:
        Dict: ``overall`` / ``per_type`` (mỗi cái có precision/recall/f1)
        và ``counts`` (tp/fp/fn thô để cộng dồn tiếp nếu cần).
    """
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    per_type = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in entity_types}

    for pred_text, gold_text in zip(pred_texts, gold_texts):
        gold_ents = parse_entities(gold_text)
        pred_ents = parse_entities(pred_text)
        overall['tp'] += len(gold_ents & pred_ents)
        overall['fp'] += len(pred_ents - gold_ents)
        overall['fn'] += len(gold_ents - pred_ents)
        for etype in entity_types:
            g = {e for e in gold_ents if e[1] == etype}
            p = {e for e in pred_ents if e[1] == etype}
            per_type[etype]['tp'] += len(g & p)
            per_type[etype]['fp'] += len(p - g)
            per_type[etype]['fn'] += len(g - p)

    return {
        'overall': dict(zip(['precision', 'recall', 'f1'], compute_prf(**overall))),
        'per_type': {
            t: dict(zip(['precision', 'recall', 'f1'], compute_prf(**per_type[t])))
            for t in entity_types
        },
        'counts': {'overall': overall, 'per_type': per_type},
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(predictions: Sequence[Dict[str, Any]],
                   gold_entities: Sequence[Tuple[str, str, int]],
                   missed: Sequence[Tuple[str, str, int]],
                   wrong: Sequence[Tuple[str, str, str]],
                   per_type_counts: Dict[str, Dict[str, int]],
                   ) -> Dict[str, Any]:
    """Phân tích lỗi từ dữ liệu thu thập trong :func:`evaluate_dataset`.

    Args:
        predictions: list dict per-record (có ``sent_len``, ``n_gold``,
            ``n_pred``, ``n_correct``).
        gold_entities: list ``(surface, type, sent_len)`` của mọi gold entity.
        missed: như trên nhưng chỉ các entity bị miss (FN).
        wrong: list ``(surface, pred_type, gold_type)`` — đúng surface sai type.
        per_type_counts: counts tp/fp/fn theo type.

    Returns:
        Dict gồm: ``miss_rate_per_type``, ``type_confusion``, ``top_missed``,
        ``entity_length_miss_rate``, ``sentence_length_f1``.
    """
    # 1. Miss rate per type
    miss_rate = {}
    for etype, c in per_type_counts.items():
        n_gold = c['tp'] + c['fn']
        miss_rate[etype] = {
            'gold': n_gold,
            'missed': c['fn'],
            'miss_pct': c['fn'] / n_gold * 100 if n_gold else 0.0,
        }

    # 2. Type confusion (gold -> pred)
    confusion = Counter((g, p) for _, p, g in wrong)
    type_confusion = [
        {'gold': g, 'pred': p, 'count': c}
        for (g, p), c in confusion.most_common()
    ]

    # 3. Top missed entities
    missed_counter = Counter((s, t) for s, t, _ in missed)
    top_missed = [
        {'surface': s, 'type': t, 'count': c}
        for (s, t), c in missed_counter.most_common(50)
    ]

    # 4. Entity length vs miss rate
    def len_bucket(surface: str) -> str:
        n = len(surface)
        if n <= 2:
            return str(n)
        return '3-4' if n <= 4 else '5+'

    gold_by_len = Counter(len_bucket(s) for s, _, _ in gold_entities)
    missed_by_len = Counter(len_bucket(s) for s, _, _ in missed)
    entity_length = {
        b: {
            'gold': gold_by_len.get(b, 0),
            'missed': missed_by_len.get(b, 0),
            'miss_pct': (missed_by_len.get(b, 0) / gold_by_len[b] * 100
                         if gold_by_len.get(b) else 0.0),
        }
        for b in ['1', '2', '3-4', '5+']
    }

    # 5. Sentence length vs F1
    def sent_bucket(length: int) -> str:
        for limit, name in [(50, '<=50'), (100, '51-100'),
                            (150, '101-150'), (200, '151-200')]:
            if length <= limit:
                return name
        return '200+'

    buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0})
    for pred in predictions:
        b = buckets[sent_bucket(pred['sent_len'])]
        b['tp'] += pred['n_correct']
        b['fp'] += pred['n_pred'] - pred['n_correct']
        b['fn'] += pred['n_gold'] - pred['n_correct']
        b['n'] += 1
    sentence_length = {}
    for name in ['<=50', '51-100', '101-150', '151-200', '200+']:
        if name in buckets:
            b = buckets[name]
            _, _, f1 = compute_prf(b['tp'], b['fp'], b['fn'])
            sentence_length[name] = {'sents': b['n'], 'f1': f1}

    return {
        'miss_rate_per_type': miss_rate,
        'type_confusion': type_confusion,
        'top_missed': top_missed,
        'entity_length_miss_rate': entity_length,
        'sentence_length_f1': sentence_length,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(model: Any, tokenizer: Any,
                     test_data: Sequence[Dict[str, Any]],
                     config: ExperimentConfig, logger: Logger,
                     prompt_fn: Optional[PromptFn] = None,
                     state_name: str = 'eval_state',
                     ckpt_prefix: str = 'eval') -> Dict[str, Any]:
    """Full evaluation loop: batch inference + metrics + error analysis.

    Hỗ trợ resume qua ``<state_name>.json`` (cạnh log) + checkpoint kết quả
    từng phần ``<ckpt_prefix>_checkpoint_<n>.json`` trong ``result_dir``;
    log F1 realtime mỗi batch.

    Args:
        model, tokenizer: model inference mode.
        test_data: list record test.
        config: experiment config.
        logger: Logger của run.
        prompt_fn: hàm build prompt (RAG truyền hàm riêng).
        state_name: tên file state resume.
        ckpt_prefix: prefix file checkpoint kết quả.

    Returns:
        Dict kết quả cuối: metrics + predictions + error_analysis.
    """
    entity_types = list(config.eval.entity_types)
    result_dir = config.paths.result_dir
    batch_size = config.eval.batch_size
    save_every = config.eval.save_every

    # ---- Resume ----
    eval_state = logger.load_state(state_name)
    start_idx = eval_state.get('n_evaluated', 0) if eval_state else 0

    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    per_type = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in entity_types}
    predictions: List[Dict[str, Any]] = []
    gold_entities: List[Tuple[str, str, int]] = []
    missed_entities: List[Tuple[str, str, int]] = []
    wrong_entities: List[Tuple[str, str, str]] = []

    if start_idx > 0:
        prev_path = os.path.join(result_dir, f'{ckpt_prefix}_checkpoint_{start_idx}.json')
        if os.path.exists(prev_path):
            with open(prev_path, encoding='utf-8') as f:
                prev = json.load(f)
            overall = prev['overall_counts']
            per_type = prev['per_type_counts']
            predictions = prev['predictions']
            gold_entities = [(e['surf'], e['type'], e['slen'])
                             for e in prev.get('gold_entities', [])]
            missed_entities = [(e['surf'], e['type'], e['slen'])
                               for e in prev.get('missed_entities', [])]
            wrong_entities = [(e['surf'], e['pred'], e['gold'])
                              for e in prev.get('wrong_entities', [])]
            logger.log(f'Resuming from record {start_idx} (loaded previous results)')
        else:
            start_idx = 0
            logger.log('Checkpoint file not found, starting fresh.')
    if start_idx == 0:
        logger.log('Starting fresh evaluate')

    remaining = list(test_data[start_idx:])
    logger.log(f'Evaluating {len(remaining)} remaining records '
               f'(total {len(test_data)})')

    # ---- Main loop ----
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_preds = generate_batch(model, tokenizer, batch, config, prompt_fn)

        for record, pred_text in zip(batch, batch_preds):
            sent_len = len(record['input'])
            gold_ents = parse_entities(record['output'])
            pred_ents = parse_entities(pred_text)

            overall['tp'] += len(gold_ents & pred_ents)
            overall['fp'] += len(pred_ents - gold_ents)
            overall['fn'] += len(gold_ents - pred_ents)
            for etype in entity_types:
                g = {e for e in gold_ents if e[1] == etype}
                p = {e for e in pred_ents if e[1] == etype}
                per_type[etype]['tp'] += len(g & p)
                per_type[etype]['fp'] += len(p - g)
                per_type[etype]['fn'] += len(g - p)

            gold_entities.extend((s, t, sent_len) for s, t in gold_ents)
            missed_entities.extend((s, t, sent_len) for s, t in gold_ents - pred_ents)
            gold_surf = {s: t for s, t in gold_ents}
            pred_surf = {s: t for s, t in pred_ents}
            for surf in set(gold_surf) & set(pred_surf):
                if gold_surf[surf] != pred_surf[surf]:
                    wrong_entities.append((surf, pred_surf[surf], gold_surf[surf]))

            predictions.append({
                'input': record['input'],
                'gold': record['output'],
                'pred': pred_text,
                'sent_len': sent_len,
                'n_gold': len(gold_ents),
                'n_pred': len(pred_ents),
                'n_correct': len(gold_ents & pred_ents),
            })

        done = start_idx + min(batch_start + batch_size, len(remaining))
        elapsed = time.time() - logger.start
        per_rec = elapsed / max(done - start_idx, 1)
        eta = per_rec * (len(test_data) - done)
        _, _, f1_now = compute_prf(**overall)
        logger.log(f'[{done:>3}/{len(test_data)}] {per_rec:.1f}s/rec | '
                   f'Elapsed: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m | '
                   f'F1: {f1_now:.4f}')

        if done % save_every == 0:
            ckpt_data = {
                'n_evaluated': done,
                'overall_counts': overall,
                'per_type_counts': per_type,
                'predictions': predictions,
                'gold_entities': [{'surf': s, 'type': t, 'slen': l}
                                  for s, t, l in gold_entities],
                'missed_entities': [{'surf': s, 'type': t, 'slen': l}
                                    for s, t, l in missed_entities],
                'wrong_entities': [{'surf': s, 'pred': p, 'gold': g}
                                   for s, p, g in wrong_entities],
            }
            ckpt_path = os.path.join(result_dir,
                                     f'{ckpt_prefix}_checkpoint_{done}.json')
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                json.dump(ckpt_data, f, ensure_ascii=False)
            logger.save_state({'n_evaluated': done}, state_name)
            logger.log(f'Checkpoint saved: {ckpt_path}')

    logger.log(f'Evaluate done! Total: {(time.time() - logger.start) / 60:.1f} min')

    # ---- Final metrics + error analysis ----
    results = {
        'experiment': config.name,
        'overall': dict(zip(['precision', 'recall', 'f1'], compute_prf(**overall))),
        'per_type': {
            t: dict(zip(['precision', 'recall', 'f1'], compute_prf(**per_type[t])))
            for t in entity_types
        },
        'n_evaluated': len(predictions),
        'error_analysis': error_analysis(
            predictions, gold_entities, missed_entities, wrong_entities, per_type),
        'predictions': predictions,
    }
    out_path = os.path.join(result_dir, f'{ckpt_prefix}_final.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.log(f'Final results saved: {out_path}')
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(metrics: Dict[str, Any],
                  baseline_f1: Optional[float] = None) -> None:
    """In bảng P/R/F1 per-type + overall, kèm delta so với baseline nếu có."""
    print('=' * 55)
    print(f"{'Entity':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print('-' * 55)
    for etype, m in metrics['per_type'].items():
        print(f"{etype:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print('-' * 55)
    o = metrics['overall']
    print(f"{'Overall':<10} {o['precision']:>10.4f} {o['recall']:>10.4f} {o['f1']:>10.4f}")
    print('=' * 55)
    print(f"\n>> Overall F1: {o['f1']:.4f}")
    if baseline_f1 is not None:
        print(f'>> Baseline  : {baseline_f1:.4f}')
        print(f">> Delta     : {o['f1'] - baseline_f1:+.4f}")

    ea = metrics.get('error_analysis')
    if not ea:
        return
    print('\n=== Miss Rate per Entity Type ===')
    print(f"{'Type':<10} {'Gold':>8} {'Missed':>8} {'Miss%':>8}")
    for etype, m in ea['miss_rate_per_type'].items():
        print(f"{etype:<10} {m['gold']:>8} {m['missed']:>8} {m['miss_pct']:>7.1f}%")

    print('\n=== Type Confusion (top 10) ===')
    for row in ea['type_confusion'][:10]:
        print(f"{row['gold']:<8} -> {row['pred']:<8} {row['count']:>6}")

    print('\n=== Top 20 Missed Entities ===')
    for row in ea['top_missed'][:20]:
        print(f"{row['surface']:<20} {row['type']:<8} {row['count']:>6}")

    print('\n=== Entity Length vs Miss Rate ===')
    for bucket, m in ea['entity_length_miss_rate'].items():
        print(f"{bucket:<6} Gold:{m['gold']:>6}  Missed:{m['missed']:>6}  "
              f"{m['miss_pct']:>5.1f}%")

    print('\n=== Sentence Length vs F1 ===')
    for bucket, m in ea['sentence_length_f1'].items():
        print(f"{bucket:<12} Sents:{m['sents']:>4}  F1:{m['f1']:.4f}")
