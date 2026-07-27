"""Xử lý data cho pipeline DVSKTT NER.

Gồm: đọc/ghi JSONL, convert BIO -> format ``{entity|TYPE}``, build prompt
SFT, parse entity từ output model, tính P/R/F1, và sample corpus pretrain.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Regex bắt {surface|TYPE} — giữ NGUYÊN pattern của notebook evaluate gốc
# để metric so sánh được giữa các experiment (đổi regex là đổi cách đếm TP/FP)
_ENTITY_RE = re.compile(r'\{([^|]+)\|([^}]+)\}')

# Template prompt Alpaca-style dùng thống nhất cho SFT + eval + RAG
PROMPT_TEMPLATE = (
    '### Instruction:\n{instruction}\n\n'
    '### Input:\n{input}\n\n'
    '### Output:\n'
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Đọc file JSONL thành list dict (bỏ qua dòng trống)."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    """Ghi list dict ra file JSONL (UTF-8, không escape unicode)."""
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# BIO <-> prompt2 format
# ---------------------------------------------------------------------------

def bio_to_prompt2(tokens: Sequence[str], labels: Sequence[str]) -> str:
    """Convert chuỗi token + nhãn BIO sang format ``{entity|TYPE}``.

    Ví dụ::

        bio_to_prompt2(['帝', '崩', '于', '長', '安'],
                       ['O', 'O', 'O', 'B-LOC', 'I-LOC'])
        # -> '帝崩于{長安|LOC}'

    Args:
        tokens: list ký tự/token của câu.
        labels: nhãn BIO tương ứng (``B-X`` / ``I-X`` / ``O``).

    Returns:
        Câu đã gắn tag inline theo format prompt2.

    Raises:
        ValueError: nếu số token và số nhãn không khớp.
    """
    if len(tokens) != len(labels):
        raise ValueError(f'tokens ({len(tokens)}) != labels ({len(labels)})')

    parts: List[str] = []
    ent_tokens: List[str] = []
    ent_type: Optional[str] = None

    def flush() -> None:
        nonlocal ent_tokens, ent_type
        if ent_tokens:
            parts.append('{%s|%s}' % (''.join(ent_tokens), ent_type))
            ent_tokens, ent_type = [], None

    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            flush()
            ent_type = label[2:]
            ent_tokens = [token]
        elif label.startswith('I-') and ent_type == label[2:]:
            ent_tokens.append(token)
        else:
            # 'O' hoặc I- lệch type (BIO không hợp lệ) -> đóng entity đang mở
            flush()
            parts.append(token)
    flush()
    return ''.join(parts)


def parse_entities(text: str) -> Set[Tuple[str, str]]:
    """Extract set ``(surface, type)`` từ text đã gắn tag ``{entity|TYPE}``.

    Dùng cho cả gold output lẫn model prediction; so khớp exact-match
    (surface + type đều phải đúng mới tính TP).
    """
    return set(_ENTITY_RE.findall(text))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def make_prompt_prefix(record: Dict[str, Any]) -> str:
    """Build phần prefix (Instruction + Input + header Output) của prompt.

    Đây là phần model nhìn thấy lúc inference; lúc train phần này bị mask
    khỏi loss (label = -100).
    """
    return PROMPT_TEMPLATE.format(
        instruction=record['instruction'], input=record['input']
    )


def format_sft_prompt(record: Dict[str, Any], tokenizer: Any) -> str:
    """Build full text SFT: prefix + gold output + EOS.

    Args:
        record: dict có key ``instruction``, ``input``, ``output``.
        tokenizer: tokenizer của model (để lấy đúng ``eos_token``).
    """
    return make_prompt_prefix(record) + record['output'] + tokenizer.eos_token


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Tính (precision, recall, f1) từ số đếm TP/FP/FN; chia 0 trả về 0."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Pretrain corpus
# ---------------------------------------------------------------------------

def build_pretrain_corpus(han_path: str,
                          sample_size: int = 5000,
                          dvsktt_size: int = 1311,
                          seed: int = 42,
                          min_len: int = 10) -> List[str]:
    """Sample corpus cho continued pretraining.

    File ``dvsktt_han_merged.txt`` có ``dvsktt_size`` dòng đầu là DVSKTT
    (giữ nguyên 100%), phần còn lại là Hán văn Trung Quốc — sample ngẫu nhiên
    cho đủ ``sample_size`` dòng tổng, rồi shuffle.

    Args:
        han_path: đường dẫn file corpus merged.
        sample_size: tổng số dòng muốn lấy.
        dvsktt_size: số dòng đầu file thuộc DVSKTT.
        seed: seed cho random (reproducible).
        min_len: bỏ các dòng ngắn hơn ngưỡng này.

    Returns:
        List các dòng text đã shuffle.
    """
    rng = random.Random(seed)
    with open(han_path, 'r', encoding='utf-8') as f:
        all_lines = [l.strip() for l in f if len(l.strip()) >= min_len]

    dvsktt_lines = all_lines[:dvsktt_size]
    chinese_lines = all_lines[dvsktt_size:]
    n_chinese = min(max(sample_size - len(dvsktt_lines), 0), len(chinese_lines))
    lines = dvsktt_lines + rng.sample(chinese_lines, n_chinese)
    rng.shuffle(lines)
    return lines
