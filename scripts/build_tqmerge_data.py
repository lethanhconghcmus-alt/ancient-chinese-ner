"""Chuan bi data tu TQ-merge (CHisIEC+C-CLUE+CMAG, schema PER/LOC/OFI da gop
ORG->LOC, TITLE/JOB->OFI, BOOK->O — xem D:\\bio_source\\chisiec-ner\\scripts\\
merge_tq_ner_data.py) cho 2 muc dich:

1. Corpus pretrain (raw text, khong nhan) — ghep tu train.txt+dev.txt, sample
   ngang quy mo corpus pretrain DVSKTT (~5000 cau) de thoi gian pretrain 27B
   tuong duong nhau giua cac kich ban.
2. Data SFT dinh dang instruction/input/output (giong data/raw/ner_sft) tu
   train.txt (SFT train) + dev.txt (SFT test), sample ngang quy mo DVSKTT
   (~1284 train / ~160 test) de SFT 27B tren TQ-merge co thoi gian chay
   tuong duong DVSKTT, khong phai train het 112K cau.

Input: thu muc chua train.txt/dev.txt dinh dang CoNLL char\\tlabel (tai ve
tu Kaggle dataset thnhcngl/tq-merge-ner-data).

Cach dung::

    python scripts/build_tqmerge_data.py --src D:/bio_source/tqmerge_dl
"""
import argparse
import json
import os
import random
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTRUCTION = (
    "你是一个专门处理中国古代历史文献的命名实体识别系统。请识别以下文本中的命名实体，"
    "并用{实体|类型}格式标注。\n实体类型：\n- PER: 人名 (人物姓名)\n"
    "- LOC: 地名 (地理位置，含机构/组织)\n- OFI: 官职 (官职称号)\n"
    "只输出标注后的文本，不要解释。"
)


def parse_conll(path):
    """Doc CoNLL char\\tlabel (BIO), tra ve list of (tokens, labels)."""
    sents = []
    cur = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            cur.append((parts[0], parts[1]))
    if cur:
        sents.append(cur)
    return sents


def bio_to_tagged_text(tokens):
    """[(char, 'B-PER'), (char,'I-PER'), ...] -> ('text', 'text{ent|TYPE}...')"""
    plain = "".join(t[0] for t in tokens)
    out = []
    buf = ""
    typ = None
    for ch, tag in tokens:
        if tag == "O":
            if buf:
                out.append(f"{{{buf}|{typ}}}")
                buf, typ = "", None
            out.append(ch)
            continue
        pos, t = tag.split("-")
        if pos == "B":
            if buf:
                out.append(f"{{{buf}|{typ}}}")
            buf, typ = ch, t
        else:  # I
            if typ == t and buf:
                buf += ch
            else:
                if buf:
                    out.append(f"{{{buf}|{typ}}}")
                buf, typ = ch, t
    if buf:
        out.append(f"{{{buf}|{typ}}}")
    return plain, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Thu muc chua train.txt/dev.txt cua tq-merge-ner-data")
    ap.add_argument("--pretrain-n", type=int, default=5000)
    ap.add_argument("--sft-train-n", type=int, default=1284)
    ap.add_argument("--sft-test-n", type=int, default=160)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    train_sents = parse_conll(os.path.join(args.src, "train.txt"))
    dev_sents = parse_conll(os.path.join(args.src, "dev.txt"))
    print(f"Loaded train={len(train_sents)} dev={len(dev_sents)} sentences")

    def ok_len(sents_tokens):
        n = len(sents_tokens)
        return args.min_len <= n <= args.max_len

    # ── 1. Pretrain corpus (raw text, sample tu ca train+dev) ──────────────
    pretrain_dir = os.path.join(REPO_ROOT, "data", "raw", "tqmerge_pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)
    pool = [s for s in (train_sents + dev_sents) if ok_len(s)]
    random.shuffle(pool)
    sample = pool[:args.pretrain_n]
    with open(os.path.join(pretrain_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        for toks in sample:
            f.write("".join(t[0] for t in toks) + "\n")
    print(f"Pretrain corpus: {len(sample)} sentences -> {pretrain_dir}/corpus.txt")

    # ── 2. SFT data (train tu train.txt, test tu dev.txt, khong trung nguon) ──
    sft_dir = os.path.join(REPO_ROOT, "data", "raw", "tqmerge_sft")
    os.makedirs(sft_dir, exist_ok=True)

    def has_entity(toks):
        return any(tag != "O" for _, tag in toks)

    train_pool = [s for s in train_sents if ok_len(s) and has_entity(s)]
    dev_pool = [s for s in dev_sents if ok_len(s) and has_entity(s)]
    random.shuffle(train_pool)
    random.shuffle(dev_pool)

    def write_split(sents, n, path):
        picked = sents[:n]
        with open(path, "w", encoding="utf-8") as f:
            for toks in picked:
                plain, tagged = bio_to_tagged_text(toks)
                rec = {"instruction": INSTRUCTION, "input": plain, "output": tagged}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(picked)

    n_train = write_split(train_pool, args.sft_train_n, os.path.join(sft_dir, "train.jsonl"))
    n_test = write_split(dev_pool, args.sft_test_n, os.path.join(sft_dir, "test.jsonl"))
    print(f"SFT data: train={n_train} test={n_test} -> {sft_dir}")


if __name__ == "__main__":
    main()
