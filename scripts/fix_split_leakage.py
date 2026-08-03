#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loại leakage chồng lấn giữa các split.

Bối cảnh: dedup của rebuild_dataset.py là dedup TUYỆT ĐỐI theo chuỗi, nên bắt
được record trùng nguyên văn nhưng KHÔNG bắt được record chỉ *chồng lấn một
phần*. map_records_to_source.py phát hiện 12 câu gốc (chỉ xuất hiện 1 lần trong
nguyên bản ĐVSKTT) lại nằm ở hai split — tức cùng một đoạn văn bị đưa vào hai
record khác nhau.

Cách xử lý: bỏ record bên TRAIN, giữ nguyên dev/test.
  - dev/test giữ đúng 160 record => so sánh được với kết quả đã báo cáo
  - mất ~0.5% train, không đáng kể

Phân biệt trùng THẬT với câu công thức: câu như 冬十月會試天下士人 lặp lại nhiều
lần trong nguyên bản (văn sử dùng công thức cố định), trùng giữa các split là
bình thường. Chỉ tính là leakage khi chuỗi xuất hiện ĐÚNG 1 LẦN trong toàn corpus
gốc mà lại có mặt ở nhiều split.

Usage:
  py scripts/fix_split_leakage.py --dry-run
  py scripts/fix_split_leakage.py
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

CRAWL = r"D:\ancient-chinese-ner\data\raw\gazetteer\dvsktt_sentences.jsonl"
RAW = r"D:\ancient-chinese-ner\data\raw\ner_sft"
CLEAN = r"D:\ancient-chinese-ner\data\processed\ner_clean"
DUP_SECTIONS = {
    "80-Phu-Mac-Hau-Hop", "77-Phu-Mac-Phuc-Nguyen",
    "71-Phu-Mac-Dang-Doanh-Mac-Phuc-Nguyen", "75-Phu-Mac-Phuc-Nguyen",
}
TAG = re.compile(r"\{([^|{}]+)\|([A-Z]+)\}")


def emit_bio_lines(rec):
    plain, spans, pos, i, out = [], [], 0, 0, rec["output"]
    while i < len(out):
        m = TAG.match(out, i)
        if m:
            plain.append(m.group(1))
            spans.append((pos, pos + len(m.group(1)), m.group(2)))
            pos += len(m.group(1))
            i = m.end()
        else:
            plain.append(out[i]); pos += 1; i += 1
    plain = "".join(plain)
    labs = ["O"] * len(plain)
    for st, en, tp in spans:
        labs[st] = "B-" + tp
        for k in range(st + 1, en):
            labs[k] = "I-" + tp
    return [f"{c} {l}" for c, l in zip(plain, labs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=10,
                    help="chỉ xét câu gốc dài >= ngần này ký tự")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # câu gốc + tần suất trong nguyên bản
    freq, sents = Counter(), []
    with open(CRAWL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            freq[s["han"]] += 1
            if s["section"] not in DUP_SECTIONS and len(s["han"]) >= args.min_len:
                sents.append(s)

    data = {}
    for sp in ["train", "dev", "test"]:
        with open(os.path.join(RAW, f"{sp}.jsonl"), encoding="utf-8") as f:
            data[sp] = [json.loads(l) for l in f]

    # câu gốc DUY NHẤT trong nguyên bản mà nằm ở nhiều split
    real, owners = [], {}
    for s in sents:
        h = s["han"]
        if freq[h] != 1:
            continue                      # câu công thức, lặp lại thật -> bỏ qua
        hit = defaultdict(list)
        for sp, recs in data.items():
            for i, r in enumerate(recs):
                if h in r["input"]:
                    hit[sp].append(i)
        if len(hit) > 1:
            real.append((s, dict(hit)))
            owners[h] = dict(hit)

    print(f"câu gốc DUY NHẤT nằm ở nhiều split: {len(real)}")
    drop = set()
    for s, hit in real:
        mark = ""
        if "train" in hit and len(hit) > 1:
            drop.update(hit["train"])
            mark = f"  -> bỏ train {hit['train']}"
        print(f"  {len(s['han']):3d} ký tự [{s['section']}] "
              f"{ {k: v for k, v in hit.items()} }{mark}")
        print(f"       {s['han'][:56]}")

    print()
    print(f"=> bỏ {len(drop)} record TRAIN: {sorted(drop)}")
    print(f"   train {len(data['train'])} -> {len(data['train']) - len(drop)}")
    print(f"   dev {len(data['dev'])} · test {len(data['test'])} giữ nguyên")

    if args.dry_run:
        print("\n[dry-run] không ghi file.")
        return

    kept = [r for i, r in enumerate(data["train"]) if i not in drop]
    for d in (RAW, CLEAN):
        with open(os.path.join(d, "train.jsonl"), "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(CLEAN, "train.bio.txt"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write("\n".join(emit_bio_lines(r)) + "\n\n")

    n_ent = sum(len(TAG.findall(r["output"])) for r in kept)
    rep = {
        "dropped_train_idx": sorted(drop),
        "n_dropped": len(drop),
        "train_after": len(kept),
        "entities_after": n_ent,
        "cross_split_unique_sentences": [
            {"han": s["han"], "section": s["section"],
             "position": s["position"], "hits": hit} for s, hit in real],
    }
    with open(os.path.join(CLEAN, "leakage_fix_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(f"\nĐã ghi: train {len(kept)} record · {n_ent} entity")
    print(f"  {os.path.join(CLEAN, 'leakage_fix_report.json')}")


if __name__ == "__main__":
    main()
