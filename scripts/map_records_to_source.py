#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Map từng record v2 về vị trí gốc trong DVSKTT (nomfoundation crawl).

Text trong Excel anh Thiều được copy từ nomfoundation.org — xác nhận bằng dấu
vết `mat-chu` và folio marker `[8a*02*02]` xuất hiện y hệt ở cả hai nguồn.
Nhờ đó khớp ngược được record -> (section, position).

Dùng để:
  1. Biết record thuộc quyển/đời nào  -> phân tích lỗi theo thời kỳ
  2. Phát hiện leakage kiểu chồng lấn  -> hai split cùng chứa một câu gốc
  3. Xuất danh sách câu crawl thuộc test/dev -> LOẠI khi mine gazetteer/pretrain

Output (--out-dir):
  record_source_map.json   record -> section, position, số câu khớp
  source_split_map.json    câu crawl -> split nào đang dùng
  gazetteer_blocklist.txt  câu crawl PHẢI loại khi mine (thuộc dev/test)

Usage:
  py scripts/map_records_to_source.py
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

CRAWL = r"D:\ancient-chinese-ner\data\raw\gazetteer\dvsktt_sentences.jsonl"
SFT = r"D:\ancient-chinese-ner\data\raw\ner_sft"
OUT = r"D:\ancient-chinese-ner\data\processed\ner_clean"
SPLITS = ["train", "dev", "test"]

# 4 section "phụ Mạc" trùng 100% với kỷ nhà Lê — site phục vụ cùng nội dung
# dưới hai slug. Bỏ để không đếm hai lần.
DUP_SECTIONS = {
    "80-Phu-Mac-Hau-Hop",
    "77-Phu-Mac-Phuc-Nguyen",
    "71-Phu-Mac-Dang-Doanh-Mac-Phuc-Nguyen",
    "75-Phu-Mac-Phuc-Nguyen",
}
MIN_LEN = 8          # câu ngắn hơn dễ khớp trùng ngẫu nhiên


def sec_num(s):
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else 999


def pos_key(p):
    """'16a*1*1' -> (16, 'a', 1, 1) để sắp thứ tự trong nguyên bản."""
    m = re.match(r"(\d+)([ab]?)\*(\d+)\*(\d+)", p or "")
    if not m:
        return (9999, "", 0, 0)
    return (int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default=CRAWL)
    ap.add_argument("--sft-dir", default=SFT)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--min-len", type=int, default=MIN_LEN)
    args = ap.parse_args()

    # ---- nạp câu crawl ----------------------------------------------------
    sents = []
    with open(args.crawl, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["section"] in DUP_SECTIONS or len(s["han"]) < args.min_len:
                continue
            sents.append(s)
    print(f"câu crawl dùng để khớp: {len(sents)}")

    # ---- nạp record, ghép thành 1 chuỗi lớn + bảng offset ----------------
    recs, parts, offs, cur = [], [], [], 0
    SEP = "\u0000"
    for sp in SPLITS:
        with open(os.path.join(args.sft_dir, f"{sp}.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                t = json.loads(line)["input"]
                recs.append({"split": sp, "idx": i, "len": len(t),
                             "sections": Counter(), "positions": [], "n_sent": 0})
                parts.append(t)
                offs.append((cur, cur + len(t), len(recs) - 1))
                cur += len(t) + 1
    big = SEP.join(parts)
    print(f"record: {len(recs)} | tổng {sum(r['len'] for r in recs)} ký tự")

    # offset -> record, tra bằng chặt nhị phân
    import bisect
    starts = [o[0] for o in offs]

    def which(pos):
        k = bisect.bisect_right(starts, pos) - 1
        return offs[k][2] if k >= 0 and pos < offs[k][1] else None

    # ---- khớp từng câu crawl vào chuỗi lớn -------------------------------
    sent_hits = {}
    for s in sents:
        h, found = s["han"], []
        p = big.find(h)
        while p != -1:
            r = which(p)
            if r is not None:
                found.append(r)
            p = big.find(h, p + 1)
        if found:
            sent_hits[(s["section"], s["position"])] = (s, sorted(set(found)))
            for r in set(found):
                recs[r]["sections"][s["section"]] += 1
                recs[r]["positions"].append(pos_key(s["position"]))
                recs[r]["n_sent"] += 1

    print(f"câu crawl khớp được vào record: {len(sent_hits)}/{len(sents)}"
          f" ({len(sent_hits)/len(sents)*100:.1f}%)")

    # ---- gán section/position cho record ---------------------------------
    unmapped = defaultdict(int)
    for r in recs:
        if not r["sections"]:
            unmapped[r["split"]] += 1
            r["section"] = None
            r["pos_from"] = r["pos_to"] = None
            continue
        r["section"] = r["sections"].most_common(1)[0][0]
        ps = sorted(r["positions"])
        r["pos_from"], r["pos_to"] = ps[0], ps[-1]

    print()
    print("=== ĐỘ PHỦ MAP ===")
    for sp in SPLITS:
        rs = [r for r in recs if r["split"] == sp]
        ok = [r for r in rs if r["section"]]
        print(f"  {sp:6s} {len(ok):5d}/{len(rs):5d} record map được"
              f" ({len(ok)/len(rs)*100:5.1f}%) | chưa map: {unmapped[sp]}")

    # ---- LEAKAGE: câu gốc bị dùng ở nhiều split --------------------------
    cross = defaultdict(list)
    for (sec, pos), (s, rlist) in sent_hits.items():
        sps = {recs[r]["split"] for r in rlist}
        if len(sps) > 1:
            cross[frozenset(sps)].append((sec, pos, s["han"][:40], sorted(sps)))

    print()
    print("=== LEAKAGE: câu gốc xuất hiện ở NHIỀU split ===")
    if not cross:
        print("  không có — các split không chia sẻ câu gốc nào")
    for k, v in sorted(cross.items(), key=lambda x: -len(x[1])):
        print(f"  {'+'.join(sorted(k)):18s} {len(v):5d} câu")
        for sec, pos, h, _ in v[:3]:
            print(f"       [{sec} {pos}] {h}")

    # ---- phân bố section theo split --------------------------------------
    print()
    print("=== PHÂN BỐ SECTION THEO SPLIT (top 12 section của test) ===")
    bysec = defaultdict(Counter)
    for r in recs:
        if r["section"]:
            bysec[r["section"]][r["split"]] += 1
    test_secs = sorted([s for s in bysec if bysec[s]["test"]],
                       key=lambda s: -bysec[s]["test"])
    print(f"  {'section':36s} {'train':>6s} {'dev':>5s} {'test':>5s}")
    for s in test_secs[:12]:
        c = bysec[s]
        print(f"  {s:36s} {c['train']:6d} {c['dev']:5d} {c['test']:5d}")
    print(f"  section có mặt trong test: {len(test_secs)}")

    # ---- ghi ---------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    rec_map = [{
        "split": r["split"], "idx": r["idx"], "section": r["section"],
        "sec_num": sec_num(r["section"]) if r["section"] else None,
        "pos_from": "*".join(map(str, r["pos_from"])) if r["pos_from"] else None,
        "pos_to": "*".join(map(str, r["pos_to"])) if r["pos_to"] else None,
        "n_source_sent": r["n_sent"], "n_chars": r["len"],
    } for r in recs]
    with open(os.path.join(args.out_dir, "record_source_map.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec_map, f, ensure_ascii=False, indent=2)

    split_map = {}
    for (sec, pos), (s, rlist) in sent_hits.items():
        split_map[f"{sec}|{pos}"] = sorted({recs[r]["split"] for r in rlist})
    with open(os.path.join(args.out_dir, "source_split_map.json"), "w",
              encoding="utf-8") as f:
        json.dump(split_map, f, ensure_ascii=False, indent=2)

    # blocklist: câu gốc chạm dev/test -> cấm dùng khi mine gazetteer/pretrain
    block = [s["han"] for (sec, pos), (s, rlist) in sent_hits.items()
             if {recs[r]["split"] for r in rlist} & {"dev", "test"}]
    with open(os.path.join(args.out_dir, "gazetteer_blocklist.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(block))

    n_free = len(sents) - len(block)
    print()
    print(f"đã ghi vào {args.out_dir}:")
    print(f"  record_source_map.json    {len(rec_map)} record")
    print(f"  source_split_map.json     {len(split_map)} câu gốc đã map")
    print(f"  gazetteer_blocklist.txt   {len(block)} câu CẤM (chạm dev/test)")
    print(f"  -> còn {n_free}/{len(sents)} câu crawl dùng được để mine"
          f" ({n_free/len(sents)*100:.1f}%)")


if __name__ == "__main__":
    main()
