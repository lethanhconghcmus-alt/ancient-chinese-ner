#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chuẩn hoá nhãn train/dev theo docs/annotation_guideline.md v1.0.

KHÔNG ĐỘNG VÀO TEST — test đang được duyệt tay qua gold_review_sheet.xlsx.

Ba tầng, áp theo thứ tự:
  T1  Sửa lỗi span hệ thống      (春正 -> 春正月 DTM)
  T2  Áp guideline toàn cục      (miếu hiệu=PER, tước=TITLE, 天下 bỏ, ...)
  T3  Majority-vote cho chuỗi còn mâu thuẫn có đa số >= --maj-threshold

Chuỗi mâu thuẫn mà đa số yếu -> KHÔNG tự sửa, xuất ra needs_review.json.

Majority ở T3 tính TRÊN train+dev, cố ý không dùng test để tránh rò nhãn test
sang train.

Usage:
  py scripts/clean_labels.py --dry-run     # xem tác động, không ghi
  py scripts/clean_labels.py               # ghi đè train/dev + BIO + report
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

TAG = re.compile(r"\{([^|{}]+)\|([A-Z]+)\}")

RAW_DIR = r"D:\ancient-chinese-ner\data\raw\ner_sft"
CLEAN_DIR = r"D:\ancient-chinese-ner\data\processed\ner_clean"
SPLITS = ["train", "dev"]          # test cố ý vắng mặt

# ---------------------------------------------------------------- guideline

# §5 + changelog: 春正 là 春正月 bị cắt
SPAN_FIX = {"春正": ("春正月", "DTM")}

# §3/§4 "KHÔNG tag": nghĩa chung, không phải thực thể
# + §2 "danh từ chung không phải chức vị" (guideline liệt kê đích danh 4 từ này)
DROP = {
    "天下", "國家", "功臣", "将臣", "將臣", "士伍", "群臣",
    # cùng lớp "danh từ chung", phát hiện khi soi dải majority 70-80%.
    # Không thêm được bằng majority-vote vì đa số của chúng đang SAI
    # (vd 文武 bị gán LOC 19/26 lần — "văn võ" không phải địa danh).
    "文武", "朝臣", "權臣", "我國", "各處", "自立", "百姓", "祖宗",
}

# Chức quan 2 ký tự — CHUC_SUFFIX chỉ bắt từ 3 ký tự trở lên nên phải liệt kê
# riêng, nếu không majority-vote sẽ lật ngược (vd 御史 ORG:25/TITLE:5, trong khi
# ORG đúng phải là 御史臺).
TWO_CHAR_TITLE = {
    "御史", "郎中", "尚書", "侍郎", "承旨", "學士", "少卿", "祭酒",
    "司業", "都事", "寺卿", "参政", "參政", "僉事", "知府", "知縣",
}

# §2 tam công / tam thiếu
TAM_CONG = {"太師", "太保", "太傅", "太尉", "少保", "少師", "少傅", "少尉"}

# §2 danh vị tôn xưng đứng riêng (giữ TITLE theo convention đa số)
GENERIC_TITLE = {"帝", "上皇", "太后", "皇后", "太子", "皇帝", "皇太后", "太皇太后"}

# §2 khoa bảng + chức thống lĩnh
KHOA_BANG = {"進士", "同進士出身", "狀元", "榜眼", "探花", "生徒", "太學生"}
THONG_LINH = {"節制", "統領官", "都督", "都統", "提督", "總兵"}

# §3 lục bộ / viện / giám đứng riêng -> ORG
CO_QUAN = {
    "吏部", "户部", "戶部", "禮部", "兵部", "刑部", "工部",
    "翰林院", "國子監", "東閣", "錦衣衛", "御史臺", "都察院",
    "中書省", "樞密院", "宗人府", "朝廷",
}

# §2 chức ghép bộ+chức lấy trọn -> TITLE (áp khi độ dài >= 3)
CHUC_SUFFIX = ("尚書", "侍郎", "御史", "郎中", "員外郎", "承旨", "學士",
               "祭酒", "司業", "少卿", "寺卿", "給事中", "都事")

# §1 miếu hiệu: 2 ký tự kết thúc 宗/祖 -> PER.
# Trừ các từ chung chỉ tổ tiên (không phải miếu hiệu đích danh một vua).
MIEU_HIEU_EXCLUDE = {"祖宗", "先祖", "皇祖", "法祖", "元祖", "列祖", "遠祖", "始祖"}

# §2 tước phong X王/公/侯/伯 -> TITLE. Trừ danh từ chung.
TUOC_SUFFIX = ("王", "公", "侯", "伯")
TUOC_EXCLUDE = {"諸侯", "王公", "公侯", "三公", "諸公", "列侯", "諸王"}


def guideline_type(surface: str):
    """Trả về type mà guideline quy định, hoặc None nếu guideline không phán."""
    if surface in DROP:
        return "DROP"
    if surface in TAM_CONG or surface in GENERIC_TITLE:
        return "TITLE"
    if surface in KHOA_BANG or surface in THONG_LINH:
        return "TITLE"
    if surface in TWO_CHAR_TITLE:
        return "TITLE"
    if surface in CO_QUAN:
        return "ORG"
    # miếu hiệu
    if len(surface) == 2 and surface[-1] in "宗祖" and surface not in MIEU_HIEU_EXCLUDE:
        return "PER"
    # chức ghép bộ + chức
    if len(surface) >= 3 and surface.endswith(CHUC_SUFFIX):
        return "TITLE"
    # tước phong
    if (len(surface) >= 2 and surface.endswith(TUOC_SUFFIX)
            and surface not in TUOC_EXCLUDE):
        return "TITLE"
    return None


# ---------------------------------------------------------------- parse/emit

def parse(output: str):
    """output có markup -> (plain, [(start, end, type), ...])"""
    plain, spans, pos, i = [], [], 0, 0
    while i < len(output):
        m = TAG.match(output, i)
        if m:
            surf, tp = m.group(1), m.group(2)
            plain.append(surf)
            spans.append([pos, pos + len(surf), tp])
            pos += len(surf)
            i = m.end()
        else:
            plain.append(output[i])
            pos += 1
            i += 1
    return "".join(plain), spans


def emit_inline(plain: str, spans):
    out, prev = [], 0
    for st, en, tp in sorted(spans):
        out.append(plain[prev:st])
        out.append("{%s|%s}" % (plain[st:en], tp))
        prev = en
    out.append(plain[prev:])
    return "".join(out)


def emit_bio(plain: str, spans):
    labs = ["O"] * len(plain)
    for st, en, tp in spans:
        labs[st] = "B-" + tp
        for k in range(st + 1, en):
            labs[k] = "I-" + tp
    return labs


# ---------------------------------------------------------------- pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--clean-dir", default=CLEAN_DIR)
    ap.add_argument("--maj-threshold", type=float, default=0.7)
    ap.add_argument("--min-count", type=int, default=5,
                    help="số lần xuất hiện tối thiểu để majority-vote có hiệu lực")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = {}
    for sp in SPLITS:
        recs = []
        with open(os.path.join(args.raw_dir, f"{sp}.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                plain, spans = parse(o["output"])
                assert plain == o["input"], f"{sp}: plain != input"
                recs.append({"o": o, "plain": plain, "spans": spans})
        data[sp] = recs

    changes = Counter()
    detail = defaultdict(Counter)

    # ---- T1: sửa span hệ thống -------------------------------------------
    for sp, recs in data.items():
        for r in recs:
            for s in r["spans"]:
                surf = r["plain"][s[0]:s[1]]
                if surf in SPAN_FIX:
                    want, tp = SPAN_FIX[surf]
                    ext = len(want) - len(surf)
                    if r["plain"][s[0]:s[1] + ext] == want:
                        # không được đè lên span khác
                        clash = any(o is not s and o[0] < s[1] + ext and o[1] > s[1]
                                    for o in r["spans"])
                        if not clash:
                            s[1] += ext
                            s[2] = tp
                            changes["T1_span_fix"] += 1
                            detail["T1"][f"{surf} -> {want}|{tp}"] += 1

    # ---- T2: áp guideline -------------------------------------------------
    for sp, recs in data.items():
        for r in recs:
            keep = []
            for s in r["spans"]:
                surf = r["plain"][s[0]:s[1]]
                want = guideline_type(surf)
                if want == "DROP":
                    changes["T2_drop"] += 1
                    detail["T2_drop"][surf] += 1
                    continue
                if want and want != s[2]:
                    changes["T2_retype"] += 1
                    detail["T2_retype"][f"{surf}: {s[2]} -> {want}"] += 1
                    s[2] = want
                keep.append(s)
            r["spans"] = keep

    # ---- T2b: chuẩn hoá BIÊN ---------------------------------------------
    # Hai lỗi ngược chiều, cả hai đều do guideline quy định:
    #   B1  §2 "cụm ghép bộ+chức lấy trọn"  -> ORG+TITLE liền nhau phải GỘP
    #   B2  §1 "chức + tên đi liền: tách"   -> TITLE+PER gộp làm 1 phải TÁCH
    # Từ điển tham chiếu dựng từ chính data (chỉ lấy chuỗi có nhãn đa số rõ)
    ref = defaultdict(Counter)
    for sp, recs in data.items():
        for r in recs:
            for s in r["spans"]:
                ref[r["plain"][s[0]:s[1]]][s[2]] += 1
    TITLE_REF = {k for k, d in ref.items()
                 if d.most_common(1)[0][0] == "TITLE" and sum(d.values()) >= 3}
    PER_REF = {k for k, d in ref.items()
               if d.most_common(1)[0][0] == "PER" and sum(d.values()) >= 3}

    for sp, recs in data.items():
        for r in recs:
            # --- B1: gộp ORG + TITLE liền nhau thành 1 TITLE
            merged, skip = [], False
            ss = sorted(r["spans"])
            for k in range(len(ss)):
                if skip:
                    skip = False
                    continue
                a = ss[k]
                b = ss[k + 1] if k + 1 < len(ss) else None
                if (b and a[1] == b[0] and a[2] == "ORG" and b[2] == "TITLE"
                        and r["plain"][a[0]:b[1]].endswith(CHUC_SUFFIX)):
                    merged.append([a[0], b[1], "TITLE"])
                    changes["B1_merge_bo_chuc"] += 1
                    detail["B1"][r["plain"][a[0]:b[1]]] += 1
                    skip = True
                else:
                    merged.append(list(a))
            r["spans"] = merged

            # --- B2: tách [chức][tên] gộp làm 1 span
            out = []
            for s in r["spans"]:
                surf = r["plain"][s[0]:s[1]]
                cut = None
                # longest-prefix: chức dài nhất khớp TITLE, phần đuôi là PER.
                # Cả hai vế phải >= 2 ký tự: nới ra 1 ký tự thì danh từ ghép bị
                # xé bừa (帝王 -> 帝|王, vì 王 trùng họ Vương trong PER_REF).
                for n in range(len(surf) - 2, 1, -1):
                    if surf[:n] in TITLE_REF and surf[n:] in PER_REF:
                        cut = n
                        break
                if cut:
                    out.append([s[0], s[0] + cut, "TITLE"])
                    out.append([s[0] + cut, s[1], "PER"])
                    changes["B2_split_chuc_ten"] += 1
                    detail["B2"][f"{surf} -> {surf[:cut]}|{surf[cut:]}"] += 1
                else:
                    out.append(s)
            r["spans"] = out

    # ---- T3: majority-vote (tính trên train+dev sau T1/T2) ---------------
    cnt = defaultdict(Counter)
    for sp, recs in data.items():
        for r in recs:
            for s in r["spans"]:
                cnt[r["plain"][s[0]:s[1]]][s[2]] += 1

    # Ngưỡng 0.7 + sàn số lần xuất hiện: đa số tính trên 3-4 mẫu không đáng tin,
    # đẩy sang review thay vì chốt bừa.
    maj, review = {}, {}
    for surf, d in cnt.items():
        if len(d) < 2:
            continue
        top, n = d.most_common(1)[0]
        tot_s = sum(d.values())
        if n / tot_s >= args.maj_threshold and tot_s >= args.min_count:
            maj[surf] = top
        else:
            review[surf] = dict(d)

    for sp, recs in data.items():
        for r in recs:
            for s in r["spans"]:
                surf = r["plain"][s[0]:s[1]]
                if surf in maj and s[2] != maj[surf]:
                    changes["T3_majority"] += 1
                    detail["T3"][f"{surf}: {s[2]} -> {maj[surf]}"] += 1
                    s[2] = maj[surf]

    # ---- báo cáo ----------------------------------------------------------
    print("=== TÁC ĐỘNG ===")
    for k, v in changes.most_common():
        print(f"  {k:16s} {v:6d}")
    print(f"  {'T3_maj_strings':16s} {len(maj):6d} chuỗi tự sửa được")
    print(f"  {'needs_review':16s} {len(review):6d} chuỗi đa số yếu -> để lại")
    print()
    for tag in ["T1", "T2_drop", "T2_retype", "B1", "B2", "T3"]:
        if detail[tag]:
            print(f"--- {tag} (top 12) ---")
            for k, v in detail[tag].most_common(12):
                print(f"    {v:5d}x  {k}")
            print()

    tot = sum(len(r["spans"]) for recs in data.values() for r in recs)
    print(f"tổng entity sau clean: {tot}")

    if args.dry_run:
        print("\n[dry-run] không ghi file.")
        return

    # ---- ghi --------------------------------------------------------------
    os.makedirs(args.clean_dir, exist_ok=True)
    type_dist = Counter()
    for sp, recs in data.items():
        pj = os.path.join(args.raw_dir, f"{sp}.jsonl")
        cj = os.path.join(args.clean_dir, f"{sp}.jsonl")
        cb = os.path.join(args.clean_dir, f"{sp}.bio.txt")
        with open(pj, "w", encoding="utf-8") as f1, \
             open(cj, "w", encoding="utf-8") as f2, \
             open(cb, "w", encoding="utf-8") as fb:
            for r in recs:
                rec = {
                    "instruction": r["o"]["instruction"],
                    "input": r["plain"],
                    "output": emit_inline(r["plain"], r["spans"]),
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                f1.write(line)
                f2.write(line)
                for c, l in zip(r["plain"], emit_bio(r["plain"], r["spans"])):
                    fb.write(f"{c} {l}\n")
                fb.write("\n")
                for s in r["spans"]:
                    type_dist[s[2]] += 1

    rep = {
        "guideline": "docs/annotation_guideline.md v1.0",
        "splits_cleaned": SPLITS,
        "test_untouched": True,
        "maj_threshold": args.maj_threshold,
        "changes": dict(changes),
        "type_distribution": dict(type_dist),
        "total_entities": sum(type_dist.values()),
        "majority_applied": maj,
        "detail": {k: dict(v) for k, v in detail.items()},
    }
    with open(os.path.join(args.clean_dir, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.clean_dir, "needs_review.json"), "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    print(f"\nĐã ghi {SPLITS} vào {args.raw_dir} + {args.clean_dir}")
    print(f"  clean_report.json  — {sum(changes.values())} thay đổi")
    print(f"  needs_review.json  — {len(review)} chuỗi cần bạn quyết")


if __name__ == "__main__":
    main()
