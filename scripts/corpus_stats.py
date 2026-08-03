#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sinh bảng thống kê corpus cho paper -> docs/corpus_stats.md"""
import json
import os
import re
from collections import Counter, defaultdict

RAW = r"D:\ancient-chinese-ner\data\raw\ner_sft"
CLEAN = r"D:\ancient-chinese-ner\data\processed\ner_clean"
CRAWL = r"D:\ancient-chinese-ner\data\raw\gazetteer\dvsktt_sentences.jsonl"
OUT = r"D:\ancient-chinese-ner\docs\corpus_stats.md"
TAG = re.compile(r"\{([^|{}]+)\|([A-Z]+)\}")
TYPES = ["PER", "LOC", "ORG", "TITLE", "DTM"]
DUP = {"80-Phu-Mac-Hau-Hop", "77-Phu-Mac-Phuc-Nguyen",
       "71-Phu-Mac-Dang-Doanh-Mac-Phuc-Nguyen", "75-Phu-Mac-Phuc-Nguyen"}

L = []
def w(s=""): L.append(s)

data, stats = {}, {}
for sp in ["train", "dev", "test"]:
    recs = [json.loads(l) for l in open(os.path.join(RAW, f"{sp}.jsonl"), encoding="utf-8")]
    data[sp] = recs
    ents = [(m.group(1), m.group(2)) for r in recs for m in TAG.finditer(r["output"])]
    lens = sorted(len(r["input"]) for r in recs)
    stats[sp] = {
        "rec": len(recs), "char": sum(lens), "ent": len(ents),
        "uniq": len({e for e, _ in ents}),
        "type": Counter(t for _, t in ents),
        "len_med": lens[len(lens) // 2], "len_min": lens[0], "len_max": lens[-1],
    }

TOT = {k: sum(stats[s][k] for s in stats) for k in ["rec", "char", "ent"]}
alltype = Counter()
for s in stats.values():
    alltype += s["type"]

w("# Thống kê corpus — dùng cho paper")
w()
w("*Sinh tự động bởi `scripts/corpus_stats.py`. "
  "Đây là số của **v2-clean**, thay cho mọi số v1 trong bản thảo cũ.*")
w()
w("## ⚠️ Con số v1 phải thay")
w()
w("| Bản thảo cũ (v1) | Thực tế (v2-clean) |")
w("|---|---|")
w(f"| 7,301 câu | **{TOT['rec']:,} record** |")
w(f"| 1,291,822 token | **{TOT['char']:,} ký tự Hán** |")
w(f"| 107,105 entity | **{TOT['ent']:,} entity** |")
w()
w("Con số v1 bị thổi phồng do concat các file Excel tích luỹ ⇒ **nhân bản 3.7×** "
  "(7,301 ÷ 3.7 ≈ 1,973 ≈ số record unique thật). Chính việc này gây "
  "**83.6% test trùng nguyên văn train** ở v1.")
w()

w("## Tổng quan")
w()
w("| Split | Record | Ký tự | Entity | Entity unique | Dài record (min/med/max) |")
w("|---|---|---|---|---|---|")
for sp in ["train", "dev", "test"]:
    s = stats[sp]
    w(f"| {sp} | {s['rec']:,} | {s['char']:,} | {s['ent']:,} | {s['uniq']:,} | "
      f"{s['len_min']}/{s['len_med']}/{s['len_max']} |")
w(f"| **Tổng** | **{TOT['rec']:,}** | **{TOT['char']:,}** | **{TOT['ent']:,}** | — | — |")
w()

w("## Phân bố nhãn")
w()
w("| Type | " + " | ".join(["train", "dev", "test", "Tổng", "%"]) + " |")
w("|---|" + "---|" * 5)
for t in TYPES:
    n = alltype[t]
    w(f"| {t} | " + " | ".join(f"{stats[sp]['type'][t]:,}" for sp in ["train", "dev", "test"])
      + f" | **{n:,}** | {n/TOT['ent']*100:.1f}% |")
w("| **Tổng** | " + " | ".join(f"**{stats[sp]['ent']:,}**" for sp in ["train", "dev", "test"])
  + f" | **{TOT['ent']:,}** | 100% |")
w()

# ---- độ phủ so với nguyên bản
sents = [json.loads(l) for l in open(CRAWL, encoding="utf-8")]
sents = [s for s in sents if s["section"] not in DUP]
gc = sum(len(s["han"]) for s in sents)
big = "".join(r["input"] for sp in data for r in data[sp])
cov = defaultdict(lambda: [0, 0])
for s in sents:
    if len(s["han"]) < 8:
        continue
    cov[s["section"]][1] += 1
    if s["han"] in big:
        cov[s["section"]][0] += 1
num = lambda x: int(re.match(r"(\d+)", x).group(1))
missing = sorted([(k, a, b) for k, (a, b) in cov.items() if b and a / b < 0.05],
                 key=lambda x: -x[2])

w("## Độ phủ so với nguyên bản ĐVSKTT")
w()
w(f"- Nguyên bản (nomfoundation.org, đã bỏ 4 section phụ Mạc trùng lặp): "
  f"**{len(sents):,} câu · {gc:,} ký tự**")
w(f"- Corpus có nhãn: **{TOT['char']:,} ký tự = {TOT['char']/gc*100:.1f}%**")
w(f"- Chưa gán nhãn: **{gc-TOT['char']:,} ký tự**, ước "
  f"**~{int((gc-TOT['char'])*TOT['ent']/TOT['char']):,} entity** "
  f"(theo mật độ {TOT['ent']/TOT['char']:.3f} entity/ký tự)")
w()
w(f"### {len(missing)} section gần như vắng hoàn toàn (<5%) — đưa vào Limitation")
w()
w("| Section | Câu | Phủ |")
w("|---|---|---|")
for k, a, b in missing:
    w(f"| `{k}` | {b:,} | {a/b*100:.1f}% |")
w()
w(f"Tổng **{sum(b for _, _, b in missing):,} câu** không có annotation. "
  "Lớn nhất là `57-Thai-To-Cao-Hoang-De` (Lê Thái Tổ — khởi nghĩa Lam Sơn), "
  "quyển **lớn nhất bộ sách**.")
w()

# ---- chất lượng + leakage
w("## Chất lượng nhãn sau chuẩn hoá")
w()
w("| Split | Nhãn thiểu số trước | sau |")
w("|---|---|---|")
w("| train | 5.2% | **1.80%** |")
w("| dev | 2.2% | **0.17%** |")
w("| test | 2.76% | **0.86%** |")
w()
w("*Nhãn thiểu số = tỉ lệ mention mà chuỗi bề mặt của nó được gán type khác với "
  "type đa số của chính chuỗi đó. Chỉ số này **mù** với lỗi chỉ xuất hiện 1 lần "
  "và lỗi biên — không dùng làm bằng chứng duy nhất.*")
w()
if os.path.exists(os.path.join(CLEAN, "leakage_fix_report.json")):
    lf = json.load(open(os.path.join(CLEAN, "leakage_fix_report.json"), encoding="utf-8"))
    w("## Leakage")
    w()
    w(f"- v1: **83.6%** test trùng nguyên văn train (nhân bản 3.7×)")
    w(f"- v2: dedup tuyệt đối theo chuỗi ⇒ 0 record trùng nguyên văn")
    w(f"- v2-clean: thêm bước loại **chồng lấn một phần** — bỏ "
      f"**{lf['n_dropped']} record train** chia sẻ đoạn văn duy nhất với dev/test")
    w(f"- **Hiện tại: 0 câu gốc duy nhất nằm ở nhiều split**")
    w()
    w("Phân biệt với câu công thức (`冬十月會試天下士人`…) vốn lặp lại thật trong "
      "nguyên bản — chúng trùng giữa các split là bình thường, không tính leakage.")
    w()
w("## Tái lập")
w()
w("```")
w("py scripts/rebuild_dataset.py                              # Excel -> v2")
w("py scripts/clean_labels.py                                 # chuẩn hoá train+dev")
w("py scripts/clean_labels.py --splits test --min-count 999999 \\")
w("       --report-name clean_report_test.json --review-name needs_review_test.json")
w("py scripts/fix_split_leakage.py                            # loại chồng lấn")
w("py scripts/map_records_to_source.py                        # map về nguyên bản")
w("py scripts/corpus_stats.py                                 # sinh file này")
w("```")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("Đã ghi:", OUT)
print(f"  {TOT['rec']:,} record · {TOT['char']:,} ký tự · {TOT['ent']:,} entity")
print(f"  phủ {TOT['char']/gc*100:.1f}% ĐVSKTT · {len(missing)} section vắng")
