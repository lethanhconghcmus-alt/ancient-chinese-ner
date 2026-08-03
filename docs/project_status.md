# TRẠNG THÁI DỰ ÁN — đọc file này TRƯỚC

**Cập nhật:** 2026-08-04
**Hạn chót:** **30/08/2026** — full paper VCL2026 (còn 26 ngày)
**Kế hoạch chi tiết:** `ke_hoach_VCL2026.xlsx` (4 sheet, có filter + dropdown trạng thái)
**Nhật ký gốc:** `report.xlsx` (sheet `Report`, ghi từ 11/2025)

---

## 0. Một câu tóm tắt

Toàn bộ số liệu trên dữ liệu DVSKTT đều chạy trên **corpus v1 bị hỏng**
(nhân bản 3.7× ⇒ 83.6% test trùng nguyên văn train). Chỉ **CHisIEC 0.9209**
là sạch. Cần rerun baseline + E2 trên v2-clean, và đóng khung lại paper thành
đóng góp về **chất lượng dữ liệu** thay vì đuổi theo F1 cao.

---

## 1. Bối cảnh — có những gì

| Hạng mục | Trạng thái |
|---|---|
| **Luận văn thạc sĩ** | NER Hán văn Việt Nam (ĐVSKTT), GVHD theo dõi qua `report.xlsx` |
| **Paper 1** | Đã định dạng ACL/LaTeX xong (08/06). **CHƯA submit.** Số liệu trên v1. |
| **Paper 2 — VCL2026** | **Hạn 30/08/2026.** Abstract ĐÃ đăng ký, một phần dùng kết quả v1. |
| **Web app** | API NER + UI highlight + cột phiên âm + Docker (05/2026). Không ảnh hưởng. |
| **Dataset** | v1 hỏng → v2 rebuild (29/07) → v2-clean (04/08) |

---

## 2. Mốc đã làm được (rút từ `report.xlsx`)

| Ngày | Việc | Kết quả | Dữ liệu |
|---|---|---|---|
| 17/11/2025 | Related work: EvaHan2025, BAC-GNN-CRF, LC-RoBERTa | chọn hướng PLM+CRF | — |
| 29/11/2025 | Tải CHisIEC | 14,194 entity / 10,087 câu | — |
| **16/03/2026** | Pipeline CHisIEC | **F1 = 0.9209** (GuwenBERT+CRF) | **CHisIEC ✅** |
| 16/03/2026 | Transfer CHisIEC→DVSK | 0.6017 → 0.7575 | v1 ❌ |
| 16/03–13/04 | GazBertCRF, gazetteer 13,188 entry | 0.8124 | v1 ❌ + leakage |
| 05/2026 | Web app, API, UI phiên âm, Docker | chạy được | — |
| **01–08/06** | **Paper 1** | **82.13% vs 75.75%** | v1 ❌ |
| 10/07/2026 | Pipeline Qwen2.5-7B+LoRA, 3 notebook | chạy 4.5h | — |
| **27/07/2026** | **Paper 2: E1/E2/E3** | 0.0952 / **0.7507** / 0.6888 | v1 ❌ |
| 28–29/07 | Phát hiện v1 hỏng → rebuild v2 | 1,604 record | — |
| 03/08 | Crawl 20K câu nomfoundation, gazetteer mới | ORG/TITLE/LOC tốt, PER/DTM rule tốt hơn | — |
| 04/08 | Clean nhãn + truy nguyên nguồn + map | xem `dataset_v2_cleaning.md` | — |

---

## 3. Bán kính ảnh hưởng của lỗi v1

### ✅ Sống sót — KHÔNG phải làm lại

- **CHisIEC F1 = 0.9209** — dataset ngoại lai, sạch. Dùng làm điểm neo.
- Web app / API NER / UI / Docker
- Pipeline code, 3 notebook, Git workflow
- Related work, `annotation_guideline.md` v1.0

### ❌ Phải làm lại

| Kết quả | Giá trị | Task |
|---|---|---|
| Paper 1 GuwenBERT-CRF | 82.13% | A4 |
| Paper 1 baseline (no CRF) | 75.75% | A4 |
| Per-entity F1 | PER .88 · LOC .85 · ORG .79 · DTM .79 · TITLE .76 | A4 |
| DVSK baseline → transfer | 0.6017 → 0.7575 | A4 |
| GazBertCRF | 0.8124 | C3+C4 |
| E2 (pretrain+SFT) | 0.7507 | **A5** |
| E3 (RAG TF-IDF) | 0.6888 | D1 (hoãn) |
| E1 | 0.0952 (NaN) | **HUỶ, không rerun** |
| Error analysis mục 6.4 | `error_analysis.txt` | B3 |

### ⚠️ Nguy hiểm nhất — không phải F1 mà là MÔ TẢ DỮ LIỆU

```
Paper 1 ghi:  7,301 câu · 1,291,822 token · 107,105 entity
Sự thật:      con số này TỰ THÂN đã sai — sản phẩm của nhân bản 3.7×
              thật ~1,961 record unique → v2 còn 1,604
```

Nếu **abstract VCL2026 đã quảng cáo quy mô corpus theo con số này**, đó là sai
về mô tả dữ liệu chứ không chỉ sai kết quả. Phải rà gấp (task B1).

Kiểm chứng: 7,301 ÷ 3.7 ≈ 1,973 ≈ 1,961 record unique; mật độ 14.7 entity/câu
khớp đúng v2 (14.4) ⇒ xác nhận Paper 1 dùng v1.

---

## 4. Điểm chặn đã gỡ được — QUAN TRỌNG

**Tưởng:** phải duyệt xong 924 dòng gold test mới rerun được.
**Thật:** không. Chỉ cần chạy `clean_labels.py` lên test (30 phút).

Lý do phải làm: train/dev đã áp guideline v1.0 (`節制`→TITLE, `春正月`,
`吏部尚書` gộp, `史臣`+`吳士連` tách…), test thì chưa. Train trên nhãn đã chuẩn
hoá rồi đo trên test chưa chuẩn hoá ⇒ **model bị trừ điểm vì trả lời ĐÚNG**,
và lệch không đều giữa các type.

Test sau clean đã: **không leakage** (1/160 record chỉ dính boilerplate) và
nhãn tốt hơn v1 rất xa. Duyệt tay là lớp **thêm**, áp sau — **không gating**.

---

## 5. Định hướng lại paper

Đổi trọng tâm đóng góp:

> ~~"GuwenBERT-CRF đạt 82.13% F1 trên NER Hán văn Việt Nam"~~
>
> **"Chúng tôi phát hiện corpus NER ĐVSKTT hiện có bị nhân bản 3.7× dẫn tới
> 83.6% rò rỉ train-test cùng lỗi nhãn hệ thống; chúng tôi rebuild corpus,
> chuẩn hoá nhãn theo guideline, và báo cáo benchmark đã hiệu chỉnh."**

Với miền ít tài nguyên như Hán văn Việt Nam, đây là contribution **thật** và
dễ bảo vệ hơn một con số F1 cao. **F1 tụt trở thành bằng chứng, không phải
thất bại.**

Vật liệu đã có sẵn: `docs/dataset_v2_cleaning.md` (rebuild · clean 1,499 sửa ·
truy nguyên nguồn · leakage định lượng) và `docs/gazetteer_findings.md`.

Thêm hai điểm mới có thể đưa vào paper:
- **Phân tích lỗi theo THỜI KỲ** — nhờ `record_source_map.json` gắn được
  section/position cho 99%+ record
- **Limitation về phạm vi**: v2 chỉ phủ **71.3%** ĐVSKTT, 10 section vắng hẳn,
  lớn nhất là `57-Thai-To` (Lê Lợi, 1,068 câu)

---

## 5b. ĐÃ XONG 04/08 — A1 · A2 · B2

| Task | Kết quả |
|---|---|
| **A1** clean test | 148 sửa, chạm 65/160 record. Chỉ áp luật guideline, **tắt majority-vote** (`--min-count 999999`) để không có quyết định thống kê nào dựa trên chính test. Nhiễu test 2.76% → **0.86%** |
| **A2** loại leakage | Bỏ **7 record train** chia sẻ đoạn văn duy nhất với dev/test. train 1,284 → **1,277**; dev/test giữ nguyên 160. **Leakage nay = 0** |
| **B2** thống kê corpus | `docs/corpus_stats.md` — bảng sẵn dùng cho paper |

**Số chốt thay cho v1** (chi tiết: `docs/corpus_stats.md`):

| Bản thảo cũ (v1) | v2-clean |
|---|---|
| 7,301 câu | **1,597 record** |
| 1,291,822 token | **263,638 ký tự Hán** |
| 107,105 entity | **22,858 entity** |

Split 1,277 / 160 / 160 · PER 32.7% · TITLE 24.8% · LOC 20.9% · DTM 11.0% · ORG 10.7%
Nhiễu nhãn: train 1.80% · dev 0.17% · test 0.86% · BIO hợp lệ 0 lỗi / 263,638 token

> **⇒ A4/A5 (rerun baseline + E2) giờ chạy được ngay. Không còn gì chặn.**
> Việc duyệt 924 dòng gold test vẫn tiếp tục độc lập, apply lên trên rồi rerun
> `clean_labels.py` là xong — script xác định, lặp lại được.

---

## 6. Đường găng 26 ngày

```
T1 (04–10/08)  A1 clean test  →  A2 dev leakage  →  A3 backup git  →  A4 rerun baseline
               B1 rà abstract
T2 (11–17/08)  A5 rerun E2 · B2 thống kê corpus · B3 error analysis v2
T3 (18–24/08)  B5 viết paper · (C3/C4 gazetteer nếu kịp)
T4 (25–30/08)  hoàn thiện, rà soát
```

Chi tiết từng task + output cụ thể: **`ke_hoach_VCL2026.xlsx`** sheet `Kế hoạch`.

---

## 7. Rủi ro CAO

| Rủi ro | Chặn bằng |
|---|---|
| `gold_review_sheet.xlsx` nằm **ngoài git, không backup** — 58 dòng đã duyệt | Task A3, làm ngay |
| Abstract có thể đã quảng cáo quy mô corpus v1 | Task B1 |
| 26 ngày | Bám P0→P1, hoãn hết P2/P3 |

Ngoài ra: `review_rules.py` (rule engine sinh 457 đề xuất) **đã mất khỏi máy**,
chỉ còn output `review_proposals_rules.json`.

---

## 8. Đừng lạc hướng — ghi lại để khỏi lặp

Ngày 03–04/08 đã tiêu khá nhiều thời gian vào **chiến lược gazetteer**
(luật vs list, thí nghiệm transfer, độ đóng từ vựng). Phân tích đúng, nhưng
**không phải đường găng**.

Phần thực sự dùng được cho 26 ngày tới chỉ là:
- Clean train/dev (đã xong)
- Phát hiện dev leakage (đã xong)
- Blocklist 2,038 câu (đã xong, dùng khi nào làm gazetteer)

Và kết luận gazetteer của phân tích đó **trùng khớp với kết luận bạn đã tự rút
ra ngày 03/08** (ORG/TITLE/LOC dùng list · PER/DTM dùng rule) — tức là *xác
nhận*, không phải phát hiện mới.

**Gazetteer là nice-to-have (P2), không gating.** Ưu tiên: test sạch → rerun →
viết.
