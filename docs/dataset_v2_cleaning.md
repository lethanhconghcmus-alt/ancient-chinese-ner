# Dataset v2 — làm sạch nhãn & truy nguyên nguồn gốc

**Ngày:** 2026-08-04
**Phạm vi:** train + dev. **Test KHÔNG bị đụng** — đang duyệt tay qua
`D:\bio_source\gold_review_sheet.xlsx`.
**Script:** `scripts/clean_labels.py`, `scripts/map_records_to_source.py`

---

## 0. TL;DR

| Việc | Kết quả |
|---|---|
| Làm sạch nhãn train/dev | nhiễu train 5.2% → **1.82%**, dev 2.2% → **0.17%** |
| Truy nguyên nguồn text | Excel anh Thiều **copy từ nomfoundation.org** — đã chứng minh |
| Map record → vị trí gốc | **99%+** record có `section` + `position` |
| Leakage | **test sạch** (chỉ dính boilerplate) · **dev dính 6/160 record** |
| Corpus dùng được để mine | **14,283/16,321 câu (87.5%)** không chạm dev/test |

---

## 1. ĐÃ CHỨNG MINH: text v2 lấy từ nomfoundation.org

Trước đây chỉ biết "cùng là DVSKTT". Nay xác định được là **cùng một bản số hoá**.

**Bằng chứng 1 — câu trùng y hệt kèm artifact:**

```
crawl [12-Ky-thuoc-Tuy-Duong 16a*1*1]  視掌如瞽俄而聶天雷震於大石之所數百聲mat-chu然開霽
dev   rec#6                            視掌如瞽俄而聶天雷震於大石之所數百聲mat-chu然開霽.役者駿奔視之石已…
```

**Bằng chứng 2 — folio marker cùng định dạng:** dev rec#92 chứa `[8a*02*02]`,
rec#143 chứa `20a*1*1` — chính là format trường `position` của crawl
(`16a*1*1`, `13a*9*2` = tờ · cột · chữ).

**Bằng chứng 3 — độ phủ ký tự:**

| Split | Record chứa ≥1 câu crawl nguyên văn | Ký tự được phủ |
|---|---|---|
| train | 1,280/1,284 (100%) | ~83% |
| dev | 158/160 (99%) | ~82% |
| test | 159/160 (99%) | ~83% |

### Hệ quả

- `mat-chu` **không phải OCR hỏng** — là **"mất chữ"**, marker cố ý của site
  đánh dấu ký tự không render được. ⇒ phải giữ dạng `<UNK>`, **không xoá**,
  vì chỗ đó có chữ thật bị mất.
- Folio marker `[8a*02*02]` là metadata số hoá lọt vào khi copy ⇒ **xoá được**.
- Dị thể tự / PUA (35 ký tự, 101 lần trong dev) là đặc thù corpus Hán Nôm ⇒ giữ,
  chỉ cần verify tokenizer/font handle được ngoài BMP.

### Quan hệ đúng giữa hai nguồn

Crawl = **văn bản gốc**. Excel anh Thiều = **nhãn gán trên chính văn bản đó**.
Không phải hai dataset cạnh tranh — là text + labels. "Trùng nhau" là quan hệ
bình thường, không phải ô nhiễm.

---

## 2. Phạm vi annotation: v2 chỉ phủ 71.6% DVSKTT

```
crawl  369,637 ký tự  (19,909 câu unique)
v2     264,540 ký tự  (1,604 record, median 169 ký tự/record)
       ->  71.6%
```

Hai con số "1,604 record" và "19,909 câu" **khác đơn vị** — 1 record ≈ 9 câu
crawl. Quy đổi: v2 ≈ 14,248 câu.

### Thiếu theo MẢNG, không rải rác

10 section gần như vắng mặt hoàn toàn (đã thử khớp lỏng 12 ký tự đầu, vẫn 0 —
thiếu thật, không phải lỗi so chuỗi):

| Section | Câu | Phủ |
|---|---|---|
| **57-Thai-To-Cao-Hoang-De** | **1,068** | 0.0% |
| 58-Thanh-Tong-Van-Hoang-De | 808 | 0.2% |
| 59-Nhan-Tong-Tuyen-Hoang-De | 554 | 0.0% |
| 32-Nhan-Tong-Hoang-De | 376 | 0.3% |
| 33-Than-Tong-Hoang-De | 243 | 0.4% |
| 54-Ky-hau-Tran | 196 | 0.0% |
| 56-Ky-thuoc-Minh | 60 | 0.0% |
| 52-Phu-Ho-Quy-Ly | 57 | 0.0% |
| 13-Ky-Nam-Bac-phan-tranh | 42 | 0.0% |
| 5-Ky-Trung-Nu-Vuong | 12 | 0.0% |

Phủ thấp bất thường: `4-Ky-thuoc-Tay-Han` 7.1% · `34-Anh-Tong` 21.2% ·
`50-Thuan-Tong` 29.7% · `60-Thanh-Tong-thuong` 55.1%.
Còn lại 37 section phủ >80%, nhiều cái 95–100%.

> **Cho thesis:** section 57 (Lê Thái Tổ — khởi nghĩa Lam Sơn) là quyển **lớn nhất
> bộ sách** và annotation không có một chữ. Cùng Hai Bà Trưng, Hồ Quý Ly, Hậu Trần,
> thuộc Minh. Đây là **lệch phân bố theo thời kỳ**, phải ghi vào limitation —
> F1 báo cáo là "NER trên 72% DVSKTT", không phải toàn bộ.

**Mặt tích cực:** 105,097 ký tự chưa ai gán nhãn, ước ~8,200 entity
(theo mật độ v2 = 0.078 entity/ký tự) — thêm ~40%. Và **không giao test**.

---

## 3. Làm sạch nhãn — `clean_labels.py`

Áp `docs/annotation_guideline.md` v1.0. Chạy **một lượt từ bản v2 gốc** (không
chồng nhiều lượt) để provenance sạch.

### Kết quả

| Split | Records | Entity | Round-trip | Nhãn thiểu số |
|---|---|---|---|---|
| train | 1,284 | 18,352 | ✅ | 5.2% → **1.82%** |
| dev | 160 | 2,372 | ✅ | 2.2% → **0.17%** |
| test | 160 | 2,211 | ✅ | 2.76% *(không đụng)* |

BIO hợp lệ: **0 lỗi / 237,842 token**. 82/160 record dev có thay đổi.

### 1,499 thay đổi, 6 tầng

| Tầng | Số | Nội dung |
|---|---|---|
| T1 span fix | 59 | `春正` → `春正月` DTM (guideline §5) |
| T2 retype | 951 | `節制` ORG→TITLE ×37 · `太宗` TITLE→PER ×35 · `御史` ×25 · `太祖` ×23 · `進士` ×22 · `聖宗` ×20 · `太師` ×19 · `平安王` PER→TITLE ×18 · `郡公` ×16 |
| T2 drop | 162 | `天下` ×64 · `文武` ×26 · `國家` ×13 · `百姓` ×10 · `祖宗` ×10 · `朝臣` ×8 |
| **B1 gộp bộ+chức** | 71 | `吏部尚書` ×11 · `兵部尚書` ×10 · `東閣大學士` ×7 — guideline §2 |
| **B2 tách chức+tên** | 80 | `史臣吳士連`→`史臣`+`吳士連` ×55 · `節制鄭松` ×9 — guideline §1 |
| T3 majority | 176 | ngưỡng **0.7** + sàn **≥5 lần** (102 chuỗi) |

Majority tính **chỉ trên train+dev**, cố ý không dùng test → không rò nhãn test.

### Ba cái bẫy dry-run bắt được (quan trọng — đừng lặp lại)

1. **`御史` bị majority lật TITLE→ORG.** Sai: ORG là `御史臺`, `御史` là chức quan.
   Luật hậu tố chỉ bắt cụm ≥3 ký tự nên chức quan 2 chữ phải liệt kê riêng
   (`TWO_CHAR_TITLE`).
2. **Hạ ngưỡng xuống 0.7 mù sẽ chốt chết đa số SAI.** `文武`→LOC (19/26),
   `祖宗`→ORG (7/10), `朝臣`→ORG (6/8). Phải đưa vào `DROP` trước rồi mới hạ ngưỡng.
3. **`帝王` bị tách thành `帝`+`王`** ×22, vì `王` trùng họ Vương trong từ điển PER.
   ⇒ B2 yêu cầu **cả hai vế ≥2 ký tự**.

Sàn `--min-count 5` chặn nhóm chuỗi chỉ xuất hiện 3–4 lần mà "đa số" vô nghĩa
(`黎皇朝` 2/1, `紹隆` 3/1, `第三甲` 3/1).

### Còn lại

`data/processed/ner_clean/needs_review.json` — **216 chuỗi**, phần lớn là
ambiguity thật cần guideline v1.1 quyết: `哀牢` ORG/LOC · `蠻` PER/ORG ·
`李氏` ORG/PER · `大行` PER/TITLE.

---

## 4. Audit độc lập (AI khác) — đối chiếu

Họ kiểm bản **sau clean** (số liệu khớp tuyệt đối: 26,013 token line,
PER 863 / TITLE 627 / LOC 410 / DTM 250 / ORG 235).

**Đúng, đã verify:** 4 lỗi nhãn (`茶全`, `黎皇朝`, `文武`, `皇太子`), lỗi biên
`皇太子` tách/gộp, 3 cụm Latin noise, 35 ký tự PUA / 101 lần.

**Sai 2 chỗ quy kết:**

1. *"Nghi do gazetteer auto-tag match nhầm"* — **không**. Gazetteer **chưa từng**
   dùng để gán nhãn. Nhãn từ Excel gán tay. `文武` sai là lỗi annotator.
2. *"OCR misread thành match-u"* — chuỗi là **`mat-chu` = "mất chữ"**, marker của
   nomfoundation (xem §1). Không phải OCR hỏng.

**Đã xử lý sau audit:**

```
茶全      LOC 4 / PER 18   → PER 24        ✅
皇太子     ORG 6 / TITLE 21 → TITLE 29      ✅ hết tách biên
文武      LOC 19           → bỏ tag         ✅
史臣吳士連  1 span PER       → TITLE + PER    ✅ §1
吏部尚書    tách ORG+TITLE   → 1 TITLE        ✅ §2
太祖髙皇帝  PER 17           → giữ nguyên      ✅ miếu+thụy hiệu, §1 cho PER
黎皇朝     PER 2 / ORG 1    → còn nguyên      ⏸ dưới sàn ≥5
```

> **Bài học về metric:** chỉ số "nhãn thiểu số" của dev sau clean là 0.17% —
> nhìn rất đẹp nhưng **mù với hai thứ**: lỗi chỉ xuất hiện 1 lần (`黎皇朝`) và
> lỗi biên (`皇太子`). Đừng dùng nó làm bằng chứng duy nhất cho chất lượng.

---

## 5. Map record → nguồn — `map_records_to_source.py`

| Split | Map được | Chưa map |
|---|---|---|
| train | 1,280/1,284 (99.7%) | 4 |
| dev | 158/160 (98.8%) | 2 |
| test | 159/160 (99.4%) | 1 |

10,265/16,321 câu crawl khớp vào record. Test trải **42 section**, tỉ lệ tương
đồng train ⇒ **split random không lệch thời kỳ**.

### Leakage: 29 câu ở nhiều split → tách 2 loại

Phân biệt bằng **số lần chuỗi đó xuất hiện trong nguyên bản**:

**17 câu công thức — VÔ HẠI.** `冬十月會試天下士人`, `二月會試天下士人`,
`春正月朔日有食之`. Văn sử lặp y hệt qua các đời; trùng vì nguyên bản trùng.

**12 câu TRÙNG THẬT** — chuỗi chỉ xuất hiện **1 lần** trong nguyên bản mà nằm ở
hai split ⇒ cùng một đoạn bị đưa vào hai record. Chúng **cụm lại**:

```
[44-Du-Tong-Hoang-De]  dev+train, 6 câu, cùng đoạn về Chiêm Thành (Chế Mỗ / Trà Hoà Bố Để)
   壬辰十二年元至正十二年春三月占城制某來奔獻白象白馬…      49 ký tự
   𥘉占主制阿難在𠱾其子制某為布田言大王也女壻茶和布底…      42 ký tự
   迨阿難卒布底遂逐制某而自立是知人臣樹黨必有異圖…          36 ký tự
```

### Record bị ảnh hưởng

| Split | Số record | Index |
|---|---|---|
| dev | **6/160 (3.8%)** | 29, 30, 69, 83, 99, 117 |
| train | 7 | 285, 601, 640, 642, 729, 1062, 1144 |
| test | 1/160 (0.6%) | 64 — chỉ dính `大越史記本紀全書卷之六` (tiêu đề quyển) |

⇒ **Test coi như sạch.** **Dev có leakage thật 3.8%** — đủ làm lệch việc chọn
checkpoint. Đây là loại lỗi **dedup theo chuỗi KHÔNG bắt được**, vì hai record chỉ
chồng lấn *một phần*. Họ hàng gần của lỗi đã giết v1.

**Chưa xử lý.** Phương án đề xuất: bỏ 7 record train (mất 0.5% train, giữ dev
nguyên 160 để so được với kết quả cũ).

---

## 6. Gazetteer: leakage được định lượng, không còn là lo ngại mơ hồ

> Thay cho §10 của `gazetteer_findings.md` — chỗ đó chỉ ghi "cùng một cuốn sách".

Corpus crawl **chứa nguyên văn ~83% ký tự của test**. Mine gazetteer từ toàn bộ
19,909 câu = câu test nằm thẳng trong pool khai thác.

**Đã có blocklist chính xác:**

```
data/processed/ner_clean/gazetteer_blocklist.txt   2,038 câu CẤM (chạm dev/test)
   -> còn 14,283/16,321 câu (87.5%) dùng được để mine / pretrain
```

### Gazetteer dựng từ nhãn train KHÔNG phải leakage

Phân biệt hai thứ hay bị gộp: **entry lấy từ đâu** vs **áp lên đâu**.

| Nguồn entry | Áp lên | Leakage? |
|---|---|---|
| Nhãn **train** | train + test | **Không** — như học embedding từ train |
| Nhãn **test** | bất kỳ | **Có** |
| Crawl thô | — | không dùng nhãn, nhưng crawl chứa câu test ⇒ **xám** ⇒ dùng blocklist |
| Nhãn train | 28% chưa gán nhãn | **Không** |

**Trên phần đã có nhãn, gazetteer không thêm gì.** Giá trị nằm ở 3 chỗ:

1. **Gán nhãn 105k ký tự chưa annotate** (~8,200 entity, +40%)
2. **Thông tin ranh giới** — lexicon chứa `郡公` dạy dạng thức `X郡公`, giúp cả với
   entity chưa từng thấy ⇒ không sụp theo kiểu 8.2% của PER
3. **Train → test lúc inference** — công dụng kinh điển, không rò nhãn test

### Bẫy của silver-labeling: FALSE NEGATIVE

Gazetteer chỉ tag cái nó biết; phần còn lại thành `O` — mà `O` nghĩa là
"chắc chắn không phải entity", không phải "chưa biết".

Trần recall (union luật + list, đo ở `gazetteer_findings.md` §7):
DTM 95.1% · TITLE 81.3% · ORG 73.6% · PER 68.6% · LOC 64.8%

⇒ **~25–35% entity trong 105k sẽ bị gán `O` sai ≈ ~2,400 nhãn sai hướng phủ định**
— nhiều hơn tổng số lỗi vừa sửa (1,499).

Ba cách xử lý, tăng dần độ an toàn:
1. **Partial annotation** — vùng không khớp đánh `unknown`, loss bỏ qua (fuzzy CRF, kiểu AutoNER)
2. **Lọc câu** — chỉ giữ câu gazetteer phủ dày
3. **Bỏ silver, dùng gazetteer làm feature** — không sinh nhãn thì không có false negative

### Ba cách dùng crawl, xếp theo tỉ lệ lợi/rủi ro

| Cách | Cần nhãn? | Rủi ro | Công |
|---|---|---|---|
| **Pretrain domain-adaptive** (370k, trừ blocklist) | Không | ~0 | Thấp — đã có `run_pretrain.py` |
| **Gazetteer làm feature** | Nhãn train | Thấp | Trung bình |
| **Silver-label 105k** | Nhãn train | **Cao** | Cao |

### Thứ tự đánh giá đúng

```
1. Dựng gazetteer   ← chỉ từ nhãn TRAIN + luật
2. Đo trên DEV      ← precision/recall như tagger độc lập
3. Chỉnh lexicon    ← lặp 2–3
4. Gán nhãn 105k    ← silver (nếu làm)
5. Train            ← gold + silver
6. Đo trên TEST     ← MỘT LẦN, con số báo cáo
```

**Không đo/tune gazetteer trên test** — làm thế là burn test set.

---

## 7. File

```
scripts/clean_labels.py                              (mới)
scripts/map_records_to_source.py                     (mới)

data/raw/ner_sft/{train,dev}.jsonl                   (đã clean)
data/processed/ner_clean/{train,dev}.jsonl           (đã clean)
data/processed/ner_clean/{train,dev}.bio.txt         (đã clean)
data/raw/ner_sft/test.jsonl                          (KHÔNG đụng)

data/processed/ner_clean/clean_report.json           1,499 thay đổi, chi tiết từng tầng
data/processed/ner_clean/needs_review.json           216 chuỗi chờ quyết
data/processed/ner_clean/record_source_map.json      1,604 record → section + position
data/processed/ner_clean/source_split_map.json       10,265 câu gốc → split
data/processed/ner_clean/gazetteer_blocklist.txt     2,038 câu cấm mine
```

---

## 8. Việc còn treo

1. **Test gold review** — đang duyệt tay, 58/982 dòng (5.9%).
   ⚠️ `gold_review_sheet.xlsx` nằm **ngoài git**, không backup. Nên đưa vào repo.
   ⚠️ `review_rules.py` (rule engine sinh 457 đề xuất) **đã mất** khỏi máy.
2. **6 record dev leakage** — chưa xử lý (đề xuất: bỏ 7 record train tương ứng).
3. **216 chuỗi `needs_review`** — cần guideline v1.1.
4. **Folio marker Latin** trong rec#92, #143 (và train) — chưa strip.
5. **`mat-chu`** — chưa chuyển thành `<UNK>`.
6. Chưa commit gì.
