# Khảo sát dữ liệu gazetteer DVSKTT — findings

**Ngày:** 2026-08-03
**Phạm vi:** khảo sát `data/raw/gazetteer/` (chưa commit) để quyết định chiến lược xây gazetteer.
**Trạng thái:** đã khảo sát + đo đạc + **build xong TITLE/ORG/LOC**
(2026-08-10, xem mục 12).

> **Cập nhật 2026-08-04** — xem `docs/dataset_v2_cleaning.md` cho các phát hiện
> đến sau và đã kiểm chứng:
> - Text của dataset v2 **lấy từ chính nomfoundation.org** ⇒ crawl và gold không
>   phải hai nguồn độc lập (§1 doc đó)
> - v2 chỉ phủ **71.6%** DVSKTT; 10 section vắng hoàn toàn, lớn nhất là
>   `57-Thai-To` 1,068 câu (§2)
> - Blocklist 2,038 câu đã sinh; **87.5% corpus dùng được** để mine (§6)
> - Mục 10 bên dưới (Leakage) **đã lỗi thời**, đọc bản thay thế ở đó

---

## 1. Nguồn và độ phủ — đã lấy trọn DVSKTT

Nguồn: `https://nomfoundation.org/nom-project/history-of-greater-vietnam/Fulltext`
(bản số hoá Đại Việt Sử Ký Toàn Thư của Nôm Foundation — nguyên văn Hán + phiên âm + chú thích;
site render server-side, phân trang bằng POST `curPg`, không có API).

Script: `scripts/scrape_dvsktt_gazetteer.py` (untracked)

| | |
|---|---|
| Slug site công bố | **73** |
| Slug script khai báo (`SECTION_SLUGS`) | **73** — trùng khít |
| Slug có nội dung | 72 |
| Slug rỗng | 1 — `51-Thieu-De` |

**Đã verify với index của site:** các ID vắng mặt (15–23, 25, 27–28, 45, 47, 53, 55, 69,
72–73, 78, 87–99) **không tồn tại trên site**. Đánh số của Nôm Foundation vốn không liên tục —
không phải crawl sót.

`51-Thieu-De` rỗng ở phía site (crawler thấy 1 trang, trích 0 câu; trang không có body text).
Không mất sử liệu: giai đoạn Trần Thiếu Đế 1398–1400 chép trong `52-Phu-Ho-Quy-Ly-Ho-Han-Thuong`.

Phạm vi: **Hồng Bàng thị → Gia Tông Mỹ Hoàng Đế (1675)** — đúng điểm đầu/cuối bộ sách,
cộng 7 mục phụ lục (`100-106`: tục biên tự/thư, ngoại kỷ toàn thư, biểu, phàm lệ,
kỷ niên mục lục, khảo tổng luận).

### Cấu trúc

| Nhóm | Section | Câu |
|---|---|---|
| Ngoại kỷ | 1–14 | ~1,700 |
| Bản kỷ | 24–56 | ~7,400 |
| Thực lục / Tục biên | 57–86 | ~12,100 |
| Phụ lục | 100–106 | 556 |

---

## 2. Trùng lặp — 4 cặp section trùng 100%

Verify bằng so tập câu (không phải trùng số lượng ngẫu nhiên):

```
79-The-Tong-Nghi-Hoang-De  ≡  80-Phu-Mac-Hau-Hop
76-Anh-Tong-Tuan-Hoang-De  ≡  77-Phu-Mac-Phuc-Nguyen
70-Trang-Tong-Du-Hoang-De  ≡  71-Phu-Mac-Dang-Doanh-Mac-Phuc-Nguyen
74-Trung-Tong-Vu-Hoang-De  ≡  75-Phu-Mac-Phuc-Nguyen
```

Đặc thù thể lệ DVSKTT thời Lê trung hưng: kỷ nhà Lê là chính, nhà Mạc chép **phụ** vào cùng chỗ
→ site phục vụ cùng nội dung dưới hai slug.

**Bỏ 4 section phụ → mất 1,751 câu → còn 19,909 câu, unique đúng bằng 19,909 (hết trùng).**

> **Con số làm việc: 19,909 câu / ~370K chữ Hán** — không phải 21,660 như file thô.

---

## 3. Chất lượng dữ liệu

### `dvsktt_sentences.jsonl` — tốt

Schema đồng nhất 21,660/21,660 câu: `han`, `han_chars`, `page_idx`, `phienam`, `position`, `section`.
**Không có trường nhãn nào** — đây là raw corpus chưa gán nhãn, không phải BIO.

| Chỉ số | Giá trị |
|---|---|
| Thiếu `han` / `phienam` | 0 / 0 |
| `len(han_chars) == len(han)` | ✅ toàn bộ |
| Tổng chữ Hán | 403,684 (thô) |
| Độ dài câu | p25=9, median=15, p90=36, max=207 |

Nhiễu trong `han`: 2,404/403,684 ký tự = **0.6%**

- `〇` ×922 — dấu ngắt đoạn bản gốc, **không phải lỗi**
- `[` `]` ×1,307 (601 câu) — chú thích chua nhỏ trong nguyên bản, vd `杜慶献黃色[鯧䰸魚鯧音昌...]`.
  Nội dung thật nhưng lồng trong câu → **sẽ làm lệch offset entity nếu không tách**
- `?` `>` ×89 (63 câu) — **glyph hỏng thật**, chữ Nôm/hiếm site không render: `移撫治於???????`
- `mat-chu` ×11 câu — rác HTML rò vào giữa chữ Hán: `太后恐乱mat-chu欲倚漢威`

### `dvsktt_terms.jsonl` — kém, cần làm lại

| | |
|---|---|
| Term | 1,777 (1,405 unique) |
| Có chữ Hán | 853 = **48%** (nửa còn lại chỉ có âm Hán Việt → không project vào text Hán được) |
| Rác cắt sai | 64 dòng = 3.6% |

Rác: `Theo Sử Ký (q. 113`, `Cương mục (TB1, 16b) chú`, `Nghĩa là`, `- 13 xứ thừa tuyên` —
parser cắt term ở dấu phẩy/ngoặc đầu tiên.

**`type_guess` sai nhiều hơn đúng — không dùng được.** Phân bố: LOC 965 / PER 622 / ORG 157 /
TITLE 23 / DTM 10. Mẫu lỗi:

| Gán | Term | Đúng ra |
|---|---|---|
| PER | Thảng Do 儻猶 — *"tên châu"* | LOC |
| PER | Huyện Hữu Lũng — *"nay thuộc tỉnh Hà Bắc"* | LOC |
| PER | Mộc Lạc 木落 — *"cây đổ, cây rụng"* | không phải entity |
| LOC | Lâm Phủ 林甫 — *"tức Lý Lâm Phủ, tể tướng đời Đường"* | PER |
| LOC | Khương Chủng 姜种 — *"là người cùng phe cánh…"* | PER |
| ORG | Nam Định 南定 — *"tên huyện nhà Đường đặt"* | LOC |

→ **Dùng `definition` để gán lại type, bỏ hẳn `type_guess`.** Definition thường nói thẳng loại
("tên châu", "là người…", "niên hiệu của…", "chức quan…").

---

## 4. Phát hiện then chốt: căn chỉnh âm tiết ↔ chữ Hán

Hán văn đơn âm tiết, phiên âm Hán Việt viết hoa danh từ riêng.

**21,049/21,660 câu (97.2%) có số âm tiết phiên âm khớp đúng số chữ Hán.**

→ Mỗi cụm viết hoa trong `phienam` chiếu 1-1 về một span ký tự trong `han`,
lấy được entity **kèm dạng chữ Hán chuẩn**, tự động, toàn corpus.

Prototype: **11,371 cụm unique** (≥2 âm tiết), độ ổn định ánh xạ cao —
Trịnh Tùng 鄭松 100%, Chiêm Thành 占城 99%, Ngô Sĩ Liên 吳士連 97%.

Khớp với gold v2: **2,770/11,245 ứng viên (24.6%)** tự gán được type; phủ ngược 33.4% entity v2.
Ứng viên mới đáng chú ý: `阮廌`, `王通`, `莫玉輦`, niên hiệu `元至元`/`明洪武`/`清順治`, can-chi.

### Lỗi hệ thống của phép chiếu

**Gộp cụm liền kề:** `明嘉靖` (Minh + niên hiệu Gia Tĩnh), `宋紹興`, `清華乂安` (Thanh Hoa + Nghệ An),
`交趾九真`. Run viết hoa tham lam nuốt hai entity cạnh nhau.

**Mất hậu tố chỉ loại (LOC):** phiên âm viết hoa hậu tố **không nhất quán**.
Trong 3,107 lần LOC gold kết thúc bằng danh từ chung chỉ loại:

| | Số | % |
|---|---|---|
| Thân hoa, **hậu tố thường** → mất hậu tố | 1,396 | **44.9%** |
| Cả cụm hoa → bắt đúng | 1,166 | 37.5% |
| Thân không hoa hẳn | 545 | 17.5% |

`交州`→"Giao Châu" ✅ nhưng `楊州`→"Dương châu" ❌, `陀江`→"Đà giang", `螺城`→"Loa thành",
`七耀山`→"Thất Diệu sơn". Tệ hơn: thân 1 âm tiết thì bộ lọc "≥2 âm tiết" **vứt luôn cả entity**.

**Cách vá:** mở rộng run sang phải khi âm tiết thường kế tiếp ∈ {châu, huyện, phủ, xã, sơn, giang,
thành, quận, trấn, lộ, động, sách, trang, hải, khẩu, quan, tân, độ, kiều, cung, điện, môn}
**và** chữ Hán tương ứng ∈ `州縣府社山江城郡鎮路洞柵莊海口關津渡橋宫殿門`. Kiểm hai lớp để tránh mở rộng bừa.

---

## 5. Độ chính xác biên của phép chiếu, theo type

Đo với gold v2. ⚠️ *Dùng `han.find()` nên có nhiễu (bắt cả chuỗi trùng ngẫu nhiên như 湖南 "hồ nam"
nghĩa thường, và nhãn rác còn sót trong v2). Chênh lệch giữa các type quá lớn để là ngẫu nhiên,
nhưng con số tuyệt đối nên coi là xấp xỉ.*

| Type | Biên đúng | Ghi chú |
|---|---|---|
| PER | **42.6%** | tốt nhất |
| LOC | 28.9% | → ~45–50% sau khi vá hậu tố |
| ORG | 5.9% | 19.4% mention là entity **1 chữ** → bị bộ lọc ≥2 âm tiết giết sạch |
| TITLE | 3.3% | viết thường gần hết (`大夫`, `將軍`) |
| DTM | 4.0% | can-chi `丁卯` = "đinh mão", thường hoàn toàn |

Tỉ lệ entity 1 chữ: ORG 19.4% · PER 6.4% · TITLE 2.3% · LOC 1.2% · DTM 1.1%

**Kết luận:** viết hoa chỉ là tín hiệu tốt cho **PER và LOC**. Ba type còn lại gần như vô hình.

---

## 6. Vì sao: viết hoa nghịch với độ đóng từ vựng

Chính tả tiếng Việt viết hoa **danh từ riêng** — thứ có sở chỉ duy nhất (người, nơi chốn).
Chức quan / niên đại / cơ quan là **danh từ chung** (`đại phu`, `tháng ba`, `bộ Lại`) nên viết thường.
Mà danh từ chung thì bản chất là từ vựng đóng.

→ **Thứ viết hoa = thứ không liệt kê hết được; thứ viết thường = thứ liệt kê được.**
Tương quan có cấu trúc, không phải trùng hợp.

Độ đóng từ vựng đo trên gold v2:

| Type | Unique | Mention | top50 phủ | top200 phủ | hapax |
|---|---|---|---|---|---|
| ORG | 953 | 2,916 | **48.0%** | **70.4%** | 67.5% |
| DTM | 1,068 | 2,447 | 46.1% | 64.3% | 80.7% |
| TITLE | 1,431 | 4,873 | 40.7% | 64.1% | 62.5% |
| LOC | 2,059 | 4,993 | 29.6% | 49.0% | 67.8% |
| PER | 3,391 | 7,859 | 16.9% | 34.8% | 64.3% |

Top-12 gold:
- **ORG**: `宋 吏部 唐 元 翰林院 漢 越 禮部 節制 刑部 東閣 莫兵`
- **TITLE**: `上皇 太后 帝 太子 太保 皇帝 太師 太尉 尚書 皇后 進士 公主`
- **DTM**: `三月 二月 九月 冬十月 八月 十二月 六月 秋七月 夏四月 五月 春正月 十一月`

---

## 7. Luật vs gazetteer: khi nào gazetteer thắng

Gazetteer thắng khi đủ **ba** điều kiện: (1) không sinh được bằng luật, (2) đầu nặng,
(3) không nhập nhằng khi khớp mù.

Luật dùng để đo: DTM = pattern tháng + 60 can-chi + `年月日`; PER = 29 chữ họ
`阮黎陳鄭莫丁吳李胡范杜武…`; TITLE = 28 hậu tố `保師尉傅尚書將軍卿使…`;
ORG = hậu tố cơ quan `部院臺監司寺閣館` + entity 1 chữ; LOC = hậu tố `州縣府社山江城郡…`

| Type | Luật phủ | List-200 phủ | Chênh |
|---|---|---|---|
| ORG | 38.0% | **70.4%** | **+32** |
| DTM | **91.9%** | 64.3% | −28 |
| LOC | 29.0% | **49.0%** | **+20** |
| TITLE | 47.0% | **64.1%** | **+17** |
| PER | **50.5%** | 34.8% | −16 |

### Nhưng chúng BÙ nhau, không đối đầu

| Type | Chỉ luật bắt | Chỉ list bắt | Cả hai | **Union** | Không ai bắt |
|---|---|---|---|---|---|
| DTM | 30.8% | 3.2% | 61.1% | **95.1%** | 4.9% |
| TITLE | 17.2% | 34.3% | 29.8% | **81.3%** | 18.7% |
| ORG | 3.2% | 35.7% | 34.7% | **73.6%** | 26.4% |
| PER | 33.8% | 18.2% | 16.6% | **68.6%** | 31.4% |
| LOC | 15.8% | 35.9% | 13.2% | **64.8%** | 35.2% |

Cột "chỉ luật bắt" ≠ 0 ở **mọi** type → không type nào nên bỏ hẳn một trong hai.
LOC có 35.2% **không ai bắt** — đây là type cần mining từ corpus nhất.

### Ranh giới luật/gazetteer là một phổ

| | Ví dụ | |
|---|---|---|
| Regex sinh thuần | `[春夏秋冬]?[số]月` | thuần nội hàm, không list |
| Luật có ô từ vựng đóng | họ ∈ {29 chữ} + 2 chữ | list nhỏ *bên trong* luật |
| Tích chéo | {太少大上} × {保師尉傅} | list × list → list lớn |
| Liệt kê thuần | `阮廌`, `翰林院` | thuần ngoại diên |

TITLE nằm ở tầng 3 → rẻ bất ngờ: curate 14 tiền tố + 28 hậu tố, tích chéo ra vài trăm mục.

---

## 8. Thí nghiệm chuyển giao: gazetteer trên "tập data mới"

Xây gazetteer từ nửa đầu DVSKTT (section ≤56, Ngoại kỷ→Trần) → áp lên nửa sau
(≥57, Lê–Mạc, cách ~200 năm). 5,250 entity → gặp 6,971 unique / 14,879 mention.

**Phủ tổng: 7.9% unique, 16.0% mention.**

| Type | Phủ mention trên "data mới" |
|---|---|
| **DTM** | **92.1%** |
| TITLE | 49.0% |
| LOC | 44.7% |
| ORG | 30.5% |
| **PER** | **8.2%** |

**Nguyên lý: khả năng chuyển giao tỉ lệ nghịch với mức "cá thể" của entity.**
`三月`/`甲子` là từ vựng → 92%. `太保`/`尚書` là định chế → 49%. Núi sông còn đó qua các đời → 45%.
`阮廌`/`鄭松` là con người cụ thể → **8.2%**.

Con số 16% tổng thể vô nghĩa — thấp chỉ vì PER chiếm phần lớn khối lượng và PER thì sụp.

**Củng cố phán quyết về PER bằng lập luận độc lập:**

| PER | In-domain | Sang text mới |
|---|---|---|
| Luật họ `阮黎陳鄭…` | 50.5% | ~giữ nguyên |
| Gazetteer tên người | 34.8% | **8.2%** |

Cảnh báo: 44.7% của LOC là **lạc quan** — "data mới" ở đây vẫn cùng cuốn sách, cùng thể loại,
cùng quy ước chính tả, cùng vùng địa lý. Áp lên *Đại Nam Thực Lục* hay Hán tịch TQ sẽ tụt mạnh hơn.
Và đây mới là **recall**, chưa đo precision (sang text mới, chuỗi từng là entity có thể thành từ thường).

---

## 9. Kết luận: chiến lược theo từng type

| Type | Công cụ chính | Cần list tay? | Dùng viết hoa? | Ghi chú |
|---|---|---|---|---|
| **TITLE** | Lexicon + tích chéo | ~300 mục | Không | ⭐ ưu tiên 1 — nhập nhằng thấp, model khó suy nhất |
| **ORG** | Lexicon + hậu tố | ~150 mục | Không | 19.4% entity 1 chữ → **bắt buộc gate ngữ cảnh** |
| **LOC** | Mining + vá hậu tố | Không | Có | 35.2% không luật/list nào bắt được |
| **DTM** | **Rule module**, không phải gazetteer | Chỉ niên hiệu | Không | luật phủ 91.9% |
| **PER** | Luật họ; gazetteer chỉ cho **RAG** | Không | Có | list thua luật cả in-domain lẫn transfer |

### Thứ tự đề nghị

1. **DTM** — thuần rule, precision gần 100%, dựng khung validate trước
2. **TITLE + ORG** — lexicon nhỏ, phủ ~65–70%, phần *đáng gọi là gazetteer* nhất
3. **LOC** — mining + vá hậu tố, kèm cờ `needs_review`
4. **PER** — mining ra cho RAG, **không** đưa vào weak-labeling

### Đưa vào như FEATURE, không như LABEL

Dòng nghiên cứu lexicon-enhanced Chinese NER (Lattice LSTM · FLAT · SoftLexicon · LEBERT)
cho thấy vai trò hợp hơn: gazetteer cấp **thông tin ranh giới từ** cho tagging cấp ký tự,
chứ không phải ghi nhớ entity. *(Nhắc theo trí nhớ — cần verify tên/năm/số liệu nếu trích dẫn.)*

BIO của repo đã là **cấp ký tự** (`少 B-TITLE / 保 I-TITLE / 豪 I-TITLE / 郡 I-TITLE / 公 I-TITLE`).
Model nhìn `公` đơn lẻ không biết là `I-TITLE` hay `B-` của gì khác — lexicon chứa `少保`/`郡公`
cấp đúng tín hiệu đang thiếu. **Giá trị này không sụp trên entity mới** vì nó dạy *dạng thức*
`X郡公`, không dạy cá thể → 8.2% của PER chỉ là bản án cho *vai trò sinh nhãn*.

| | Gazetteer → **nhãn** | Gazetteer → **đặc trưng** |
|---|---|---|
| Gazetteer sai | thành gold sai, **không sửa được** | model học cách bỏ qua |
| Entity mới | trượt | vẫn giúp (gợi ý ranh giới) |
| Rủi ro | lặp lại cơ chế hỏng v1 | gold giữ sạch |

Cụ thể: với GuwenBERT-CRF nhồi tín hiệu lexicon vào embedding ký tự kiểu SoftLexicon;
với Qwen SFT đưa danh sách entity ứng viên vào prompt.

---

## 10. Rủi ro phải chặn

**Leakage.** ⚠️ *Mục này đã được thay thế bằng số liệu đã kiểm chứng —
xem `docs/dataset_v2_cleaning.md` §1 và §6.*

Không chỉ "cùng một cuốn sách": **text của v2 được copy từ chính nomfoundation.org**
(chứng minh bằng marker `mat-chu` và folio marker giống hệt). Corpus crawl **chứa
nguyên văn ~83% ký tự của test set**.

Đã có blocklist chính xác thay cho phỏng đoán:

```
data/processed/ner_clean/gazetteer_blocklist.txt   2,038 câu CẤM (chạm dev/test)
   -> còn 14,283/16,321 câu (87.5%) dùng được để mine / pretrain
```

Sinh bởi `scripts/map_records_to_source.py`. Lưu ý phân biệt: gazetteer dựng từ
**nhãn train** thì KHÔNG phải leakage; chỉ có nhãn test, hoặc mine trên câu crawl
thuộc dev/test, mới là leakage.

**Nhập nhằng entity 1–2 chữ.** Không được string-replace mù:
- `宋` = triều Tống (ORG) / họ Tống (PER)
- `帝` = TITLE / nằm trong `黄帝`, `帝來` (PER)
- `越` = ORG (nước) hay LOC (đất)? — **guideline chưa chốt**
- `元` = triều Nguyên / "nguyên" trong `元年`

→ longest-match ưu tiên, và gán PER/LOC **trước** khi gán TITLE/ORG 1 chữ.

**Dị thể tự.** Gazetteer khớp chuỗi chính xác sẽ trượt: `髙帝` (髙 dị thể của 高),
`清華`/`清化` cùng là Thanh Hoa, chữ Nôm hiếm bị render thành `?`.
**Cần chuẩn hoá dị thể tự trước khi khớp**, nếu không độ phủ thực tế thấp hơn con số đo được.

**Ghi nhớ thay vì học mẫu.** Weak-label PER bằng gazetteer → model học "chuỗi 鄭松 là PER"
thay vì "họ + 2 chữ sau `遣`/`命`/`封` là PER". Chỉ lộ ra khi test trên văn bản khác.

---

## 11. Quyết định còn treo

1. **Mining trên toàn bộ 19,909 câu, hay loại phần giao với test v2?** (leakage)
2. **Quy tắc biên TITLE** — gold v2 gộp `少保豪郡公` thành **một** span; cần xác nhận
   `國子監司業` có cùng quy ước không. Quyết định lexicon chứa **đơn vị nguyên tử** hay
   **chuỗi đầy đủ**. Chặn đường nhiều nhất vì TITLE là ưu tiên 1.
3. **`越` là ORG hay LOC** — guideline v1.0 đã chốt miếu/thụy hiệu=PER và triều đại=ORG,
   nhưng chưa chạm điểm này.
4. **Mục tiêu cuối: chỉ DVSKTT, hay Hán văn Việt Nam nói chung?** Quyết định có ưu tiên
   khả năng chuyển giao hay không.

---

## Phụ lục: tiền xử lý bắt buộc trước khi build

1. Bỏ 4 section phụ Mạc (`80`, `77`, `71`, `75`) → 19,909 câu
2. Quyết định xử lý `[...]` (601 câu) — tách chú thích ra khỏi câu, nếu không lệch offset
3. Bỏ/đánh dấu 63 câu có `?`/`>` (glyph hỏng) và 11 câu dính `mat-chu`
4. `〇` giữ lại hoặc bỏ nhất quán — là dấu ngắt đoạn bản gốc, không phải rác
5. Chuẩn hoá dị thể tự

---

## 12. Build TITLE/ORG/LOC (2026-08-10)

Script: `scripts/build_gazetteer.py`. Output: `data/processed/gazetteer/{title,org,loc}.jsonl` + `report.json`.

**Quyết định phạm vi mining:** chỉ dùng entity gold **train** (không đụng dev/test) —
lexicon dùng làm feature đánh giá trên dev/test, đưa entity dev/test vào sẽ tự thổi phồng
số liệu chính nó đo. Mục 11 câu hỏi #1 coi như chốt theo hướng an toàn nhất.
Guideline v1.0 đã chốt luôn `越`=ORG (câu hỏi #3) và convention TITLE lấy trọn cụm
(câu hỏi #2) — 3/4 quyết định treo ở mục 11 nay đã giải quyết qua guideline/lựa chọn này;
câu hỏi #4 (phạm vi DVSKTT-only vs Hán văn VN nói chung) vẫn treo, hiện build chỉ nhắm DVSKTT.

**TITLE** — gold train (1,358 surface) + cross-product tiền tố×hậu tố (ngưỡng ≥5 lần xuất
hiện làm tiền/hậu tố trong entity gold đa âm tiết → 78 tiền tố × 40 hậu tố → 2,998 ứng viên,
`needs_review=true`). Coverage trên dev (mention-level, exact match): **73.4%**.

**ORG** — chỉ gold train (629 surface), không mở rộng tự động (1-char ORG cần context-gate,
để dành cho lúc dùng làm feature, không phải lúc build list). Coverage dev: **67.0%**.

**LOC** — mining qua căn chỉnh âm tiết phiên âm ↔ chữ Hán trên toàn bộ crawl, **trừ 2,037 câu
blocklist** (leakage guard), cộng vá hậu tố loại từ (§4). Sau khi lọc nhiễu PER (span mở đầu
bằng 1 trong 30 chữ họ phổ biến, span trùng surface PER/TITLE/ORG gold train, hoặc kết thúc
bằng `宗` — miếu hiệu): còn **6,967 unique / 21,042 mention**. Coverage dev: **54.9%**.
Vẫn còn lọt một số nhiễu khó lọc bằng rule đơn giản (ví dụ `元至元` — gộp quốc hiệu + niên hiệu
liền kề, đúng như lỗi hệ thống đã nêu ở §4) — do vậy dùng LOC list này **có `needs_review`
theo tần suất** (freq<2 → true), không nên coi là danh sách sạch tuyệt đối.

**Chưa làm ở lần build này:** mở rộng ORG bằng mining (hiện chỉ gold), curate tay lexicon
TITLE (thay vì cross-product ngưỡng tần suất), context-gate cho ORG/entity 1 chữ khi dùng làm
feature thật trong model. Đây là các việc downstream — bản thân gazetteer build coi như **xong
scope TITLE/ORG/LOC như đã lên kế hoạch ở mục 9**.
