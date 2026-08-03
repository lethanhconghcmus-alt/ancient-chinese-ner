# Thống kê corpus — dùng cho paper

*Sinh tự động bởi `scripts/corpus_stats.py`. Đây là số của **v2-clean**, thay cho mọi số v1 trong bản thảo cũ.*

## ⚠️ Con số v1 phải thay

| Bản thảo cũ (v1) | Thực tế (v2-clean) |
|---|---|
| 7,301 câu | **1,597 record** |
| 1,291,822 token | **263,638 ký tự Hán** |
| 107,105 entity | **22,858 entity** |

Con số v1 bị thổi phồng do concat các file Excel tích luỹ ⇒ **nhân bản 3.7×** (7,301 ÷ 3.7 ≈ 1,973 ≈ số record unique thật). Chính việc này gây **83.6% test trùng nguyên văn train** ở v1.

## Tổng quan

| Split | Record | Ký tự | Entity | Entity unique | Dài record (min/med/max) |
|---|---|---|---|---|---|
| train | 1,277 | 210,927 | 18,287 | 6,975 | 12/169/339 |
| dev | 160 | 26,013 | 2,372 | 1,533 | 27/168/224 |
| test | 160 | 26,698 | 2,199 | 1,404 | 58/169/241 |
| **Tổng** | **1,597** | **263,638** | **22,858** | — | — |

## Phân bố nhãn

| Type | train | dev | test | Tổng | % |
|---|---|---|---|---|---|
| PER | 5,905 | 870 | 701 | **7,476** | 32.7% |
| LOC | 3,905 | 399 | 465 | **4,769** | 20.9% |
| ORG | 1,974 | 218 | 245 | **2,437** | 10.7% |
| TITLE | 4,524 | 635 | 507 | **5,666** | 24.8% |
| DTM | 1,979 | 250 | 281 | **2,510** | 11.0% |
| **Tổng** | **18,287** | **2,372** | **2,199** | **22,858** | 100% |

## Độ phủ so với nguyên bản ĐVSKTT

- Nguyên bản (nomfoundation.org, đã bỏ 4 section phụ Mạc trùng lặp): **19,909 câu · 369,637 ký tự**
- Corpus có nhãn: **263,638 ký tự = 71.3%**
- Chưa gán nhãn: **105,999 ký tự**, ước **~9,190 entity** (theo mật độ 0.087 entity/ký tự)

### 10 section gần như vắng hoàn toàn (<5%) — đưa vào Limitation

| Section | Câu | Phủ |
|---|---|---|
| `57-Thai-Tong-Cao-Hoang-De` | 1,068 | 0.0% |
| `58-Thanh-Tong-Van-Hoang-De` | 808 | 0.2% |
| `59-Nhan-Tong-Tuyen-Hoang-De` | 554 | 0.0% |
| `32-Nhan-Tong-Hoang-De` | 376 | 0.3% |
| `33-Than-Tong-Hoang-De` | 243 | 0.4% |
| `54-Ky-hau-Tran` | 196 | 0.0% |
| `56-Ky-thuoc-Minh` | 60 | 0.0% |
| `52-Phu-Ho-Quy-Ly-Ho-Han-Thuong` | 57 | 0.0% |
| `13-Ky-Nam-Bac-phan-tranh` | 42 | 0.0% |
| `5-Ky-Trung-Nu-Vuong` | 12 | 0.0% |

Tổng **3,416 câu** không có annotation. Lớn nhất là `57-Thai-To-Cao-Hoang-De` (Lê Thái Tổ — khởi nghĩa Lam Sơn), quyển **lớn nhất bộ sách**.

## Chất lượng nhãn sau chuẩn hoá

| Split | Nhãn thiểu số trước | sau |
|---|---|---|
| train | 5.2% | **1.80%** |
| dev | 2.2% | **0.17%** |
| test | 2.76% | **0.86%** |

*Nhãn thiểu số = tỉ lệ mention mà chuỗi bề mặt của nó được gán type khác với type đa số của chính chuỗi đó. Chỉ số này **mù** với lỗi chỉ xuất hiện 1 lần và lỗi biên — không dùng làm bằng chứng duy nhất.*

## Leakage

- v1: **83.6%** test trùng nguyên văn train (nhân bản 3.7×)
- v2: dedup tuyệt đối theo chuỗi ⇒ 0 record trùng nguyên văn
- v2-clean: thêm bước loại **chồng lấn một phần** — bỏ **7 record train** chia sẻ đoạn văn duy nhất với dev/test
- **Hiện tại: 0 câu gốc duy nhất nằm ở nhiều split**

Phân biệt với câu công thức (`冬十月會試天下士人`…) vốn lặp lại thật trong nguyên bản — chúng trùng giữa các split là bình thường, không tính leakage.

## Tái lập

```
py scripts/rebuild_dataset.py                              # Excel -> v2
py scripts/clean_labels.py                                 # chuẩn hoá train+dev
py scripts/clean_labels.py --splits test --min-count 999999 \
       --report-name clean_report_test.json --review-name needs_review_test.json
py scripts/fix_split_leakage.py                            # loại chồng lấn
py scripts/map_records_to_source.py                        # map về nguyên bản
py scripts/corpus_stats.py                                 # sinh file này
```
