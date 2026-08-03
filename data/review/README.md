# Artifact rà soát gold test

Trước 2026-08-04 những file này chỉ nằm ở `D:\bio_source` — ngoài git, không
backup. Đưa vào repo để mỗi lần duyệt thêm là một mốc khôi phục được.

| File | Nội dung |
|---|---|
| `gold_review_sheet.xlsx` | 982 đề xuất rà gold test (160 record). Cột `DUYỆT`: OK / SỬA / BỎ |
| `review_proposals_rules.json` | 457 đề xuất do rule engine sinh |
| `manual_batch1..4.json` | 602 đề xuất rà tay |

982 = 1,059 đề xuất thô sau khi khử trùng 77 chỗ hai nguồn cùng đề xuất.

Phân bố: 429 ADD · 182 RETYPE · 176 DEL · 158 BOUNDARY · 32 NOTE · 5 FIXTEXT.

**Tiến độ tính đến 2026-08-04: 58/982 dòng (5.9%)** — 50 OK · 7 SỬA · 1 BỎ,
liên tục từ dòng #2 đến #59, chạm 5/154 record có đề xuất.

## Lưu ý

- **`review_rules.py` — script sinh ra `review_proposals_rules.json` — ĐÃ MẤT
  khỏi máy.** Chỉ còn output. Muốn chạy rule engine trên dev/train phải viết lại.
- Sau khi duyệt xong cần viết script apply -> sinh test-gold v2.1 -> validator.
  Script apply CHƯA có.
- 86% đề xuất được OK thẳng (50/58) => nên batch-accept nhóm có căn cứ chắc,
  chỉ soi tay RETYPE và BOUNDARY (hai nhóm sửa nhãn đã có, sai thì hại hơn ADD).
