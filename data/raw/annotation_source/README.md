# Nguồn annotation gốc

Excel gán nhãn của anh Thiều (cùng lab) — **nguồn gốc của toàn bộ dataset v2**.
`scripts/rebuild_dataset.py` đọc thẳng từ đây.

Trước 2026-08-04 chỉ nằm ở `D:\bio_source\NER_zip\NER\NER_AnhThieu` (giải nén từ
`NER.ZIP` trong Downloads) — mất thư mục đó là dataset không tái tạo được.

## Hai loại file

| Nhóm | Đặc điểm |
|---|---|
| `NER-1-100` … `NER-901-1000` | 782 dòng có inline markup `"X"(TYPE)` sạch |
| `NER 1288-*` | ~1,148 dòng chỉ có cột NER dict. **File TÍCH LUỸ, chồng lấn nhau** |

Chính việc concat các file tích luỹ này gây **nhân bản 3.7×** ở corpus v1
(7,301 câu nhưng chỉ ~1,961 unique) => **83.6% test trùng nguyên văn train**.
`rebuild_dataset.py` khử trùng tuyệt đối trước khi split nên v2 không còn lỗi này.

## Lưu ý về nguồn văn bản

Phần chữ Hán trong các Excel này được **copy từ nomfoundation.org**, không phải
số hoá độc lập — chứng minh ở `docs/dataset_v2_cleaning.md` §1. Vì vậy mới còn
sót marker `mat-chu` (= "mất chữ") và folio marker `[8a*02*02]` trong text.

`annotations_consolidated.jsonl` — bản hợp nhất 1,930 dòng unique (~342K ký tự)
rút từ các Excel trên.
