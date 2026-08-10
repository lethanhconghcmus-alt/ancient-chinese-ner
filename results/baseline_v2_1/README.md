# Baseline GuwenBERT-CRF — rerun trên test v2.1 (2026-08-10)

Rerun sau khi áp gold review lên test set (xem `data/raw/annotation_source/gold_review_sheet.xlsx`,
commit `572ab8f`). Dataset Kaggle `thnhcngl/dvsk-data` đã update: train/dev v2-clean +
test v2.1. Notebook: `chisiec_dvsk_pipeline.ipynb` (kernel `thnhcngl/dvsktt-guwenbert-crf-v2-1`,
version 6, COMPLETE).

## Kết quả

| | Precision | Recall | F1 |
|---|---|---|---|
| **micro avg (test v2.1)** | 0.6554 | 0.6492 | **0.6522** |
| PER | 0.730 | 0.774 | 0.751 |
| TITLE | 0.643 | 0.666 | 0.654 |
| LOC | 0.593 | 0.626 | 0.609 |
| DTM | 0.656 | 0.497 | 0.565 |
| ORG | 0.544 | 0.489 | 0.515 |

Dev F1 tốt nhất: 0.6255 (epoch 25/33 — early stopping trigger đúng nghĩa, patience=8
hết hạn không cải thiện thêm, không bị cắt cứng bởi epoch cap).

**CHisIEC anchor** (dataset ngoại lai, không đổi): test micro F1 = 0.9131 (so với 0.9209
gốc) — xác nhận pipeline chạy đúng, chênh lệch nhỏ do khác biệt torch/transformers version
(phải pin `torch==2.3.1` + `transformers==4.42.4` vì Kaggle base image mới không còn hỗ trợ
GPU P100 với torch bản mới nhất).

## So với v1 (hỏng, leakage 83.6%)

| | v1 (leak) | v2.1 (clean) |
|---|---|---|
| GuwenBERT-CRF test F1 | 0.8124 | **0.6522** |

F1 tụt ~16 điểm — đúng như dự đoán trong `docs/project_status.md`: số liệu v1 chủ yếu là
memorization do train/test trùng lặp, không phải năng lực NER thật. Đây là bằng chứng
chính cho đóng góp của paper (data quality), không phải thất bại của model.

## Training log

`dvsk_train_history.json` — lịch sử train đầy đủ qua 3 giai đoạn resume:
- Step 2: transfer từ CHisIEC, 30 epoch cap
- Step 3: resume, lr=5e-6, 10 epoch cap
- Step 3b: resume thêm, lr=2e-6, 60 epoch cap, patience=8 — dừng tự nhiên ở epoch 33
  (best epoch 25), không cần resume thêm nữa.
