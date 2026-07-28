# DVSKTT NER — Qwen2.5-7B + LoRA

Named Entity Recognition cho **Hán văn Việt Nam cổ**, huấn luyện và đánh giá trên
**Đại Việt Sử Ký Toàn Thư (大越史記全書)**. Fine-tune Qwen2.5-7B (4-bit, Unsloth)
với LoRA theo hướng generative NER: model sinh lại câu với entity gắn tag inline
`{entity|TYPE}`.

**Entity types:** `PER` (人名) · `LOC` (地名) · `ORG` (机构名) · `DTM` (时间) · `TITLE` (官职)

## Pipeline

```
                 ┌────────────────────────┐
                 │  Han corpus (merged)   │
                 │  DVSKTT 1,311 lines +  │
                 │  Chinese classics      │
                 └───────────┬────────────┘
                             │ sample 5,000 lines
                             ▼
┌──────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Qwen2.5-7B   │──▶│ 1. Continued         │──▶│ 2. SFT (NER data)    │
│ 4-bit +LoRA  │   │    Pretraining (CLM) │   │    loss chỉ trên     │
│ (Unsloth)    │   │    pretrain_final    │   │    Output span       │
└──────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                          │ sft_best
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                   ┌──────────────────┐   ┌──────────────────────┐
                                   │ 3. Evaluate      │   │ 4. Evaluate + RAG    │
                                   │    (zero-shot    │   │    few-shot (TF-IDF  │
                                   │     prompting)   │   │    hoặc BGE-M3)      │
                                   └──────────────────┘   └──────────────────────┘
```

Input SFT dạng instruction (Alpaca-style): Instruction (mô tả task + entity types)
→ Input (câu Hán văn) → Output (câu gắn tag `{entity|TYPE}`).

## Kết quả

> ⚠️ **2026-07-29 — dataset đã rebuild (v2), mọi kết quả v1 vô hiệu.**
> Audit dataset v1 phát hiện 3 lỗi nghiêm trọng: (1) nhãn nhiễu nặng do pipeline
> convert cũ (audit tay 100 entity: chỉ ~13% đúng hoàn toàn); (2) ~10% câu dính
> markup annotation trong input; (3) corpus bị nhân bản ~3.7x do concat các file
> annotation tích lũy → **83.6% câu test trùng nguyên văn với train** (leakage).
> F1 v1 (E2 0.7507, GuwenBERT-CRF 0.8200) chủ yếu đo memorization.
> Dataset v2 rebuild từ Excel annotation gốc bằng `scripts/rebuild_dataset.py`
> (0 leakage, 0 punct-in-entity) — mọi experiment cần chạy lại trên v2.

| Exp | Setup | seq_len | rank | P | R | **F1** | Ghi chú |
|-----|-------|---------|------|------|------|--------|---------|
| E1 | Zero-shot (base model) | 512 | — | — | — | _chờ chạy v2_ | baseline prompting |
| E2 | Pretrain + SFT | 512 | 16 | — | — | _chờ chạy v2_ | v1: 0.7507 (invalid) |
| E3 | E2 + RAG 1-shot TF-IDF | 1024 | 16 | — | — | _chờ chạy v2_ | v1: 0.6888 (invalid) |
| E4 | Pretrain + SFT | 1024 | 16 | — | — | _chưa chạy_ | fix truncation câu dài |
| E5 | Pretrain + SFT | 1024 | 32 | — | — | _chưa chạy_ | tăng capacity adapter |
| — | GuwenBERT + CRF | — | — | — | — | _chờ chạy v2_ | v1: 0.8200 (cùng leakage) |

Kết quả chi tiết (per-type P/R/F1, error analysis, predictions) lưu trong
`results/<experiment>/`.

## Cấu trúc repo

```
ancient-chinese-ner/
├── configs/            # YAML config từng experiment (base.yaml = default chung)
├── data/raw/
│   ├── han_pretrain/   # corpus Hán văn cho continued pretraining
│   └── ner_sft/        # train/dev/test.jsonl v2 (1284/160/160 records)
├── notebooks/          # Colab notebooks (pipeline gốc)
├── scripts/            # CLI entry points (run_pretrain/sft/evaluate/rag)
├── src/                # toàn bộ logic: config, data, train, evaluate, rag
└── results/            # metrics + error analysis từng experiment
```

## Reproduce

### Setup

```bash
git clone https://github.com/lethanhconghcmus-alt/ancient-chinese-ner.git
cd ancient-chinese-ner
pip install -r requirements.txt
```

Cần GPU ~16GB VRAM (Colab T4 chạy được). Checkpoint LoRA đã train host trên
Kaggle (`thnhcngl/dvsktt-sft-best-checkpoint`) — cần `KAGGLE_USERNAME`/`KAGGLE_KEY`
nếu muốn tải về thay vì tự train.

### E4 — SFT seq_len=1024 (từ đầu đến cuối)

```bash
# 1. Continued pretraining trên corpus Hán văn
python scripts/run_pretrain.py --config configs/e4_sft_1024.yaml

# 2. SFT (tự động khởi đầu từ pretrain_final)
python scripts/run_sft.py --config configs/e4_sft_1024.yaml

# 3. Evaluate trên test set (mặc định dùng sft_best)
python scripts/run_evaluate.py --config configs/e4_sft_1024.yaml
```

Session chết giữa chừng? Chạy lại đúng lệnh cũ với `--resume` (train) —
evaluation tự resume theo state đã lưu.

### E1 — Zero-shot baseline

```bash
python scripts/run_evaluate.py --config configs/e1_zero_shot.yaml --checkpoint base
```

### E3 — RAG few-shot

```bash
python scripts/run_rag.py --config configs/e3_rag_tfidf.yaml \
    --checkpoint /path/to/sft_best --retriever tfidf --shots 1
# Dense retrieval (BGE-M3):
python scripts/run_rag.py --config configs/e3_rag_tfidf.yaml \
    --checkpoint /path/to/sft_best --retriever bge --shots 1
```

### Eval checkpoint tải từ Kaggle (chỉ chứa LoRA adapter)

```bash
python scripts/run_evaluate.py --config configs/e2_sft_512.yaml \
    --checkpoint base --adapter /path/to/kaggle-checkpoint
```

## Dataset

- **NER SFT v2** (`data/raw/ner_sft/`): 1,604 records unique từ DVSKTT
  (~23K entities), rebuild từ Excel annotation gốc (anh Thiều) bằng
  `scripts/rebuild_dataset.py`. Split train/dev/test = 1,284/160/160,
  **không leakage** (dedup tuyệt đối trước khi split). Trong đó 657 records
  từ inline markup `"entity"(TYPE)` (tin cậy cao), 947 records từ NER dict
  projection. BIO tương ứng + report: `data/processed/ner_clean/`.
  (v1 cũ 6,987 câu thực chất chỉ có 1,961 câu unique nhân bản 3.7x — xem
  warning ở mục Kết quả; file v1 còn trong git history.)
- **Pretrain corpus** (`data/raw/han_pretrain/dvsktt_han_merged.txt`): 1,311 dòng
  DVSKTT (giữ 100%) + Hán văn cổ Trung Quốc, sample tổng 5,000 dòng mỗi run.

## Ghi chú kỹ thuật

- **Loss masking**: SFT chỉ tính loss trên phần `### Output:` — prompt template
  và padding mask về `-100`.
- **Custom training loop** (không dùng `Trainer`): resume từ step bất kỳ,
  auto-save checkpoint về Drive mỗi N steps, NaN detection (SFT: giảm lr rồi
  chạy tiếp, tối đa 3 lần).
- **Batched generation**: `padding_side='left'` (decoder-only), exact-match
  metric trên cặp `(surface, type)`.
- Notebooks trong `notebooks/` là pipeline Colab gốc; logic đã được tách dần
  sang `src/` — chạy mới nên dùng `scripts/`.
