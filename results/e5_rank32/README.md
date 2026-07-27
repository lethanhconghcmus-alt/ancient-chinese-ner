# E5 — LoRA rank=32, seq_len=1024

Giả thuyết: tăng capacity adapter (rank 16 → 32, alpha 32 → 64) giúp học tốt hơn
các entity hiếm (ORG, TITLE).

**Chưa chạy.** Reproduce:

```bash
python scripts/run_pretrain.py --config configs/e5_rank32.yaml
python scripts/run_sft.py      --config configs/e5_rank32.yaml
python scripts/run_evaluate.py --config configs/e5_rank32.yaml
```
