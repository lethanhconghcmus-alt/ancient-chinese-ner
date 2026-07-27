# E4 — SFT seq_len=1024

Giả thuyết: seq 512 truncate câu dài của DVSKTT → mất entity ở đuôi câu.

**Chưa chạy.** Reproduce:

```bash
python scripts/run_pretrain.py --config configs/e4_sft_1024.yaml
python scripts/run_sft.py      --config configs/e4_sft_1024.yaml
python scripts/run_evaluate.py --config configs/e4_sft_1024.yaml
```
