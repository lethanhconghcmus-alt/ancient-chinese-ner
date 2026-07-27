# E1 — Zero-shot baseline

Base model `unsloth/qwen2.5-7b-unsloth-bnb-4bit`, không train, chỉ prompting.

**Chưa chạy.** Reproduce:

```bash
python scripts/run_evaluate.py --config configs/e1_zero_shot.yaml --checkpoint base
```
