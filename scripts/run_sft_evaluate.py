import os, sys, time, json, warnings, re
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")

import torch
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
print("PyTorch:", torch.__version__)

import numpy as np, pandas as pd
print("NumPy:", np.__version__, "| Pandas:", pd.__version__)

from unsloth import FastLanguageModel
from peft import PeftModel
from datasets import Dataset
from transformers import default_data_collator, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.optim as optim

BASE_MODEL = "unsloth/qwen2.5-7b-unsloth-bnb-4bit"
SFT_DATA_DIR = "/kaggle/working/ancient-chinese-ner/data/raw/ner_sft"
PRETRAIN_CKPT_SRC = "/kaggle/input/datasets/thnhcngl/dvsktt-pretrain-final-ckpt"

CONFIG = {
    "work_dir":   "/kaggle/working/run",
    "ckpt_dir":   "/kaggle/working/run/checkpoints",
    "result_dir": "/kaggle/working/run/results",
    "log_dir":    "/kaggle/working/run/logs",

    "max_seq_len":  512,
    "lora_rank":    16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,

    "sft_lr":         5e-5,
    "sft_epochs":     3,
    "sft_batch":      1,
    "sft_grad_accum": 4,

    "eval_batch":     4,
    "max_new_tokens": 900,
    "save_every":     200,
}
for d in ["work_dir", "ckpt_dir", "result_dir", "log_dir"]:
    os.makedirs(CONFIG[d], exist_ok=True)

print("Data dir:", SFT_DATA_DIR, os.listdir(SFT_DATA_DIR))
print("Pretrain ckpt:", PRETRAIN_CKPT_SRC, os.listdir(PRETRAIN_CKPT_SRC))


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.start = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n===== Session started: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def log(self, msg, also_print=True):
        elapsed = time.time() - self.start
        line = f"[{elapsed:>8.1f}s] {msg}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        if also_print:
            print(line, flush=True)


logger = Logger(f"{CONFIG['log_dir']}/sft.log")

# PRETRAIN_CKPT_SRC la LoRA adapter-only (adapter_config.json +
# adapter_model.safetensors, khong co config.json cua full model) vi no duoc
# luu bang model.save_pretrained() tren mot PeftModel. Ban unsloth==2024.8
# (buoc phai dung vi torch 2.3.1/P100) khong tu nhan dien duoc adapter-only
# dir qua from_pretrained (khac ban moi) -> tu lam 2 buoc: load base model
# roi ap adapter pretrain bang PeftModel, giu is_trainable=True de train tiep.
logger.log(f"Loading base model: {BASE_MODEL}")

# unsloth's loader swallows the real AutoConfig error and re-raises a generic
# "Can't load the configuration" message. Probe the actual failure first so
# the real cause (network/auth/hub) shows up in the log instead of a guess.
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=BASE_MODEL, filename="config.json")
    logger.log(f"Diagnostic hf_hub_download OK: {p}")
except Exception as e:
    import traceback
    logger.log(f"Diagnostic hf_hub_download FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=CONFIG["max_seq_len"],
    load_in_4bit=True,
    dtype=None,
)
logger.log(f"Applying pretrain LoRA adapter: {PRETRAIN_CKPT_SRC}")
model = PeftModel.from_pretrained(model, PRETRAIN_CKPT_SRC, is_trainable=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
logger.log("Model loaded!")
model.print_trainable_parameters()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


train_data = load_jsonl(f"{SFT_DATA_DIR}/train.jsonl")
logger.log(f"Train records: {len(train_data):,}")


def make_prefix(record):
    return (
        f"### Instruction:\n{record['instruction']}\n\n"
        f"### Input:\n{record['input']}\n\n"
        f"### Output:\n"
    )


def format_prompt(record):
    return make_prefix(record) + record["output"] + tokenizer.eos_token


train_prefixes = [make_prefix(r) for r in train_data]
train_texts = [format_prompt(r) for r in train_data]
train_dataset = Dataset.from_dict({"text": train_texts, "prefix": train_prefixes})


def tokenize_sft(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_seq_len"],
        padding="max_length",
    )
    labels = [ids.copy() for ids in result["input_ids"]]
    for i, prefix in enumerate(examples["prefix"]):
        prefix_len = len(tokenizer(prefix, truncation=True, max_length=CONFIG["max_seq_len"])["input_ids"])
        for j in range(min(prefix_len, len(labels[i]))):
            labels[i][j] = -100
        for j in range(len(labels[i])):
            if result["attention_mask"][i][j] == 0:
                labels[i][j] = -100
    result["labels"] = labels
    return result


train_tokenized = train_dataset.map(tokenize_sft, batched=True, remove_columns=["text", "prefix"], num_proc=2)
sft_collator = default_data_collator
logger.log(f"SFT tokenized: {len(train_tokenized):,} examples")

EPOCHS = CONFIG["sft_epochs"]
BATCH = CONFIG["sft_batch"]
GRAD_ACCUM = CONFIG["sft_grad_accum"]
SAVE_EVERY = CONFIG["save_every"]

sft_loader = DataLoader(train_tokenized, batch_size=BATCH, shuffle=True, collate_fn=sft_collator)
total_steps = len(sft_loader) * EPOCHS
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["sft_lr"])
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.05),
    num_training_steps=total_steps,
)

logger.log(f"SFT total steps: {total_steps:,}")

model.train()
global_step = 0
best_loss = float("inf")

for epoch in range(EPOCHS):
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(sft_loader):
        global_step += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)

        if torch.isnan(outputs.loss):
            logger.log(f"NaN loss at step {global_step}! Stopping epoch.")
            break

        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        total_loss += outputs.loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if global_step % 100 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - logger.start
            eta = elapsed / global_step * (total_steps - global_step)
            logger.log(
                f"Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | "
                f"Loss: {avg:.4f} | ETA: {eta/3600:.2f}h"
            )

        if global_step % SAVE_EVERY == 0:
            avg_loss = total_loss / (step + 1)
            ckpt_path = f"{CONFIG['ckpt_dir']}/sft_step{global_step}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            logger.log(f"Checkpoint saved: {ckpt_path}")

    avg_epoch = total_loss / len(sft_loader)
    logger.log(f"Epoch {epoch+1} done | Avg Loss: {avg_epoch:.4f}")

    if avg_epoch < best_loss:
        best_loss = avg_epoch
        best_path = f"{CONFIG['ckpt_dir']}/sft_best"
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        logger.log(f"Best model saved: {best_path} (loss: {best_loss:.4f})")

logger.log(f"SFT complete! Best loss: {best_loss:.4f}")

# ── Evaluate ────────────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"
logger2 = Logger(f"{CONFIG['log_dir']}/evaluate.log")

test_data = load_jsonl(f"{SFT_DATA_DIR}/test.jsonl")
logger2.log(f"Test records: {len(test_data):,}")

ENTITY_TYPES = ["PER", "LOC", "ORG", "DTM", "TITLE"]


def parse_entities(text):
    return set(re.findall(r"\{([^|]+)\|([^}]+)\}", text))


def make_prompt(record):
    return (
        f"### Instruction:\n{record['instruction']}\n\n"
        f"### Input:\n{record['input']}\n\n"
        f"### Output:\n"
    )


def generate_batch(records):
    prompts = [make_prompt(r) for r in records]
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=CONFIG["max_seq_len"], padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(o[input_len:], skip_special_tokens=True).strip() for o in outputs]


def compute_prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f


overall = {"tp": 0, "fp": 0, "fn": 0}
per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES}
predictions = []

BATCH_SIZE = CONFIG["eval_batch"]
for batch_start in range(0, len(test_data), BATCH_SIZE):
    batch = test_data[batch_start:batch_start + BATCH_SIZE]
    batch_preds = generate_batch(batch)

    for record, pred_text in zip(batch, batch_preds):
        gold_ents = parse_entities(record["output"])
        pred_ents = parse_entities(pred_text)

        overall["tp"] += len(gold_ents & pred_ents)
        overall["fp"] += len(pred_ents - gold_ents)
        overall["fn"] += len(gold_ents - pred_ents)

        for etype in ENTITY_TYPES:
            g = {e for e in gold_ents if e[1] == etype}
            p = {e for e in pred_ents if e[1] == etype}
            per_type[etype]["tp"] += len(g & p)
            per_type[etype]["fp"] += len(p - g)
            per_type[etype]["fn"] += len(g - p)

        predictions.append({
            "input": record["input"], "gold": record["output"], "pred": pred_text,
            "n_gold": len(gold_ents), "n_pred": len(pred_ents),
            "n_correct": len(gold_ents & pred_ents),
        })

    done = batch_start + len(batch)
    elapsed = time.time() - logger2.start
    eta = elapsed / done * (len(test_data) - done)
    _, _, f1_now = compute_prf(**overall)
    logger2.log(f"[{done:>3}/{len(test_data)}] Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | F1: {f1_now:.4f}")

logger2.log(f"Evaluate done! Total: {(time.time()-logger2.start)/60:.1f} min")

print("=" * 55)
print(f"{'Entity':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)
for etype in ENTITY_TYPES:
    p, r, f = compute_prf(**per_type[etype])
    print(f"{etype:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
print("-" * 55)
p, r, f = compute_prf(**overall)
print(f"{'Overall':<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
print("=" * 55)
print(f"\n>> Overall F1: {f:.4f}")

final_results = {
    "overall": dict(zip(["precision", "recall", "f1"], compute_prf(**overall))),
    "per_type": {t: dict(zip(["precision", "recall", "f1"], compute_prf(**per_type[t]))) for t in ENTITY_TYPES},
    "n_evaluated": len(predictions),
    "predictions": predictions,
}
out_path = f"{CONFIG['result_dir']}/eval_final_v2_1.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print(f"Saved: {out_path}")
