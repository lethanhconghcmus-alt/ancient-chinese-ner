"""Pretrain pilot: continued causal-LM pretrain LoRA tren corpus raw text.

Dung chung cho ca 3 kich ban cua thi nghiem Qwen3.6-27B (xem trao doi voi
user 2026-08-26): pretrain tren corpus DVSKTT rieng, hoac tren corpus
TQ-merge (CHisIEC+C-CLUE+CMAG), tuy PRETRAIN_CORPUS truyen vao.

BASE_MODEL / PRETRAIN_CORPUS / RUN_TAG doc tu env var, xem kernel notebook.
Luu checkpoint LoRA vao /kaggle/working/run_<tag>/checkpoints/pretrain_final
de kernel SFT sau do load lai (qua Kaggle dataset tao tu output cua kernel nay).
"""
import os, sys, time, json, warnings, random
warnings.filterwarnings("ignore")

import torch

BASE_MODEL = os.environ.get("PRETRAIN_BASE_MODEL", "unsloth/qwen2.5-7b-unsloth-bnb-4bit")
CORPUS_PATH = os.environ["PRETRAIN_CORPUS"]  # duong dan file .txt, 1 cau/dong
RUN_TAG = os.environ.get("PRETRAIN_RUN_TAG", "pretrain")

CONFIG = {
    "ckpt_dir": f"/kaggle/working/run_{RUN_TAG}/checkpoints",
    "log_dir":  f"/kaggle/working/run_{RUN_TAG}/logs",
    "max_seq_len":  512,
    "lora_rank":    16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,
    "lr":         5e-5,
    "epochs":     1,
    "batch":      int(os.environ.get("PRETRAIN_BATCH", "1")),
    "grad_accum": 4,
    "sample_size": int(os.environ.get("PRETRAIN_SAMPLE_SIZE", "5000")),
}
for d in ["ckpt_dir", "log_dir"]:
    os.makedirs(CONFIG[d], exist_ok=True)


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


logger = Logger(f"{CONFIG['log_dir']}/pretrain.log")
logger.log(f"BASE_MODEL={BASE_MODEL} CORPUS_PATH={CORPUS_PATH} RUN_TAG={RUN_TAG}")
logger.log(f"PyTorch: {torch.__version__}")

if not torch.cuda.is_available():
    logger.log("FATAL: no CUDA device visible.")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
cap = f"{props.major}.{props.minor}"
logger.log(f"GPU: {torch.cuda.get_device_name(0)} | compute capability {cap} | VRAM {props.total_memory/1024**3:.1f}GB")

if props.major < 7:
    logger.log(f"FATAL: GPU compute capability {cap} (< 7.0, P100 hoac cu hon) "
               f"khong tuong thich transformers/torch moi can cho Qwen3.6. Can T4 tro len.")
    sys.exit(1)

import transformers, peft, bitsandbytes
logger.log(f"transformers={transformers.__version__} peft={peft.__version__} bitsandbytes={bitsandbytes.__version__}")

from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import default_data_collator, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.optim as optim
from huggingface_hub import hf_hub_download

logger.log(f"Loading base model: {BASE_MODEL}")
try:
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
# Qwen3.6 la model da phuong thuc (text/image/video) -> unsloth co the tra ve
# mot processor bao ngoai tokenizer text thuan. Goi truc tiep processor voi
# text se co the bi hieu nham la image source va crash. Unwrap ve tokenizer
# text neu co.
if hasattr(tokenizer, "tokenizer"):
    logger.log("Tokenizer la processor da phuong thuc, unwrap ve .tokenizer")
    tokenizer = tokenizer.tokenizer

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_rank"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
logger.log("Model + LoRA adapter ready.")
model.print_trainable_parameters()

with open(CORPUS_PATH, encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
random.seed(42)
random.shuffle(lines)
lines = lines[:CONFIG["sample_size"]]
logger.log(f"Corpus sentences used: {len(lines):,}")

texts = [l + tokenizer.eos_token for l in lines]
dataset = Dataset.from_dict({"text": texts})


def tokenize_fn(examples):
    result = tokenizer(
        examples["text"], truncation=True,
        max_length=CONFIG["max_seq_len"], padding="max_length",
    )
    result["labels"] = [
        [(tok if mask == 1 else -100) for tok, mask in zip(ids, attn)]
        for ids, attn in zip(result["input_ids"], result["attention_mask"])
    ]
    return result


tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=2)
logger.log(f"Tokenized: {len(tokenized):,} examples")

EPOCHS = CONFIG["epochs"]
BATCH = CONFIG["batch"]
GRAD_ACCUM = CONFIG["grad_accum"]

loader = DataLoader(tokenized, batch_size=BATCH, shuffle=True, collate_fn=default_data_collator)
total_steps = len(loader) * EPOCHS
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps,
)
logger.log(f"Total steps: {total_steps:,}")

model.train()
global_step = 0
t0 = time.time()

for epoch in range(EPOCHS):
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        global_step += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        if torch.isnan(outputs.loss):
            logger.log(f"NaN loss at step {global_step}! Stopping.")
            break
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        total_loss += outputs.loss.item()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if global_step % 50 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - t0
            eta = elapsed / global_step * (total_steps - global_step)
            logger.log(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | Loss {avg:.4f} | ETA {eta/60:.1f}m")
    avg_epoch = total_loss / len(loader)
    logger.log(f"Epoch {epoch+1} done | Avg Loss: {avg_epoch:.4f}")

final_path = f"{CONFIG['ckpt_dir']}/pretrain_final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
logger.log(f"Saved: {final_path}")
print(f"\n>> [{RUN_TAG}] Pretrain done. Checkpoint: {final_path}")
