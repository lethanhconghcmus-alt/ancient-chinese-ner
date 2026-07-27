"""Custom training loops cho pretrain + SFT (không dùng Trainer/SFTTrainer).

Lý do tự viết loop: cần resume từ step bất kỳ trên Colab free (session chết
giữa chừng), auto-save checkpoint về Drive, và NaN detection — những thứ
Trainer làm được nhưng khó kiểm soát trên môi trường hay bị ngắt.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

from .config import ExperimentConfig
from .data_utils import (
    build_pretrain_corpus,
    format_sft_prompt,
    load_jsonl,
    make_prompt_prefix,
)
from .logger import Logger
from .model_utils import save_checkpoint

# Số lần giảm lr tối đa khi gặp NaN trước khi bỏ cuộc
_MAX_NAN_RETRIES = 3


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_pretrain_dataset(tokenizer: Any, config: ExperimentConfig,
                           logger: Logger) -> Dataset:
    """Sample corpus Hán văn rồi tokenize cho causal LM."""
    pc = config.pretrain
    lines = build_pretrain_corpus(
        config.paths.han_corpus_path,
        sample_size=pc.sample_size,
        dvsktt_size=pc.dvsktt_size,
        seed=pc.seed,
    )
    logger.log(f'Corpus: {len(lines):,} lines '
               f'(DVSKTT: {pc.dvsktt_size}, sampled Han: {len(lines) - pc.dvsktt_size})')

    dataset = Dataset.from_dict({'text': lines})

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.model.max_seq_len,
            padding='max_length',
        )
        result['labels'] = [ids.copy() for ids in result['input_ids']]
        return result

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=['text'], num_proc=2)
    logger.log(f'Tokenized: {len(tokenized):,} examples')
    return tokenized


def build_sft_dataset(tokenizer: Any, config: ExperimentConfig,
                      logger: Logger,
                      train_path: Optional[str] = None) -> Dataset:
    """Load train.jsonl, build prompt và tokenize với loss mask.

    Chỉ tính loss trên phần "### Output:" — mask Instruction/Input/header và
    padding về -100 để model tập trung học pattern gắn tag NER, thay vì tốn
    sức học lại nguyên prompt template.
    """
    import os

    train_path = train_path or os.path.join(config.paths.sft_data_dir, 'train.jsonl')
    train_data = load_jsonl(train_path)
    logger.log(f'Train records: {len(train_data):,}')

    max_len = config.model.max_seq_len
    dataset = Dataset.from_dict({
        'text': [format_sft_prompt(r, tokenizer) for r in train_data],
        'prefix': [make_prompt_prefix(r) for r in train_data],
    })

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_len,
            padding='max_length',
        )
        labels = [ids.copy() for ids in result['input_ids']]
        for i, prefix in enumerate(examples['prefix']):
            prefix_len = len(tokenizer(prefix, truncation=True,
                                       max_length=max_len)['input_ids'])
            for j in range(min(prefix_len, len(labels[i]))):
                labels[i][j] = -100
            for j in range(len(labels[i])):
                if result['attention_mask'][i][j] == 0:
                    labels[i][j] = -100
        result['labels'] = labels
        return result

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=['text', 'prefix'], num_proc=2)
    logger.log(f'SFT tokenized: {len(tokenized):,} examples')
    return tokenized


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def pretrain_loop(model: Any, tokenizer: Any, config: ExperimentConfig,
                  logger: Logger,
                  tokenized: Optional[Dataset] = None) -> str:
    """Continued pretraining trên corpus Hán văn.

    Resume từ step bất kỳ (đọc ``pretrain_state.json`` cạnh log), auto-save
    mỗi ``save_every`` steps, dừng sớm khi gặp NaN loss.

    Returns:
        Đường dẫn checkpoint final.
    """
    pc = config.pretrain
    if tokenized is None:
        tokenized = build_pretrain_dataset(tokenizer, config, logger)

    resume_state = logger.load_state('pretrain_state')
    resume_step = resume_state.get('step', 0) if resume_state else 0
    logger.log(f'Resuming from step {resume_step}' if resume_step
               else 'Starting fresh pretrain')

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    dataloader = DataLoader(tokenized, batch_size=pc.batch_size,
                            shuffle=True, collate_fn=collator)
    total_steps = len(dataloader) * pc.epochs
    optimizer = optim.AdamW(model.parameters(), lr=pc.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * pc.warmup_ratio),
        num_training_steps=total_steps,
    )
    logger.log(f'Total steps: {total_steps:,} | Resume from: {resume_step}')

    model.train()
    global_step = 0
    for epoch in range(pc.epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            global_step += 1
            if global_step <= resume_step:
                if global_step % 200 == 0:
                    logger.log(f'Skipping step {global_step}/{resume_step}...')
                continue

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)

            if torch.isnan(outputs.loss):
                logger.log(f'NaN loss at step {global_step}! Stopping pretrain.')
                final_path = f'{config.paths.ckpt_dir}/pretrain_final'
                save_checkpoint(model, tokenizer, final_path,
                                {'step': global_step, 'status': 'nan_stop'},
                                logger, 'pretrain_state')
                return final_path

            loss = outputs.loss / pc.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % pc.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pc.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 100 == 0:
                avg = total_loss / (step + 1)
                elapsed = time.time() - logger.start
                steps_done = global_step - resume_step
                eta = elapsed / steps_done * (total_steps - global_step) if steps_done else 0
                logger.log(f'Epoch {epoch + 1} | Step {global_step}/{total_steps} | '
                           f'Loss: {avg:.4f} | ETA: {eta / 3600:.2f}h')

            if global_step % pc.save_every == 0:
                save_checkpoint(
                    model, tokenizer,
                    f'{config.paths.ckpt_dir}/pretrain_step{global_step}',
                    {'step': global_step, 'epoch': epoch,
                     'loss': total_loss / (step + 1)},
                    logger, 'pretrain_state')

        logger.log(f'Epoch {epoch + 1} done | '
                   f'Avg Loss: {total_loss / len(dataloader):.4f}')

    final_path = f'{config.paths.ckpt_dir}/pretrain_final'
    save_checkpoint(model, tokenizer, final_path,
                    {'step': total_steps, 'status': 'completed'},
                    logger, 'pretrain_state')
    logger.log(f'Pretrain complete! Saved: {final_path}')
    return final_path


def sft_loop(model: Any, tokenizer: Any, config: ExperimentConfig,
             logger: Logger,
             tokenized: Optional[Dataset] = None) -> str:
    """Supervised fine-tuning trên data NER.

    Như :func:`pretrain_loop`, thêm: save best model theo avg loss mỗi epoch,
    và khi gặp NaN loss thì giảm lr (``nan_lr_factor``) rồi chạy tiếp thay vì
    dừng hẳn (tối đa ``_MAX_NAN_RETRIES`` lần).

    Returns:
        Đường dẫn checkpoint ``sft_best``.
    """
    sc = config.sft
    if tokenized is None:
        tokenized = build_sft_dataset(tokenizer, config, logger)

    resume_state = logger.load_state('sft_state')
    resume_epoch = resume_state.get('epoch', 0) if resume_state else 0
    resume_step = resume_state.get('step', 0) if resume_state else 0
    best_loss = resume_state.get('best_loss', float('inf')) if resume_state else float('inf')
    logger.log(f'Resuming SFT from epoch {resume_epoch + 1}, step {resume_step}'
               if resume_step else 'Starting fresh SFT')

    # labels đã tính sẵn trong build_sft_dataset — không dùng
    # DataCollatorForLanguageModeling vì nó ghi đè 'labels' bằng input_ids
    dataloader = DataLoader(tokenized, batch_size=sc.batch_size,
                            shuffle=True, collate_fn=default_data_collator)
    total_steps = len(dataloader) * sc.epochs
    optimizer = optim.AdamW(model.parameters(), lr=sc.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * sc.warmup_ratio),
        num_training_steps=total_steps,
    )
    logger.log(f'SFT total steps: {total_steps:,} | Resume from step: {resume_step}')

    model.train()
    global_step = 0
    nan_count = 0
    best_path = f'{config.paths.ckpt_dir}/sft_best'

    for epoch in range(resume_epoch, sc.epochs):
        total_loss = 0.0
        n_loss_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            global_step += 1
            if global_step <= resume_step:
                continue

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)

            if torch.isnan(outputs.loss):
                nan_count += 1
                if nan_count > _MAX_NAN_RETRIES:
                    logger.log(f'NaN loss {nan_count} times — giving up at step {global_step}.')
                    save_checkpoint(model, tokenizer, best_path,
                                    {'epoch': epoch, 'step': global_step,
                                     'best_loss': best_loss, 'status': 'nan_stop'},
                                    logger, 'sft_state')
                    return best_path
                # bỏ batch lỗi, xả gradient bẩn, giảm lr rồi chạy tiếp
                optimizer.zero_grad()
                for group in optimizer.param_groups:
                    group['lr'] *= sc.nan_lr_factor
                logger.log(f'NaN loss at step {global_step}! '
                           f"Reduced lr to {optimizer.param_groups[0]['lr']:.2e} "
                           f'({nan_count}/{_MAX_NAN_RETRIES}), skipping batch.')
                continue

            loss = outputs.loss / sc.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()
            n_loss_steps += 1

            if (step + 1) % sc.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), sc.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 100 == 0:
                avg = total_loss / max(n_loss_steps, 1)
                elapsed = time.time() - logger.start
                eta = elapsed / (global_step - resume_step) * (total_steps - global_step)
                logger.log(f'Epoch {epoch + 1}/{sc.epochs} | '
                           f'Step {global_step}/{total_steps} | '
                           f'Loss: {avg:.4f} | ETA: {eta / 3600:.2f}h')

            if global_step % sc.save_every == 0:
                save_checkpoint(
                    model, tokenizer,
                    f'{config.paths.ckpt_dir}/sft_step{global_step}',
                    {'epoch': epoch, 'step': global_step,
                     'loss': total_loss / max(n_loss_steps, 1),
                     'best_loss': best_loss},
                    logger, 'sft_state')

        avg_epoch = total_loss / max(n_loss_steps, 1)
        logger.log(f'Epoch {epoch + 1} done | Avg Loss: {avg_epoch:.4f}')

        if avg_epoch < best_loss and n_loss_steps > 0:
            best_loss = avg_epoch
            save_checkpoint(model, tokenizer, best_path,
                            {'epoch': epoch, 'step': global_step,
                             'best_loss': best_loss},
                            logger, 'sft_state')
            logger.log(f'Best model saved: {best_path} (loss: {best_loss:.4f})')

    final_path = f'{config.paths.ckpt_dir}/sft_final'
    save_checkpoint(model, tokenizer, final_path,
                    {'step': total_steps, 'status': 'completed',
                     'best_loss': best_loss},
                    logger, 'sft_state')
    logger.log(f'SFT complete! Best loss: {best_loss:.4f}')
    return best_path
