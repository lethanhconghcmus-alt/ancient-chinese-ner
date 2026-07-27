"""Load/save model Qwen2.5-7B 4-bit + LoRA qua Unsloth.

Import ``unsloth`` được để bên trong function để module này vẫn import được
trên máy không có GPU (chạy test data/metric local).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from .config import ExperimentConfig
from .logger import Logger


def load_model_for_training(
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load model 4-bit + attach LoRA adapter để train.

    Args:
        config: experiment config (dùng section ``model``).
        checkpoint_path: nếu có thì load từ checkpoint đã save (resume /
            train tiếp từ pretrain_final); ``None`` thì load base model.

    Returns:
        ``(model, tokenizer)`` đã gắn LoRA, sẵn sàng train.
    """
    from unsloth import FastLanguageModel

    mc = config.model
    model_path = checkpoint_path or mc.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=mc.max_seq_len,
        load_in_4bit=mc.load_in_4bit,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=mc.lora_rank,
        lora_alpha=mc.lora_alpha,
        lora_dropout=mc.lora_dropout,
        target_modules=list(mc.target_modules),
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=mc.random_state,
    )
    return model, tokenizer


def load_model_for_inference(
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load model ở chế độ inference (generate nhanh, padding trái).

    Hai cách dùng:
      * ``checkpoint_path``: thư mục checkpoint đầy đủ (vd ``sft_best`` trên
        Drive) — load thẳng.
      * ``adapter_path``: load base model rồi gắn adapter LoRA rời (vd
        checkpoint tải từ Kaggle chỉ chứa adapter).
      * cả hai ``None``: base model thuần (zero-shot E1).

    Returns:
        ``(model, tokenizer)`` đã bật ``for_inference`` + ``padding_side='left'``.
    """
    from unsloth import FastLanguageModel

    mc = config.model
    if checkpoint_path:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=mc.max_seq_len,
            load_in_4bit=mc.load_in_4bit,
            dtype=None,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=mc.base_model,
            max_seq_length=mc.max_seq_len,
            load_in_4bit=mc.load_in_4bit,
            dtype=None,
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

    FastLanguageModel.for_inference(model)
    # decoder-only model: batched generation cần pad bên trái
    tokenizer.padding_side = 'left'
    return model, tokenizer


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    path: str,
    state: Optional[Dict[str, Any]] = None,
    logger: Optional[Logger] = None,
    state_name: str = 'train_state',
) -> None:
    """Save adapter + tokenizer, kèm training state JSON để resume.

    Args:
        model: model đang train (LoRA adapter sẽ được save).
        tokenizer: tokenizer đi kèm.
        path: thư mục checkpoint đích.
        state: dict trạng thái (step, epoch, loss, ...); ghi qua ``logger``
            nếu có, ngược lại ghi ``<path>/<state_name>.json``.
        logger: Logger của run (state đặt cạnh file log để loop resume tìm thấy).
        state_name: tên file state.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if state is not None:
        state = {**state, 'ckpt': path}
        if logger is not None:
            logger.save_state(state, state_name)
        else:
            with open(os.path.join(path, f'{state_name}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.log(f'Checkpoint saved: {path}')


def load_checkpoint_state(path: str) -> Optional[Dict[str, Any]]:
    """Đọc file state JSON (đường dẫn đầy đủ); trả ``None`` nếu không có."""
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None
