"""Cấu hình tập trung cho toàn bộ pipeline DVSKTT NER.

Mọi hyperparameter và đường dẫn đều nằm ở đây (hoặc trong file YAML ở
``configs/``) — notebook và script chỉ load config, không hardcode.

Cách dùng::

    from src.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml('configs/e4_sft_1024.yaml')
    print(cfg.model.max_seq_len)   # 1024

File YAML chỉ cần ghi các giá trị muốn override — phần còn lại lấy từ
``configs/base.yaml`` (nếu có) rồi đến default trong các dataclass dưới đây.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Từng nhóm config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Cấu hình base model + LoRA adapter."""

    base_model: str = 'unsloth/qwen2.5-7b-unsloth-bnb-4bit'
    max_seq_len: int = 512
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    target_modules: tuple = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    random_state: int = 42


@dataclass
class PretrainConfig:
    """Cấu hình continued pretraining trên corpus Hán văn."""

    lr: float = 5e-5
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 4
    sample_size: int = 5000       # tổng số dòng corpus (DVSKTT + Hán sample)
    dvsktt_size: int = 1311       # số dòng đầu file merged là DVSKTT — giữ 100%
    save_every: int = 500         # auto-save checkpoint mỗi N steps
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class SFTConfig:
    """Cấu hình supervised fine-tuning trên data NER."""

    lr: float = 5e-5
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 4
    save_every: int = 200
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    # Khi gặp NaN loss: giảm lr còn lr * nan_lr_factor rồi thử lại epoch sau
    nan_lr_factor: float = 0.5


@dataclass
class EvalConfig:
    """Cấu hình evaluation trên test set."""

    batch_size: int = 4
    max_new_tokens: int = 900
    save_every: int = 50          # auto-save kết quả mỗi N records
    entity_types: tuple = ('PER', 'LOC', 'ORG', 'DTM', 'TITLE')


@dataclass
class RAGConfig:
    """Cấu hình RAG few-shot retrieval."""

    num_shots: int = 1
    retriever: str = 'tfidf'                  # 'tfidf' | 'bge'
    ngram_range: tuple = (1, 2)               # cho TF-IDF (char n-gram)
    max_features: int = 50000                 # cho TF-IDF
    embedding_model: str = 'BAAI/bge-m3'      # cho BGE retriever


@dataclass
class PathConfig:
    """Đường dẫn data / checkpoint / kết quả.

    Default trỏ theo môi trường Colab (``/content/...``); khi chạy Kaggle
    hoặc local chỉ cần override trong YAML hoặc gọi :meth:`resolve_env`.
    """

    data_dir: str = '/content/repo/data/raw'
    ckpt_dir: str = '/content/drive/MyDrive/dvsktt_ner/checkpoints'
    result_dir: str = '/content/drive/MyDrive/dvsktt_ner/results'
    log_dir: str = '/content/drive/MyDrive/dvsktt_ner/logs'

    # Kaggle datasets (checkpoint quá lớn cho git nên tải qua Kaggle API)
    kaggle_sft_ckpt: str = 'thnhcngl/dvsktt-sft-best-checkpoint'
    kaggle_download_dir: str = '/content/data'

    @property
    def sft_data_dir(self) -> str:
        """Thư mục chứa train/dev/test.jsonl."""
        return os.path.join(self.data_dir, 'ner_sft')

    @property
    def han_corpus_path(self) -> str:
        """File corpus Hán văn cho pretraining."""
        return os.path.join(self.data_dir, 'han_pretrain', 'dvsktt_han_merged.txt')

    def resolve_env(self) -> 'PathConfig':
        """Tự nhận diện môi trường (Kaggle / Colab / local) và sửa đường dẫn.

        Chỉ đổi các đường dẫn còn đang mang giá trị default Colab; giá trị
        đã được override trong YAML thì giữ nguyên.
        """
        defaults = PathConfig()
        if os.path.exists('/kaggle'):
            base = '/kaggle/working/dvsktt_ner'
            repo_data = '/kaggle/working/repo/data/raw'
        elif os.path.exists('/content'):
            return self  # đúng môi trường của default rồi
        else:
            base = str(Path.cwd() / 'runs')
            repo_data = str(Path(__file__).resolve().parents[1] / 'data' / 'raw')

        if self.data_dir == defaults.data_dir:
            self.data_dir = repo_data
        for name, sub in [('ckpt_dir', 'checkpoints'),
                          ('result_dir', 'results'),
                          ('log_dir', 'logs')]:
            if getattr(self, name) == getattr(defaults, name):
                setattr(self, name, os.path.join(base, sub))
        return self

    def makedirs(self) -> None:
        """Tạo các thư mục output nếu chưa có."""
        for d in [self.ckpt_dir, self.result_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Config tổng
# ---------------------------------------------------------------------------

# Map key YAML -> dataclass tương ứng
_SECTIONS = {
    'model': ModelConfig,
    'pretrain': PretrainConfig,
    'sft': SFTConfig,
    'eval': EvalConfig,
    'rag': RAGConfig,
    'paths': PathConfig,
}


@dataclass
class ExperimentConfig:
    """Config đầy đủ cho một experiment (E1..E5)."""

    name: str = 'default'
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> 'ExperimentConfig':
        """Build config từ dict (đã merge); key lạ sẽ báo lỗi ngay."""
        cfg = cls(name=raw.get('name', 'default'))
        for section, klass in _SECTIONS.items():
            values = raw.get(section) or {}
            valid = {f.name for f in fields(klass)}
            unknown = set(values) - valid
            if unknown:
                raise KeyError(
                    f"Unknown key(s) {sorted(unknown)} in section '{section}' "
                    f"(valid: {sorted(valid)})"
                )
            # tuple fields (ngram_range, entity_types, ...) đến từ YAML là list
            coerced = {}
            for f in fields(klass):
                if f.name in values:
                    v = values[f.name]
                    coerced[f.name] = tuple(v) if isinstance(v, list) else v
            setattr(cfg, section, klass(**coerced))
        return cfg

    @classmethod
    def from_yaml(cls, path: str,
                  base_path: Optional[str] = None) -> 'ExperimentConfig':
        """Load config từ YAML, merge lên ``configs/base.yaml`` nếu tồn tại.

        Args:
            path: file YAML của experiment (vd ``configs/e4_sft_1024.yaml``).
            base_path: file base YAML; mặc định tìm ``base.yaml`` cùng thư mục.
        """
        import yaml

        path = os.path.abspath(path)
        with open(path, 'r', encoding='utf-8') as f:
            overlay = yaml.safe_load(f) or {}

        if base_path is None:
            candidate = os.path.join(os.path.dirname(path), 'base.yaml')
            base_path = candidate if os.path.exists(candidate) else None

        merged: Dict[str, Any] = {}
        if base_path and os.path.abspath(base_path) != path:
            with open(base_path, 'r', encoding='utf-8') as f:
                merged = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, overlay)

        cfg = cls.from_dict(merged)
        if 'name' not in merged:
            cfg.name = Path(path).stem
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize về dict (dùng để log lại config của mỗi run)."""
        out: Dict[str, Any] = {'name': self.name}
        for section in _SECTIONS:
            group = getattr(self, section)
            section_dict = {}
            for f in fields(group):
                v = getattr(group, f.name)
                section_dict[f.name] = list(v) if isinstance(v, tuple) else v
            out[section] = section_dict
        return out


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge đệ quy: giá trị trong ``overlay`` thắng ``base``."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
