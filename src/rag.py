"""RAG few-shot retrieval cho NER: TF-IDF (E3) và BGE-M3 (E6).

Ý tưởng: với mỗi câu test, tìm k câu train giống nhất, chèn vào prompt làm
few-shot example (input + gold output) trước câu cần tag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import ExperimentConfig
from .data_utils import make_prompt_prefix
from .evaluate import evaluate_dataset
from .logger import Logger


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """Retriever TF-IDF trên char n-gram (Hán văn không có khoảng trắng)."""

    def __init__(self, train_data: Sequence[Dict[str, Any]],
                 ngram_range: tuple = (1, 2),
                 max_features: int = 50000) -> None:
        """Build index ngay khi khởi tạo.

        Args:
            train_data: list record train (dùng field ``input`` để index).
            ngram_range: khoảng char n-gram cho TfidfVectorizer.
            max_features: giới hạn vocabulary.
        """
        self.train_data = list(train_data)
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.vectorizer = self.build_index([r['input'] for r in self.train_data])

    def build_index(self, texts: Sequence[str]) -> Any:
        """Fit TfidfVectorizer và lưu ma trận vector của corpus train."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        self.matrix = vectorizer.fit_transform(texts)
        return vectorizer

    def retrieve(self, query: str, k: int = 1) -> List[Dict[str, Any]]:
        """Trả về k record train giống ``query`` nhất (cosine similarity)."""
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.train_data[i] for i in top_idx]


class BGERetriever:
    """Retriever dense embedding BGE-M3 (multilingual, hỗ trợ Hán văn)."""

    def __init__(self, model_name: str = 'BAAI/bge-m3') -> None:
        """Load model embedding; gọi :meth:`build_index` trước khi retrieve."""
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.train_data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts thành dense vectors (đã L2-normalize)."""
        output = self.model.encode(list(texts), batch_size=32,
                                   max_length=512)['dense_vecs']
        vecs = np.asarray(output, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def build_index(self, train_data: Sequence[Dict[str, Any]]) -> None:
        """Encode toàn bộ câu train làm index."""
        self.train_data = list(train_data)
        self.embeddings = self.encode([r['input'] for r in self.train_data])

    def retrieve(self, query: str, k: int = 1) -> List[Dict[str, Any]]:
        """Trả về k record train giống ``query`` nhất (dot product)."""
        if self.embeddings is None:
            raise RuntimeError('Call build_index(train_data) first.')
        q_vec = self.encode([query])[0]
        sims = self.embeddings @ q_vec
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.train_data[i] for i in top_idx]


def build_retriever(name: str, train_data: Sequence[Dict[str, Any]],
                    config: ExperimentConfig) -> Any:
    """Factory: tạo retriever theo tên ('tfidf' | 'bge') từ config."""
    rc = config.rag
    if name == 'tfidf':
        return TFIDFRetriever(train_data, ngram_range=rc.ngram_range,
                              max_features=rc.max_features)
    if name == 'bge':
        retriever = BGERetriever(rc.embedding_model)
        retriever.build_index(train_data)
        return retriever
    raise ValueError(f"Unknown retriever '{name}' (expected 'tfidf' or 'bge')")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_rag_prompt(record: Dict[str, Any],
                     shots: Sequence[Dict[str, Any]],
                     tokenizer: Any,
                     max_seq_len: int = 1024) -> str:
    """Build prompt few-shot, có kiểm soát token budget.

    Cấu trúc: Instruction + các cặp Example (Input/Output) + Input thật.
    Example nào làm prompt vượt ``max_seq_len`` token (chừa chỗ cho phần
    output sẽ generate) thì bị bỏ — thà ít shot còn hơn truncate mất câu
    cần tag.

    Args:
        record: record test cần tag.
        shots: các record train lấy từ retriever (đã sắp theo độ giống).
        tokenizer: để đếm token.
        max_seq_len: budget token cho toàn prompt.

    Returns:
        Prompt hoàn chỉnh kết thúc bằng ``### Output:\\n``.
    """
    base_prompt = make_prompt_prefix(record)
    # Phần bắt buộc (instruction + câu test) phải luôn nằm trong budget
    budget = max_seq_len - len(tokenizer(base_prompt)['input_ids'])

    example_blocks: List[str] = []
    for shot in shots:
        block = (
            f"### Example Input:\n{shot['input']}\n\n"
            f"### Example Output:\n{shot['output']}\n\n"
        )
        n_tokens = len(tokenizer(block)['input_ids'])
        if n_tokens > budget:
            break
        example_blocks.append(block)
        budget -= n_tokens

    return (
        f"### Instruction:\n{record['instruction']}\n\n"
        + ''.join(example_blocks)
        + f"### Input:\n{record['input']}\n\n### Output:\n"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_rag(model: Any, tokenizer: Any, retriever: Any,
                      test_data: Sequence[Dict[str, Any]],
                      config: ExperimentConfig, logger: Logger) -> Dict[str, Any]:
    """Full RAG evaluation: retrieve few-shot rồi chạy evaluate loop chung.

    Dùng lại :func:`src.evaluate.evaluate_dataset` với ``prompt_fn`` chèn
    few-shot — nên có đủ resume + checkpoint + error analysis như eval thường
    (state/checkpoint mang prefix ``rag`` để không đè kết quả eval thường).
    """
    num_shots = config.rag.num_shots
    max_seq_len = config.model.max_seq_len

    def rag_prompt(record: Dict[str, Any]) -> str:
        shots = retriever.retrieve(record['input'], k=num_shots)
        return build_rag_prompt(record, shots, tokenizer, max_seq_len)

    logger.log(f'RAG eval: retriever={type(retriever).__name__}, '
               f'shots={num_shots}, max_seq_len={max_seq_len}')
    return evaluate_dataset(model, tokenizer, test_data, config, logger,
                            prompt_fn=rag_prompt,
                            state_name='rag_eval_state', ckpt_prefix='rag')
