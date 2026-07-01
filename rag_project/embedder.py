"""
embedder.py
텍스트를 벡터로 변환하는 역할만 담당.
query와 chunk가 반드시 동일한 모델을 써야 cosine 비교가 의미 있음
→ 이 모듈 하나로 양쪽 다 통일해서 사용.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 행렬로 변환.
        normalize=True면 L2 정규화 → FAISS IndexFlatIP로 cosine similarity 구현 가능.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """단일 쿼리 임베딩 (검색 시 사용)."""
        return self.encode([query], normalize=normalize)[0]
