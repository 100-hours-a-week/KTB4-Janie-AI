"""
vectorstore.py
FAISS 인덱스의 생성, 저장, 로드, 검색을 담당.
메타데이터(청크 원문, title 등)는 FAISS가 못 들고 있으므로 별도 리스트로 관리하고
인덱스 내 위치(row index)로 매핑.
"""

import pickle
import faiss
import numpy as np
from typing import List, Dict, Tuple
from config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIM


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 정규화된 벡터 + IP = cosine similarity
        self.metadata: List[Dict] = []  # index row와 1:1 매핑되는 청크 메타데이터

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """임베딩과 메타데이터를 인덱스에 추가."""
        assert len(embeddings) == len(metadata), "임베딩 개수와 메타데이터 개수가 일치해야 함"
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        쿼리 임베딩으로 top_k개 청크 검색.
        반환: [(메타데이터, 유사도 점수), ...] 유사도 내림차순
        """
        query_vec = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))

        return results

    def save(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """인덱스와 메타데이터를 디스크에 저장."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """디스크에서 인덱스와 메타데이터를 로드."""
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
