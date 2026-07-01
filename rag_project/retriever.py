"""
retriever.py
Embedder + VectorStore를 조합해서 '쿼리 → 관련 청크' 검색 인터페이스를 제공.
generator.py는 이 모듈의 retrieve()만 호출하면 됨 (내부 구현 몰라도 됨).
"""

from typing import List, Dict
from embedder import Embedder
from vectorstore import VectorStore
from config import TOP_K


class Retriever:
    def __init__(self, embedder: Embedder, vectorstore: VectorStore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        쿼리를 받아 관련 청크 리스트 반환.
        반환 형식: [{"text": str, "title": str, "score": float, ...}, ...]
        """
        query_embedding = self.embedder.encode_query(query)
        results = self.vectorstore.search(query_embedding, top_k=top_k)

        retrieved = []
        for metadata, score in results:
            retrieved.append({**metadata, "score": score})

        return retrieved

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """검색된 청크들을 LLM 프롬프트에 넣을 하나의 문자열로 합침."""
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get("title", "")
            text = chunk.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")

        return "\n\n".join(parts)
