"""
chunker.py
문서를 검색 단위인 '청크'로 분할.
overlap을 두는 이유: 청크 경계에서 문맥이 잘리는 걸 방지하기 위함.
"""

from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    단일 텍스트를 chunk_size 길이로 자르되 overlap만큼 겹치게 슬라이딩.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size는 overlap보다 커야 함")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    문서 리스트를 받아 청크 리스트로 변환.
    각 청크는 원본 문서의 메타데이터(title, source 등)를 유지.

    반환 형식:
    [{"chunk_id": str, "doc_id": str, "title": str, "text": str, "source": str}, ...]
    """
    chunked = []

    for doc in documents:
        text_chunks = split_text(doc["text"])
        for idx, chunk_text in enumerate(text_chunks):
            chunked.append({
                "chunk_id": f"{doc['id']}_{idx}",
                "doc_id": doc["id"],
                "title": doc.get("title", ""),
                "text": chunk_text,
                "source": doc.get("source", ""),
            })

    return chunked
