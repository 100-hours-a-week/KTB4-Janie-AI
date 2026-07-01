"""
main.py
전체 RAG 파이프라인 오케스트레이션.
지금 단계: 문서 로딩 → 청킹 → 임베딩 → 인덱스 구축 → 검색 테스트
(생성 단계는 generator.py 완성 후 여기에 연결)
"""

from data_loader import load_documents
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import VectorStore
from retriever import Retriever
from config import RAW_DATA_PATH, FAISS_INDEX_PATH


def build_index():
    """원본 문서 → 청크 → 임베딩 → FAISS 인덱스 저장까지의 전체 인덱싱 과정."""
    print("1. 문서 로딩 중...")
    documents = load_documents(RAW_DATA_PATH)
    print(f"   {len(documents)}개 문서 로드 완료")

    print("2. 청킹 중...")
    chunks = chunk_documents(documents)
    print(f"   {len(chunks)}개 청크 생성 완료")

    print("3. 임베딩 생성 중...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    print("4. FAISS 인덱스 구축 중...")
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save()
    print(f"   인덱스 저장 완료: {FAISS_INDEX_PATH}")

    return store, embedder


def test_retrieval(store: VectorStore, embedder: Embedder, query: str):
    """저장된 인덱스로 검색 테스트."""
    retriever = Retriever(embedder, store)
    results = retriever.retrieve(query)

    print(f"\n쿼리: {query}")
    for i, r in enumerate(results, 1):
        print(f"[{i}] (score={r['score']:.4f}) {r.get('title', '')}")
        print(f"    {r['text'][:100]}...")


if __name__ == "__main__":
    store, embedder = build_index()
    test_retrieval(store, embedder, "테스트 쿼리를 입력하세요")
