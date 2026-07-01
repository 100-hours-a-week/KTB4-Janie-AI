from config import FOLDER_PATH
from data_loader import load_documents_from_folder
from chunker import clean_text, split_into_chunks
from embedder import model, embed_texts
from vectorstore import build_index
from retriever import retrieve
from generator import rag_answer

# 여러 파일 읽기
files = load_documents_from_folder(FOLDER_PATH)

print(f"로드된 파일 수: {len(files)}")
for i, doc in enumerate(files, 1):
    print(f"[{i}] {doc['metadata']['source']}")
    print(doc['text'][:120])
    print("-" * 40)

# 전처리
docs_cleaned = []
for doc in files:
    cleaned = clean_text(doc["text"])
    if len(cleaned) > 100:
        docs_cleaned.append({"text": cleaned})
print(f"전처리 후: {len(docs_cleaned)}개")

# 청킹
all_chunks = []
for doc_idx, doc in enumerate(docs_cleaned, start=1):
    chunks = split_into_chunks(doc["text"])
    for chunk_idx, chunk in enumerate(chunks, start=1):
        all_chunks.append({
            "title": f"doc_{doc_idx}_chunk_{chunk_idx}",
            "chunk": chunk,
        })
print(f"총 청크 수: {len(all_chunks)}")

# 임베딩 생성
texts = [c["chunk"] for c in all_chunks]
embeddings = embed_texts(texts)
print(f"임베딩 생성 완료: {len(embeddings)}개")

# FAISS 인덱싱
index = build_index(embeddings)
print(f"FAISS 인덱스 크기: {index.ntotal}")

# 검색 테스트
results = retrieve('머신러닝에 대해서 알려줘', model, index, all_chunks)
for r in results:
    print("내용:", r["chunk"][:150])
    print("---")

# RAG 답변 생성
answer = rag_answer(
    '머신러닝에 대해서 알려줘',
    lambda q: retrieve(q, model, index, all_chunks),
)
print(answer)
