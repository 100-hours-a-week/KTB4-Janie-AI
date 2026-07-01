import numpy as np


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def search(query, model, embeddings, all_chunks, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = query_embedding[0]
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((i, score, all_chunks[i]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def retrieve(query, model, index, all_chunks, k=10):
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, idxs = index.search(np.array(q_emb, dtype=np.float32), k)
    return [all_chunks[i] for i in idxs[0]]
