from vectorstore import vector_store
from embedder import embeddings
from generator import llm, chain, format_docs, extract_search_params, generate_youtube_answer
from retriever import spotify_search_with_check, youtube_search, description_to_documents, youtube_rag_search


# 통합 검색
def music_search(user_question):
    params = extract_search_params(user_question, llm)
    results, need_fallback = spotify_search_with_check(user_question, params, vector_store)

    if not need_fallback:
        context = format_docs(results)
        answer = chain.invoke({"document": context, "question": user_question})
        return {'source': 'spotify', 'answer': answer}

    raw = youtube_search(query_suffix=params["search_style"], artist=params["artist"], song=params["song"])
    yt_docs = description_to_documents(raw)
    filtered_docs = youtube_rag_search(yt_docs, params['search_style'], embeddings)
    answer = generate_youtube_answer(user_question, filtered_docs)
    return {'source': 'youtube', 'answer': answer}


# ---------- 실행 (각 케이스 결과 1개씩만) ----------
if __name__ == "__main__":
    result = music_search('한로로 노래 추천해줘')
    print(f"[{result['source']}]\n{result['answer']}")
