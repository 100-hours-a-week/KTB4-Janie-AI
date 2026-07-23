from vectorstore import vector_store
from embedder import embeddings
from generator import llm, chain, format_docs, extract_search_params, generate_youtube_answer
from retriever import spotify_search_with_check, youtube_search, description_to_documents, youtube_rag_search
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

# 통합 검색
def music_search(user_question):
    params = extract_search_params(user_question, llm)
    results, need_fallback = spotify_search_with_check(user_question, params, vector_store)

    if not need_fallback:
        context = format_docs(results)
        answer = chain.invoke({"document": context, "question": user_question})
        return {'source': 'spotify', 'answer': answer}

    raw = youtube_search(query_suffix=params["search_style"], artist_variants=params["artist_variants"], song=params["song"])
    yt_docs = description_to_documents(raw)
    filtered_docs = youtube_rag_search(yt_docs, params['search_style'], embeddings)
    answer = generate_youtube_answer(user_question, filtered_docs)
    return {'source': 'youtube', 'answer': answer}

# LangGraph
class MusicState(TypedDict):
    question: str
    intent: str
    artist: str
    song: str
    search_style: str
    spotify_results: list
    need_fallback: bool
    answer: str
    source: str

# node
def detect_intent_node(state: MusicState) -> dict:
    params = extract_search_params(state['question', llm])
    return {
        'intent': params.get('intent', 'spotify_first'),
        'artist': params['artist'],
        'song': params['song'],
        'search_style': params['search_style']
    }

def spotify_search_node(state: MusicState) -> dict:
    params = {
        'artist': state['artist'],
        'song': state['song'],
        'search_style': state['search_style']
    }
    results, need_fallback = spotify_search_with_check(state['question'], params, vector_store)
    return {'spotify_results': results or [], 'need_fallback': need_fallback}

def generate_spotify_answer_node(state: MusicState) -> dict:
    context = format_docs(state['spotify_results'])
    answer = chain.invoke({'document': context, 'question': state['question']})
    return {'answer': answer, 'source': 'spotify'}

def youtube_search_node(state: MusicState) -> dict:
    raw = youtube_search(
        query_suffix=state['search_style'],
        artist = state['artist'],
        song = state['song'],
    )
    yt_docs = description_to_documents(raw)
    filtered_docs = youtube_rag_search(yt_docs, state['search_style'], embeddings)
    answer = generate_youtube_answer(state['question'], filtered_docs)
    return {'answer': answer, 'source': 'youtube'}

# router
def route_intent(state: MusicState) ->Literal['youtube', 'spotify']:
    if state['intent'] == 'youtube_direct':
        return 'youtube' 
    
def route_after_spotify(state: MusicState) -> Literal['youtube', 'generate_spotify']:
    if state['need_fallback']:
        return 'youtube'
    else:
        return 'generate_spotify'

# Graph 구성
builder = StateGraph(MusicState)

builder.add_node('detect_intent_node', detect_intent_node)
builder.add_node('spotify_search_node', spotify_search_node)
builder.add_node('youtube_search_node', youtube_search_node)
builder.add_node('generate_spotify_answer_node', generate_spotify_answer_node)

builder.add_edge(START, 'detect_intent_node')

builder.add_conditional_edges(
    'detect_intent_node',
    route_intent,
    {'youtube': 'youtube_search_node',
     'spotify': 'spotify_search_node'},
)

builder.add_conditional_edges(
    'spotify_search',
    route_after_spotify,
    {
        'youtube': 'youtube_search_node',
        'generate_spotify': 'generate_spotify_answer_node'
    }
)

builder.add_edge('youtube_search_node', END)
builder.add_edge('generate_spotify_answer_node', END)

graph = builder.compile()
# ---------- 실행 ----------
def music_search(user_question: str) -> dict:
    result = graph.invoke({"question": user_question})
    return {"source": result["source"], "answer": result["answer"]}

if __name__ == "__main__":
    result = music_search('비 오는 날 듣기 좋은 노래 추천해줘')
    print(f"[{result['source']}]\n{result['answer']}")
