from vectorstore import vector_store
from embedder import embeddings
from generator import llm, chain, format_docs, extract_search_params, generate_youtube_answer, format_docs, format_docs_with_title, stream_spotify_answer, stream_youtube_answer, insert_links
from retriever import spotify_search_with_check, youtube_search, description_to_documents, youtube_rag_search
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class MusicState(TypedDict):
    question: str
    history: list
    intent: str
    artist_variants: list
    song: str
    search_style: str
    spotify_results: list
    need_fallback: bool
    answer: str
    source: str
    

# node
def detect_intent_node(state: MusicState) -> dict:
    params = extract_search_params(state['question'], llm, state.get('history'))
    return {
        'intent': params.get('intent', 'spotify_first'),
        'artist_variants': params.get('artist_variants') or [],
        'song': params.get('song'),
        'search_style': params.get('search_style')
    }

def spotify_search_node(state: MusicState) -> dict:
    params = {
        'artist_variants': state['artist_variants'],
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
        artist_variants = state['artist_variants'],
        song = state['song'],
    )
    yt_docs = description_to_documents(raw)
    resolved_artist = state['artist_variants'][0] if state['artist_variants'] else None
    filtered_docs = youtube_rag_search(yt_docs, state['search_style'], embeddings)
    answer = generate_youtube_answer(state['question'], filtered_docs, resolved_artist=resolved_artist)
    return {'answer': answer, 'source': 'youtube'}

def out_of_scope_node(state: MusicState) -> dict:
    return {
        'answer': '저는 음악 추천을 도와드리는 챗봇이에요. 듣고 싶은 분위기나 아티스트를 말씀해주시면 곡을 찾아드릴게요!',
        'source': 'none'
    }

# router
def route_after_spotify(state: MusicState) -> Literal['youtube', 'generate_spotify']:
    if state['need_fallback']:
        return 'youtube'
    else:
        return 'generate_spotify'

def route_intent(state) -> Literal['youtube', 'spotify', 'out_of_scope']:
    if state['intent'] == 'out_of_scope':
        return 'out_of_scope'
    elif state['intent'] == 'youtube_direct':
        return 'youtube'
    else:
        return 'spotify'
    
# Graph 구성
builder = StateGraph(MusicState)

builder.add_node('detect_intent_node', detect_intent_node)
builder.add_node('spotify_search_node', spotify_search_node)
builder.add_node('youtube_search_node', youtube_search_node)
builder.add_node('generate_spotify_answer_node', generate_spotify_answer_node)
builder.add_node('out_of_scope_node', out_of_scope_node)

builder.add_edge(START, 'detect_intent_node')

builder.add_conditional_edges(
    'detect_intent_node',
    route_intent,
    {'youtube': 'youtube_search_node',
     'spotify': 'spotify_search_node',
     'out_of_scope': 'out_of_scope_node'},
)

builder.add_conditional_edges(
    'spotify_search_node',
    route_after_spotify,
    {
        'youtube': 'youtube_search_node',
        'generate_spotify': 'generate_spotify_answer_node'
    }
)

builder.add_edge('youtube_search_node', END)
builder.add_edge('generate_spotify_answer_node', END)
builder.add_edge('out_of_scope_node', END)

graph = builder.compile()
# ---------- 실행 ----------
def music_search(user_question: str, history: list = None) -> dict:
    #print("DEBUG main.py received history:", history)
    result = graph.invoke({"question": user_question, 'history': history or []})
    return {"source": result["source"], "answer": result["answer"]}

async def music_search_stream(user_question: str, history: list=None):
    params = extract_search_params(user_question, llm, history)
    intent = params.get('intent', 'spotify_first')

    if intent == 'out_of_scope':
        yield {'type': 'meta', 'source': 'none'}
        yield {'type': 'token', 'text': '저는 음악 추천을 도와드리는 챗봇이에요.'}
        return
    if intent == 'youtube_direct':
        raw = youtube_search(query_suffix=params['search_style'],
                              artist_variants=params['artist_variants'], song=params['song'])
        yt_docs = description_to_documents(raw)
        filtered_docs = youtube_rag_search(yt_docs, params['search_style'], embeddings)
        context = format_docs_with_title(filtered_docs)
        yield {"type": "meta", "source": "youtube"}
        full_text = ''
        async for chunk in stream_youtube_answer(context, user_question):
            full_text += chunk
            yield {"type": "token", "text": chunk}

        linked_answer = insert_links(full_text, filtered_docs)
        yield {'type': 'final', 'text': linked_answer}
        return

        

    # spotify_first
    results, need_fallback = spotify_search_with_check(user_question, params, vector_store)
    if not need_fallback:
        context = format_docs(results)
        yield {"type": "meta", "source": "spotify"}
        async for chunk in stream_spotify_answer(context, user_question):
            yield {"type": "token", "text": chunk}
        return
    
    raw = youtube_search(query_suffix=params['search_style'],
                          artist_variants=params['artist_variants'], song=params['song'])
    yt_docs = description_to_documents(raw)
    filtered_docs = youtube_rag_search(yt_docs, params['search_style'], embeddings)
    context = format_docs_with_title(filtered_docs)
    yield {"type": "meta", "source": "youtube"}
    full_text = ''
    async for chunk in stream_youtube_answer(context, user_question):
        full_text += chunk
        yield {"type": "token", "text": chunk}
    linked_answer = insert_links(full_text, filtered_docs)
    yield {'type': 'final', 'text': linked_answer}
