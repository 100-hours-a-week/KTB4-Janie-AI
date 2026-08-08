import os
import re
import uuid
import random
from googleapiclient.discovery import build
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from chunker import chunk_documents
from rapidfuzz import fuzz

load_dotenv()
youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

def find_artist_docs(vector_store, artist_variants, batch_size=5000, fuzzy_threshold=90):
    variants_lower = {v.lower() for v in artist_variants}
    total = vector_store._collection.count()
    matched_ids = []

    for offset in range(0, total, batch_size):
        batch = vector_store.get(limit=batch_size, offset=offset, include=['metadatas'])
        for doc_id, metadata in zip(batch['ids'], batch['metadatas']):
            individual_artists = {a.strip().lower() for a in metadata.get('artists', '').split(';')}
            if variants_lower & individual_artists:
                matched_ids.append(doc_id)
                continue

            if any(fuzz.ratio(v, a) >= fuzzy_threshold
                   for v in variants_lower for a in individual_artists):
                matched_ids.append(doc_id)

    return list(matched_ids)

def spotify_search_with_check(user_question, params, vector_store, top_k=5, score_threshold=0.9, taste_genres=None):
    artist_variants = params.get('artist_variants') or []

    if artist_variants:
        # 아티스트 검색 - 취향 반영 없음 (그대로)
        matched_ids = find_artist_docs(vector_store, artist_variants)
        if not matched_ids:
            return None, True

        selected_ids = random.sample(matched_ids, min(top_k, len(matched_ids)))
        result = vector_store.get(ids=selected_ids)
        docs = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(result['documents'], result['metadatas'])
        ]
        return docs, False

    # 아티스트 없음 - 무드 기반 검색
    # After
    if params.get('search_style'):
        search_query = f"{params['search_style']} 분위기의 음악"
    else:
        search_query = user_question
    results_with_scores = vector_store.similarity_search_with_score(search_query, k=top_k)
    if not results_with_scores:
        return None, True

    # 취향 반영: 이번 질문에 명시적 무드가 없고, 이전 취향 장르가 있으면 가산점
    if taste_genres and not params.get('search_style'):
        boosted = []
        for doc, score in results_with_scores:
            if doc.metadata.get('genre') in taste_genres:
                score = max(0, score - 0.05)
            boosted.append((doc, score))
        results_with_scores = boosted

    good_results = [doc for doc, score in results_with_scores if score < score_threshold]
    if not good_results:
        return None, True
    return good_results, False

# ---------- YouTube ----------
def youtube_search(query_suffix, artist_variants=None, song=None, max_results=30):
    artist_str = artist_variants[0] if artist_variants else ''
    base = f"{artist_str or ''} {song or ''}".strip()
    query = f"{base} {query_suffix or ''}".strip()
    request = youtube.search().list(q=query, part='snippet', type='video', maxResults=max_results)
    return request.execute()


def clean_description(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?\s', '', text)
    return text.strip()


def description_to_documents(youtube_response):
    docs = []
    for item in youtube_response['items']:
        video_id = item.get('id', {}).get('videoId')
        if not video_id:
            continue 

        title = item['snippet']['title']
        desc = clean_description(item['snippet']['description'])
        content = desc if desc else title
        docs.append(Document(
            page_content=content,
            metadata={'video_id': video_id, 'title': title, 'url': f"https://www.youtube.com/watch?v={video_id}"}
        ))
    return docs


def youtube_rag_search(docs, search_style, embeddings, chunk_size=400, chunk_overlap=80, top_k=3):
    if not docs:
        return []
    split_docs = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = [d for d in split_docs if d.page_content.strip()]
    if not split_docs:
        return []

    temp_collection_name = f"temp_{uuid.uuid4().hex[:8]}" 
    try:
        temp_vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=temp_collection_name)
        result = temp_vectorstore.similarity_search(search_style or '관련영상', k=top_k)
        return result
    except Exception as e:
        return []

from config import KNOWN_GENRES

def infer_taste_from_history(history: list) -> dict:
    """이전 답변 텍스트에서 실제 데이터셋 장르가 언급됐는지 확인"""
    if not history:
        return {}
    
    text = " ".join(h.get('answer', '') for h in history).lower()
    matched = [g for g in KNOWN_GENRES if g in text]
    return {'genres': matched}
