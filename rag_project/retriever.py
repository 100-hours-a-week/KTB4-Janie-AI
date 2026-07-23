import os
import re
import uuid
from googleapiclient.discovery import build
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from embedder import embeddings
from chunker import chunk_documents


def spotify_search_with_check(user_question, params, vector_store, top_k=5, score_threshold=0.95):
    artist_variants = params.get('artist_variants') or ([params['artist']] if params.get('artist') else [])
    search_k = 1000 if artist_variants else top_k
    results_with_scores = vector_store.similarity_search_with_score(user_question, k=search_k)
    if not results_with_scores:
        return None, True

    if artist_variants:
        from rapidfuzz import fuzz
        matched = []
        for doc, score in results_with_scores:
            dataset_artist = doc.metadata.get("artists", "").lower()
            if any(fuzz.partial_ratio(v.lower(), dataset_artist) >= 85 for v in artist_variants):
                matched.append((doc, score))
        if not matched:
            return None, True
        return [doc for doc, score in matched[:top_k]], False
    # 아티스트 지정 안 한 경우(유사한 것 담는 list)
    good_results = [doc for doc, score in results_with_scores if score < score_threshold]
    if not good_results:
        return None, True
    return good_results, False


youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
load_dotenv()

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
            continue  # videoId 없는 항목(채널/재생목록 등)은 건너뜀

        title = item['snippet']['title']
        desc = clean_description(item['snippet']['description'])
        content = desc if desc else title
        docs.append(Document(
            page_content=content,
            metadata={'video_id': video_id, 'title': title, 'url': f"https://www.youtube.com/watch?v={video_id}"}
        ))
    return docs


def youtube_rag_search(docs, search_style, embeddings, chunk_size=400, chunk_overlap=80, top_k=3):
    split_docs = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    temp_collection_name = f"temp_{uuid.uuid4().hex[:8]}"  # collection_name 지정해 이전 정보 재사용 방지
    temp_vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=temp_collection_name)
    filtered = temp_vectorstore.similarity_search(search_style or '관련영상', k=top_k)
    return filtered
