import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main import music_search
from api.models import SearchRequest, SearchResponse

app = FastAPI(title="Music Recommendation RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post('/search', response_model=SearchResponse)
def search(request: SearchRequest):
    result = music_search(request.question)
    return SearchResponse(**result)

app.mount('/static', StaticFiles(directory='frontend', html=True), name='static')