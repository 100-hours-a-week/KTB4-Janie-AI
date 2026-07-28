import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import JSONResponse
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
    print('DEBUG history:', request.history)
    history = [h.dict() for h in request.history]
    result = music_search(request.question, history=history)
    return SearchResponse(**result)

app.mount('/static', StaticFiles(directory='frontend', html=True), name='static')

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={'source': 'error', 'answer': '죄송해요, 답변을 생성하던 중 문제가 발생했어요. 다시 질문해주시겠어요?'}
    )