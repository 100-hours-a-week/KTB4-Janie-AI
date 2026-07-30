import sys
import os
import logging
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import JSONResponse
from main import music_search
from api.models import SearchRequest, SearchResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Music Recommendation RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        history = [h.model_dump() for h in request.history]
        return SearchResponse(**music_search(request.question, history=history))
    except Exception:
        logger.exception("music_search failed | question=%r", request.question)
        raise  

app.mount('/', StaticFiles(directory='frontend', html=True), name='static')

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        'Unhandled error on %s %s', request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={'source': 'none', 'answer': '죄송해요, 답변을 생성하던 중 문제가 발생했어요. 다시 질문해주시겠어요?'},
    )