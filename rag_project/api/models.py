from pydantic import BaseModel
from typing import Literal

class SearchRequest(BaseModel):
    question: str

class SearchResponse(BaseModel):
    source: Literal['spotify', 'youtube']
    answer: str