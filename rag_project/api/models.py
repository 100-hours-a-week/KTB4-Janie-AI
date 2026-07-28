from pydantic import BaseModel
from typing import Literal

class HistoryItem(BaseModel):
    question: str
    answer: str

class SearchRequest(BaseModel):
    question: str
    history: list[HistoryItem] = []

class SearchResponse(BaseModel):
    source: Literal['spotify', 'youtube']
    answer: str

