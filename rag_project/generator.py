import os

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL_NAME

load_dotenv()  # .env 파일에서 환경 변수 로드

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def rag_answer(query, retrieve_fn):
    retrieved = retrieve_fn(query)
    context = "\n---\n".join([c["chunk"] for c in retrieved])
    prompt = f"""아래 문서들 중 질문과 관련된 것만 골라서 답해줘. 관련 없는 문서는 무시하고, 문서에 정보가 없으면 모른다고 해.

[문서]
{context}

[질문]
{query}
"""
    resp = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return resp.text
