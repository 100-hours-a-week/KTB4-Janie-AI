"""
generator.py
검색된 컨텍스트를 바탕으로 LLM(Gemini)이 답변을 생성하는 모듈.

⚠️ 지금 단계는 '검색(retrieval)까지'만 구현 대상이라 실제 API 호출은
    다음 단계에서 채울 스켈레톤만 잡아둠.
"""

from typing import List, Dict
from config import SYSTEM_PROMPT, GEMINI_MODEL_NAME


class Generator:
    def __init__(self, model_name: str = GEMINI_MODEL_NAME):
        self.model_name = model_name
        # TODO: 다음 단계에서 google-genai 클라이언트 초기화
        # import google.generativeai as genai
        # genai.configure(api_key=os.environ[GEMINI_API_KEY_ENV])
        # self.model = genai.GenerativeModel(model_name)

    def build_prompt(self, query: str, context: str) -> str:
        """검색된 컨텍스트 + 질문을 하나의 프롬프트로 조합."""
        return f"""{SYSTEM_PROMPT}

[참고 문서]
{context}

[질문]
{query}

[답변]"""

    def generate(self, query: str, context: str) -> str:
        """
        TODO: Gemini API 호출 구현 (다음 단계)
        prompt = self.build_prompt(query, context)
        response = self.model.generate_content(prompt)
        return response.text
        """
        raise NotImplementedError("생성 단계는 다음 스텝에서 구현 예정")
