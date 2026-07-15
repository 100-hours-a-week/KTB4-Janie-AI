import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_ollama import ChatOllama

from config import UPSTAGE_MODEL_NAME, OLLAMA_MODEL_NAME

load_dotenv()


# llm
def llm_fallback():
    try:
        model = ChatUpstage(
            api_key=os.getenv('UPSTAGE_API_KEY'),
            model=os.getenv("UPSTAGE_MODEL", UPSTAGE_MODEL_NAME),
            temperature=0)
        test = model.invoke('test')
        return model
    except Exception as e:
        print(f'gemini api key X -> ollama{e}')
        return ChatOllama(model=OLLAMA_MODEL_NAME)


llm = llm_fallback()
parser = StrOutputParser()

# ---------- Spotify RAG ----------
prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 음악 추천 챗봇이야. 아래 검색된 후보 곡 목록만 참고해서 사용자 질문에 맞게 추천 답변을 작성해. '
               '후보 목록에 없는 곡은 절대 지어내서 추천하지 마. '
               '각 곡을 추천할 때 곡명, 아티스트, 왜 추천하는지 간단히 설명해.\n\n'),
    ('human', '[후보 곡 목록]\n{document}\n\n[질문]\n{question}')
])

chain = prompt | llm | parser  # music_search에서 검증된 docs로 답변 생성할 때 씀


def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)


# ---------- 질문 분석 ----------
def extract_search_params(user_question, model):
    prompt_text = f"""당신은 음악 추천 시스템의 질의 분석기입니다.
사용자 질문에서 다음 세 가지 정보만 추출해 JSON으로 답하세요. 다른 키는 절대 추가하지 마세요.

- artist: 언급된 아티스트/가수 이름을 영문 표기로 변환해서 추출 (예: 아이유 → IU, 뉴진스 → NewJeans). 모르면 원문 그대로.
- song: 언급된 곡 제목 (없으면 null)
- search_style: 원하는 영상 스타일이나 분위기를 짧은 영어/한국어 키워드로 (예: "live", "piano cover", "신나는", 없으면 null)

반드시 이 세 개의 키만 포함한 JSON으로 답하세요: {{"artist": ..., "song": ..., "search_style": ...}}

예시:
질문: "아이유 좋은날 라이브로 듣고 싶어"
답: {{"artist": "아이유", "song": "좋은날", "search_style": "live"}}

질문: "신나는 곡 추천해줘"
답: {{"artist": null, "song": null, "search_style": "신나는"}}

질문: "{user_question}"
답:"""
    import json
    result = model.invoke(prompt_text).content
    cleaned = result.replace("```json", "").replace("```", "").strip()
    try:
        params = json.loads(cleaned)
        params = {k: params.get(k) for k in ["artist", "song", "search_style"]}
    except json.JSONDecodeError:
        params = {"artist": None, "song": None, "search_style": None}
    return params


youtube_prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 음악 추천 챗봇이야. 아래 영상 목록만 참고해서 사용자 질문에 답변해. '
               '목록에 없는 곡이나 영상은 절대 지어내서 언급하지 마. '
               '영상 목록에 있는 것만 근거로 답변하고, 목록에 충분한 정보가 없으면 있는 것만 소개해. '
               'URL이나 링크는 답변에 포함시키지 마. '
               '영상 제목은 원문 그대로 정확히 언급해줘 (줄이거나 바꾸지 마).'),
    ('human', '[영상 목록]\n{document}\n\n[질문]\n{question}')
])
youtube_chain = youtube_prompt | llm | parser


def format_docs_with_title(docs):
    return "\n\n".join(f"제목: {d.metadata['title']}\n내용: {d.page_content}" for d in docs)


def generate_youtube_answer(question, docs):
    context = format_docs_with_title(docs)
    answer = youtube_chain.invoke({"document": context, "question": question})
    seen_titles = set()

    for doc in docs:
        title = doc.metadata['title']
        if title in seen_titles:
            continue
        seen_titles.add(title)
        url = doc.metadata['url']
        if title in answer:
            answer = answer.replace(title, f"{title} ({url})")
    return answer
