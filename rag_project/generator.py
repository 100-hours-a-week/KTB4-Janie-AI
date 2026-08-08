import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from config import CLAUDE_MODEL_NAME, OLLAMA_MODEL_NAME

load_dotenv()

def llm_fallback():
    try:
        model = ChatAnthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model=CLAUDE_MODEL_NAME
        )
        model.invoke('test')
        return model
    except Exception as e:
        print(f'gemini api key X -> ollama{e}')
        return ChatOllama(model=OLLAMA_MODEL_NAME)


llm = llm_fallback()
parser = StrOutputParser()

# ---------- Spotify RAG ----------
prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 음악 추천 챗봇이야. 아래 검색된 후보 곡 목록만 참고해서 사용자 질문에 맞게 최대 5곡까지 추천 답변을 작성해. '
               '후보 목록에 없는 곡은 절대 지어내서 추천하지 마. '
               '각 곡을 추천할 때 곡명, 아티스트, 왜 추천하는지 간단히 설명해.\n\n'),
    ('human', '[후보 곡 목록]\n{document}\n\n[질문]\n{question}')
])

chain = prompt | llm | parser  

def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)

def _extract_text_content(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

# ---------- 질문 분석 ----------
def extract_search_params(user_question, model, history=None):
    print('DEBUG generator.py received history:', history)
    history_text = ""
    if history:
        recent = history[-1:]
        history_text="\n\n[이전 대화 참고]\n" + "\n".join(
            f"- 질문: {h['question']}\n 답변 요약: {h['answer'][:150]}"
            for h in recent
        )

    prompt_text = f"""당신은 음악 추천 시스템의 질의 분석기입니다.
사용자 질문에서 다음 정보를 추출해 반드시 JSON으로만 답하세요. 다른 키는 절대 추가하지 마세요.

- artist_variants: 언급된 아티스트의 표기 변형들을 배열로 추출 (예: "5SOS" → ["5SOS", "5 Seconds of Summer"])
  없으면 빈 배열 []
- song: **지금 질문(가장 최근 한 문장)에서 구체적인 곡명이 언급된 경우에만** 채우세요.
  이전 대화에서 곡명이 나왔더라도, 지금 질문이 그 곡 자체가 아니라 아티스트 전반에 대한
  것(예: "다른 곡도 있어?", "플레이리스트도 있어?", "이 가수 다른 노래는?")이면 song은
  반드시 null로 두세요. artist_variants는 이전 대화의 아티스트를 계속 유지해도 됩니다.
- search_style: 원하는 영상 스타일이나 분위기를 짧은 영어/한국어 키워드로 (예: "live", "piano cover", "신나는", 없으면 null)
- intent: 다음 중 하나
  - "youtube_direct": 요청한 콘텐츠가 "Spotify 같은 음원 스트리밍 카탈로그에
    존재할 수 없는 종류"인 경우.
    Spotify 카탈로그는 공식 발매된 개별 음원만 담고 있으며, 영상 콘텐츠, 팬이 만든 2차 창작물,
    여러 곡을 묶은 편집물, 그리고 데이터 수집 시점 이후에 나온 곡은 포함하지 않습니다.
    (예: 라이브 영상, 커버/편곡, 플레이리스트, 최신 발매곡 등)
    또는 사용자가 유튜브를 검색 플랫폼으로 명시한 경우.
  - "spotify_first": 그 외 일반적인 곡/분위기/아티스트 추천 (기본값, 애매하면 이걸로)
  - "out_of_scope": 음악/노래/아티스트와 명백히 무관한 질문
    (예: 날씨, 코딩, 일반 상식, 잡담)
    조금이라도 음악 추천으로 해석 가능하면 이 값을 쓰지 말 것

만약 지금 질문이 "방금 말한 가수", "이 노래", "그 영상" 처럼 이전 대화를 참조하고 있다면,
아래 [이전 대화 참고]에서 언급된 아티스트/곡을 찾아 artist_variants와 song에 채우세요.
{history_text}

반드시 아래 네 개의 키만 포함한 JSON으로만 답하세요. 다른 텍스트는 포함하지 마세요:
{{"artist_variants": [...], "song": ..., "search_style": ..., "intent": ...}}

예시:
질문: "5 Seconds of Summer의 youngblood 라이브로 듣고 싶어"
답: {{"artist_variants": ["5 Seconds of Summer", "5SOS"], "song": "youngblood", "search_style": "live", "intent": "youtube_direct"}}

질문: "신나는 곡 추천해줘"
답: {{"artist_variants": [], "song": null, "search_style": "신나는", "intent": "spotify_first"}}

질문: "{user_question}"
답:"""
    response = model.invoke(prompt_text)
    result = _extract_text_content(response)
    cleaned = result.replace("```json", "").replace("```", "").strip()
    try:
       params = json.loads(cleaned)
       params = {k: params.get(k) for k in ["artist_variants", "song", "search_style", "intent"]}
    except json.JSONDecodeError:
       params = {'artist_variants': [], "song":None, "search_style": None, "intent":"spotify_first"}
    return params


youtube_prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 음악 추천 챗봇이야. 아래 영상 목록만 참고해서 최대 5곡까지 사용자 질문에 답변해. '
               '목록에 없는 곡이나 영상은 절대 지어내서 언급하지마. '
               '영상 목록에 있는 것만 근거로 답변하고, 목록에 충분한 정보가 없으면 있는 것만 소개해. '
               'URL이나 링크는 답변에 포함시키지마. '
               '영상 제목은 원문 그대로 정확히 언급해줘 (줄이거나 바꾸지마).'),
    ('human', '[영상 목록]\n{document}\n\n[질문]\n{question}')
])
youtube_chain = youtube_prompt | llm | parser

NEWNESS_KEYWORDS = ['신곡', '최신곡', '요즘 나온', '새 앨범', '따끈']

def force_youtube_if_newness(question: str, intent: str) -> str:
    if intent == 'spotify_first' and any(k in question for k in NEWNESS_KEYWORDS):
        return 'youtube_direct'
    return intent
def format_docs_with_title(docs):
    return "\n\n".join(f"제목: {d.metadata['title']}\n내용: {d.page_content}" for d in docs)


def generate_youtube_answer(question, docs, resolved_artist=None):
    context = format_docs_with_title(docs)
    effective_question = question
    if resolved_artist:
        effective_question = f"{resolved_artist}에 대한 질문: {question}"
    answer = youtube_chain.invoke({"document": context, "question": effective_question})
    return insert_links(answer, docs)

def insert_links(answer: str, docs) -> str:
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


async def stream_spotify_answer(context: str, question: str):
    async for chunk in chain.astream({'document': context, 'question': question}):
        yield chunk
async def stream_youtube_answer(context: str, question: str):
    async for chunk in youtube_chain.astream({'document': context, 'question': question}):
        yield chunk
