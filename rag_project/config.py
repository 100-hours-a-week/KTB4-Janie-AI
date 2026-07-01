"""
config.py
전역 설정값 관리. 경로, 모델명, 청킹/검색 파라미터를 한 곳에서 관리.
다른 모듈은 여기서 값을 import해서 사용.
"""

from pathlib import Path

# ── 경로 설정 ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# ── 임베딩 모델 ─────────────────────────────
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIM = 768  # ko-sroberta-multitask 출력 차원

# ── 청킹 파라미터 ───────────────────────────
CHUNK_SIZE = 500          # 문자 기준
CHUNK_OVERLAP = 50        # 겹치는 구간

# ── 검색 파라미터 ───────────────────────────
TOP_K = 5                 # 검색 시 반환할 청크 개수
SIMILARITY_METRIC = "cosine"  # IndexFlatIP + 정규화로 cosine 구현

# ── LLM (Gemini) 설정 ───────────────────────
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # 환경변수명

# ── 프롬프트 템플릿 ─────────────────────────
SYSTEM_PROMPT = """당신은 주어진 문서를 기반으로 정확하게 답변하는 어시스턴트입니다.
문서에 없는 내용은 추측하지 말고 모른다고 답하세요."""
