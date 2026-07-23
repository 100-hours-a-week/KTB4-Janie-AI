# 음악 추천 RAG 파이프라인 구조

## 1. 전체 흐름도

```
사용자 질문
    ↓
의도 분류 (LLM)
    ├─ "추천" → Spotify RAG 1차 시도
    ├─ "라이브" → YouTube 직행
    └─ "편곡/커버" → YouTube 직행
         ↓
    Spotify 결과 있음? 
         ├─ Yes → 답변 생성
         └─ No → YouTube 폴백
              ↓
        YouTube RAG 검색
              ↓
         답변 생성
```

---

## 2. Spotify RAG 파이프라인

### 2.1 데이터 준비 (한 번만 실행)

```
load_dataset("maharshipandya/spotify-tracks-dataset")
    ↓
row_to_text(row)  
  수치(popularity, danceability, energy, liveness, instrumentalness)
  → 자연어 문장으로 변환
    ↓
Document(page_content=..., metadata={genre, artists, instrumentalness, ...})
    ↓
Chroma 벡터스토어 (배치+딜레이로 114,000개 전체 임베딩)
  - 배치 크기: 50개
  - 배치 간 딜레이: 20초
  - 재실행 시 중단점부터 이어서 진행
```

### 2.2 검색 및 생성

```
사용자 질문 ("신나는 곡 추천해줘")
    ↓
spotify_retriever.invoke(질문)
  - 벡터 유사도 검색
  - 선택사항: genre 필터링 (메타데이터)
  - 상위 k=5개 반환
    ↓
LLM 체인 (fallback: Gemini → Upstage → Ollama)
  prompt | model | parser
    ↓
답변 생성
```

### 2.3 파일 구조

```
spotify_rag/
├── config.py              (설정값)
├── data_loader.py         (데이터 로딩 + row_to_text)
├── embedder.py            (Upstage 임베딩)
├── vectorstore.py         (배치 임베딩 + persist)
├── retriever.py           (검색 인터페이스)
├── generator.py           (LLM fallback + 추천 프롬프트)
└── main.py                (실행 진입점)
```

---

## 3. YouTube RAG 파이프라인

### 3.1 파라미터 추출

```
사용자 질문 ("아이우 노래 피아노 버전으로 추천해줘")
    ↓
extract_search_params(질문, LLM)
  LLM이 다음을 추출:
  - artist: "아이유"
  - song: None 또는 "좋은날"
  - search_style: "piano cover"
    ↓
{"artist": "...", "song": "...", "search_style": "..."}
```

### 3.2 검색

```
youtube_search(query_suffix, artist, song)
  검색어 = "{artist} {song} {query_suffix}"
  예: "아이유 좋은날 piano cover"
    ↓
YouTube API 호출 (최대 10개 결과)
    ↓
raw_results (title, description, videoId 등)
```

### 3.3 Document 변환 + 청킹

```
raw_results
    ↓
description_to_documents(raw_results)
  - description → page_content
  - description 비어있으면 title로 폴백
  - metadata: {video_id, title}
    ↓
clean_description(text)  (URL, 해시태그, 타임스탬프 제거)
    ↓
RecursiveCharacterTextSplitter
  - chunk_size: 400
  - chunk_overlap: 80
    ↓
split_docs
```

### 3.4 임시 벡터스토어 + 재검색

```
Chroma.from_documents(split_docs, embeddings)
  ⚠️  persist_directory 없음 (휘발성, 검색마다 새로 생성)
    ↓
vectorstore.similarity_search(search_style, k=3)
  예: "piano cover" 문자열로 유사도 재검색
    ↓
filtered (최종 3개 결과)
```

### 3.5 파일 구조

```
youtube_search.py
├── extract_search_params(질문, model)
├── youtube_search(query_suffix, artist, song)
├── description_to_documents(raw_results)
├── clean_description(text)
├── youtube_rag_search(docs, search_style, embeddings)
└── youtube_pipeline(질문, model, embeddings)  ← 진입점
```

---

## 4. 통합 흐름 (LangGraph 이전)

```python
# 1. Spotify RAG 단독 테스트
result = spotify_retriever.invoke("신나는 곡")
answer = chain.invoke({"document": docs, "question": "..."})

# 2. YouTube RAG 단독 테스트
filtered = youtube_pipeline("아이유 피아노 버전", model, embeddings)
```

---

## 5. LangGraph 구조 (다음 단계)

```
State: {question, intent, artist, song, search_style, 
        spotify_results, youtube_results, answer}

Nodes:
  1. detect_intent_node
     → intent, artist, song, search_style 추출
  
  2. route_by_intent (conditional)
     intent="direct_youtube" → youtube 노드로
     나머지 → spotify 노드로
  
  3. spotify_search_node
     → spotify_results, need_fallback 설정
  
  4. route_after_spotify (conditional)
     need_fallback=True → youtube 노드로
     False → generate 노드로
  
  5. youtube_search_node
     → youtube_results 설정
  
  6. generate_answer_node
     → spotify_results + youtube_results 합쳐서 답변

Edges:
  START → detect_intent → route_by_intent
         ├→ spotify_search → route_after_spotify
         │                  ├→ generate_answer → END
         │                  └→ youtube_search → generate_answer → END
         └→ youtube_search → generate_answer → END
```

---

## 6. 현재 진행 상황

### ✅ 완료
- Spotify RAG 파이프라인 뼈대 (config, data_loader, embedder, vectorstore, retriever, generator)
- YouTube 파라미터 추출 함수 뼈대
- 각 단계별 로직 설계

### 🔄 진행 중
- YouTube RAG 함수 구현 (extract_search_params, youtube_search, youtube_rag_search)

### ⏳ 예정
- LangGraph 노드 연결
- 전체 통합 테스트

---

## 7. 주요 설계 원칙

| 원칙 | 적용 |
|---|---|
| 한 번만 임베딩 | Spotify persist 패턴 (재실행 시 재임베딩 X) |
| 배치+딜레이 | rate limit 방지, 중단 후 재개 가능 |
| 함수 모듈화 | 각 함수는 하나의 책임만 (검색, 변환, 생성 분리) |
| LLM fallback | Gemini → Upstage → Ollama (3단계) |
| 범용 YouTube 검색 | query_suffix 파라미터로 무한 확장성 |
| 폴백 라우팅 | "결과 있냐 없냐"가 유일한 판단 기준 |
| 에러 처리 | JSON 파싱 실패 시 기본값, API 재시도 |

---

## 8. 데이터 흐름 예시

### 시나리오: "아이유 곡 라이브로 들어서 피아노 버전 추천해줘"

```
1. LLM (의도 분류)
   입력: "아이유 곡 라이브로 들어서 피아노 버전 추천해줘"
   출력: {intent: "direct_youtube", artist: "아이유", 
           song: null, search_style: "piano cover live"}

2. YouTube (직행)
   입력: youtube_pipeline("아이우 곡...", model, embeddings)
   
3. 파라미터 추출
   artist="아이유", song=None, search_style="piano cover live"
   
4. YouTube 검색
   query="아이유 piano cover live"
   
5. 결과 (예: 10개)
   - [0] "아이유 - 좋은날 (Piano Cover Live)"
   - [1] "아이유 Spring Day Piano Version..."
   - ...
   
6. Document 변환 + 청킹
   각 영상의 description → 여러 청크로 분할
   
7. 재검색
   "piano cover live" 유사도로 상위 3개
   
8. 답변 생성
   LLM이 3개 영상을 기반으로 설명

답변: "아이유의 라이브 피아노 버전은..."
```

---

## 9. 다음 단계 체크리스트

- [ ] `extract_search_params()` 테스트
- [ ] `youtube_search()` 테스트 (손으로 파라미터 입력)
- [ ] `description_to_documents()` 테스트
- [ ] `youtube_rag_search()` 전체 흐름 테스트
- [ ] `youtube_pipeline()` end-to-end 테스트
- [ ] LangGraph 노드 구현
- [ ] 전체 통합 테스트