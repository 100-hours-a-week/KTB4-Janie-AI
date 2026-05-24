### WEEK02 - 과제2. FastAPI 커뮤니티 서비스 백엔드 구현
---
#### 프로젝트 구조

```
week02-2/
├── main.py              # 진입점 — 라우터 등록 및 서버 시작
├── database.py          # DB 연결 설정
├── models/
│   ├── __init__.py
│   ├── post.py          # 게시글 DB 테이블 모델
│   └── comment.py       # 댓글 DB 테이블 모델
├── schemas/
│   ├── __init__.py
│   ├── post.py          # 게시글 요청/응답 Pydantic 모델
│   └── comment.py       # 댓글 요청/응답 Pydantic 모델
├── controllers/
│   ├── __init__.py
│   ├── posts.py         # 게시글 비즈니스 로직
│   └── comments.py      # 댓글 비즈니스 로직
└── routers/
    ├── __init__.py
    ├── posts.py         # 게시글 엔드포인트
    └── comments.py      # 댓글 엔드포인트
```
---

#### 실행 방법
1. 레포지토리 클론
```
git clone https://github.com/100-hours-a-week/KTB4-Janie-AI
cd KTB4-Janie-AI/02/week02-2
```
2. 패키지 설치
```
uv pip install fastapi uvicorn sqlalchemy ollama
```
3. 서버 실행
```
uvicorn main:app --reload
```
Swagger 문서: http://127.0.0.1:8000/docs

---
#### 회고

이번 과제에서는 특히 ollama 연동이나 DB 연결이 어려웠던 것 같다. Ollama 연동 시 ollama.chat은 동기 함수라 async/await를 쓰려면 AsyncClient를 따로 사용해야 한다는 것을 몰랐다. 
처음에는 일반 ollama.chat에 await를 붙여서 에러가 났고 AsyncClient().chat으로 바꾸고 나서야 정상 동작했다.
또한 DB에 대해 배웠던 적은 있지만 직접 DB를 연결하려니 엄두가 안 났던 것 같고 특히 기존에 인메모리 리스트로 관리하던 방식에서 SQLAlchemy로 바꾸면서 객체 접근 방식이 달라져 헷갈렸다.
api에 대한 지식이 부족했지만 FastAPI로 백엔드를 구현해보며 직접 코드를 작성하고 AI도 사용하니 개념에 대해 정립하는데 확실히 도움이 되었다.
