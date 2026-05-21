from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from ollama import AsyncClient


app = FastAPI()
ollama = AsyncClient()

class PostCreate(BaseModel):
    title: str
    content: str

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

# 임시 저장소
posts = list()

def find_post(post_id: int):
    for post in posts:
        if post['id'] == post_id:
            return post
    return None

# 게시글 요약 함수(Ollama 활용)
async def summarize_content(content:str):
    response = await ollama.chat(model='gemma4:e4b', 
                           messages=[{'role': 'system', 'content': '너는 요약을 담당하는 AI야'},
                                     {'role': 'user', 'content': f'게시글을 3문장 이내로 요약해줘:\n {content}'}])
    return response['message']['content']


# 게시물 생성
@app.post('/posts', status_code=201)
async def create_post(post: PostCreate):
    post_id = max([p['id'] for p in posts], default=0) + 1
    summary = await summarize_content(post.content)
    posts.append({'id':post_id, 'title': post.title, 'content': post.content, 'summary': summary})
    return {'post_id':post_id, 'message': 'Post created'}

# 게시물 목록 조회
@app.get('/posts')
async def get_posts():
    return posts


# 게시물 조회
@app.get('/posts/{post_id}')
async def get_post(post_id: int):
    post = find_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail='Post not found')
    return post

# 게시물 수정
@app.patch('/posts/{post_id}')
async def update_post(post_id: int, post: PostUpdate):
    existing_post = find_post(post_id)
    if not existing_post:
        raise HTTPException(status_code = 404, detail='Post not found')
    update_data = post.model_dump(exclude_unset=True)
    
    # 내용 수정시 요약 다시 생성
    if 'content' in update_data:
        update_data['summary'] = await summarize_content(update_data['content'])  
        existing_post['summary'] = update_data['summary']

    for key, value in update_data.items():
        existing_post[key] = value

    return existing_post 

# 게시물 삭제
@app.delete('/posts/{post_id}')
def delete_post(post_id: int):
    existing_post = find_post(post_id)
    if not existing_post:
        raise HTTPException(status_code=404, detail='Post not found')
    posts.remove(existing_post)
    return {'message': 'Post deleted'}


