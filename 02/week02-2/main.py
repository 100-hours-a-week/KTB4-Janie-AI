from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
import ollama

app = FastAPI()

class PostCreate(BaseModel):
    title: str
    content: str

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


posts = list()

def find_post(post_id: int):
    for post in posts:
        if post['id'] == post_id:
            return post
    return None

def generate_summary(content:str):
    response = ollama.chat(model='gemma4:e4b', 
                           messages=[{'role': 'system', 'content': '너는 요약을 담당하는 AI야'},
                                     {'role': 'user', 'content': f'다음 글을 50자 이내로 요약해줘:\n {content}'}])
    return response['message']['content']


# 게시물 생성
@app.post('/posts', status_code=201)
def create_post(post: PostCreate):
    post_id = max([p['id'] for p in posts], default=0) + 1
    posts.append({'id':post_id, 'title': post.title, 'content': post.content})
    return {'post_id':post_id, 'message': 'Post created'}

# 게시물 목록 조회
@app.get('/posts')
def get_posts():
    return posts

# 게시물 조회
@app.get('/posts/{post_id}')
def get_post(post_id: int):
    post = find_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail='Post not found')
    return post

# 게시물 조회(요약)
@app.get('/posts/{post_id}/summary')
def get_post_summary(post_id: int):
    post = find_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail='Post not found')
    summary = generate_summary(post['content'])
    return {'summary': summary}

# 게시물 수정
@app.patch('/posts/{post_id}')
def update_post(post_id: int, post: PostUpdate):
    existing_post = find_post(post_id)
    if not existing_post:
        raise HTTPException(status_code = 404, detail='Post not found')
    update_data = post.model_dump(exclude_unset=True)
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

