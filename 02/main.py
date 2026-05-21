from fastapi import FastAPI
from pydantic import BaseModel
import requests


app = FastAPI()

class Post(BaseModel):
    title: str
    content: str
    
posts = list()

@app.post('/posts')
async def create_post(post: Post):
    post_id = len(posts)
    posts.append({'id':post_id, 'title': post.title, 'content': post.content})
    return {'post_id':post_id, 'message': 'Post created'}

@app.get('/posts')
async def get_posts():
    return posts

@app.get('/posts/{post_id}')
async def get_post(post_id: int):
    return posts[post_id]

@app.put('/posts/{post_id}')
async def update_post(post_id: int, post: Post):
    posts[post_id] = post
    return post

@app.delete('/posts/{post_id}')
def delete_post(post_id: int):
    posts.pop(post_id)
    return {'message': 'Post deleted'}
