from fastapi import HTTPException
from ollama import AsyncClient
from database import SessionLocal
from models.post import Post

ollama = AsyncClient()

async def summarize_content(content: str):
    response = await ollama.chat(model='gemma4:e4b',
                           messages=[{'role': 'system', 'content': '너는 요약을 담당하는 AI야'},
                                     {'role': 'user', 'content': f'게시글을 3문장 이내로 요약해줘:\n {content}'}])
    return response['message']['content']

async def create_post(post):
    summary = await summarize_content(post.content)
    new_post = Post(title=post.title, content=post.content, summary=summary)
    with SessionLocal() as session:
        session.add(new_post)
        session.commit()
        session.refresh(new_post)
    return new_post

def get_posts():
    with SessionLocal() as session:
        return session.query(Post).all()

def get_post(post_id: int):
    with SessionLocal() as session:
        post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail='Post not found')
    return post

async def update_post(post_id: int, post):
    with SessionLocal() as session:
        existing_post = session.query(Post).filter(Post.id == post_id).first()
        if not existing_post:
            raise HTTPException(status_code=404, detail='Post not found')
        update_data = post.model_dump(exclude_unset=True)
        if 'content' in update_data:
            update_data['summary'] = await summarize_content(update_data['content'])
        for key, value in update_data.items():
            setattr(existing_post, key, value)
        session.commit()
        session.refresh(existing_post)
    return existing_post

def delete_post(post_id: int):
    with SessionLocal() as session:
        existing_post = session.query(Post).filter(Post.id == post_id).first()
        if not existing_post:
            raise HTTPException(status_code=404, detail='Post not found')
        session.delete(existing_post)
        session.commit()
    return {'message': 'Post deleted'}