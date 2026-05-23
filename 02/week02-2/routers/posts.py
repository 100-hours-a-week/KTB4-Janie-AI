from fastapi import APIRouter
from schemas.post import PostCreate, PostUpdate
import controllers.posts as post_controller

router = APIRouter()

@router.post('/posts', status_code=201)
async def create_post(post: PostCreate):
    return await post_controller.create_post(post)

@router.get('/posts')
def get_posts():
    return post_controller.get_posts()

@router.get('/posts/{post_id}')
def get_post(post_id: int):
    return post_controller.get_post(post_id)

@router.patch('/posts/{post_id}')
async def update_post(post_id: int, post: PostUpdate):
    return await post_controller.update_post(post_id, post)

@router.delete('/posts/{post_id}')
def delete_post(post_id: int):
    return post_controller.delete_post(post_id)