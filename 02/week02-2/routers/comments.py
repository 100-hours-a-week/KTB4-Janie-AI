from fastapi import APIRouter
from schemas.comment import CommentCreate, CommentUpdate
import controllers.comments as comment_controller

router = APIRouter()

@router.post('/posts/{post_id}/comments', status_code=201)
def create_comment(post_id: int, comment: CommentCreate):
    return comment_controller.create_comment(post_id, comment)

@router.get('/posts/{post_id}/comments')
def get_comments(post_id: int):
    return comment_controller.get_comments(post_id)

@router.patch('/posts/{post_id}/comments/{comment_id}')
def update_comment(post_id: int, comment_id: int, comment: CommentUpdate):
    return comment_controller.update_comment(post_id, comment_id, comment)

@router.delete('/posts/{post_id}/comments/{comment_id}')
def delete_comment(post_id: int, comment_id: int):
    return comment_controller.delete_comment(post_id, comment_id)