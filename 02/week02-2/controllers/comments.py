from fastapi import HTTPException
from database import SessionLocal
from models.post import Post
from models.comment import Comment

def create_comment(post_id: int, comment):
    with SessionLocal() as session:
        post = session.query(Post).filter(Post.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail='Post not found')
        new_comment = Comment(post_id=post_id, content=comment.content)
        session.add(new_comment)
        session.commit()
        session.refresh(new_comment)
    return new_comment

def get_comments(post_id: int):
    with SessionLocal() as session:
        return session.query(Comment).filter(Comment.post_id == post_id).all()

def update_comment(post_id: int, comment_id: int, comment):
    with SessionLocal() as session:
        existing_comment = session.query(Comment).filter(Comment.id == comment_id).first()
        if not existing_comment:
            raise HTTPException(status_code=404, detail='Comment not found')
        update_data = comment.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(existing_comment, key, value)
        session.commit()
        session.refresh(existing_comment)
    return existing_comment

def delete_comment(post_id: int, comment_id: int):
    with SessionLocal() as session:
        existing_comment = session.query(Comment).filter(Comment.id == comment_id).first()
        if not existing_comment:
            raise HTTPException(status_code=404, detail='Comment not found')
        session.delete(existing_comment)
        session.commit()
    return {'message': 'Comment deleted'}