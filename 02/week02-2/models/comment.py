from sqlalchemy import Column, Integer, Text
from database import Base

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)