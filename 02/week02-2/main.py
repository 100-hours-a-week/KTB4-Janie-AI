from fastapi import FastAPI
from database import Base, engine
from routers import posts, comments

app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(posts.router)
app.include_router(comments.router)