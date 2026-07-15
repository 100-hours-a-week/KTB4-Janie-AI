from chromadb.api.shared_system_client import SharedSystemClient
SharedSystemClient.clear_system_cache()
from langchain_chroma import Chroma

from config import CHROMA_DB_DIR
from embedder import embeddings

# batch_size = 500
vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# retriever
spotify_retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5}
)

print(f'Retriever 객체: {type(spotify_retriever).__name__}')

#shutil.rmtree('./chroma_db', ignore_errors=True)
