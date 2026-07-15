from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME

# embeddings = UpstageEmbeddings(
#     api_key = os.getenv('UPSTAGE_API_KEY'),
#     model = 'solar-embedding-2-query'
# )
# embeddings = GoogleGenerativeAIEmbeddings(
#     api_key=os.getenv('GEMINI_API_KEY'),
#     model = 'models/gemini-embedding-001'
# )

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)
