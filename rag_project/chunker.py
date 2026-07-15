from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs, chunk_size=400, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
