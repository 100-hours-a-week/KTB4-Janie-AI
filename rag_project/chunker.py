import re

from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def split_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks
