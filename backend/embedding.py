# backend/embedding.py
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def row_to_string(row, max_words=100):
    text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
    return " ".join(text.split()[:max_words])

def generate_embeddings(df, batch_size=64, max_rows=None):
    if max_rows:
        df = df.head(max_rows)

    texts = df.apply(row_to_string, axis=1).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")

def build_faiss_index(embeddings):
    """Xây dựng và trả về FAISS index trong memory (không cần lưu file)"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index_to_file(index, index_path="models/faiss_index.index"):
    """Nếu bạn vẫn muốn lưu ra file, có thể gọi thêm hàm này."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
