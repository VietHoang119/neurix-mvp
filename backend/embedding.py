# backend/embedding.py (cập nhật)
import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Mô hình cực nhanh, dùng tốt cho semantic search
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def row_to_string(row, max_words=100):
    """Cắt ngắn input để tăng tốc"""
    text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
    return " ".join(text.split()[:max_words])  # Giới hạn 100 từ

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

def save_faiss_index(embeddings, index_path="models/faiss_index.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
