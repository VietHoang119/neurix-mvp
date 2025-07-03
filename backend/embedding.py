# backend/embedding.py
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Tải mô hình embedding DeepSeek
model = SentenceTransformer("deepseek-ai/deepseek-embedding-v2")

def row_to_string(row):
    """Ghép thông tin dòng thành 1 chuỗi mô tả ngữ nghĩa"""
    return " | ".join([f"{col}: {str(val)}" for col, val in row.items()])

def generate_embeddings(df):
    """Tạo vector embedding từ từng dòng dữ liệu"""
    texts = df.apply(row_to_string, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

def save_faiss_index(embeddings, index_path="models/faiss_index.index"):
    """Lưu các vectors vào FAISS index"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
