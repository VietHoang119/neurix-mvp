# backend/query_engine.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load lại model giống embedding.py
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def preprocess_query(query_text):
    """Tiền xử lý truy vấn (nếu cần mở rộng về sau)"""
    return query_text.strip()

def encode_query(query_text):
    """Tạo embedding từ câu hỏi"""
    query_embedding = model.encode(
        [preprocess_query(query_text)],
        normalize_embeddings=True
    )
    return np.array(query_embedding).astype("float32")

def load_faiss_index(index_path="models/faiss_index.index"):
    """Tải FAISS index đã tạo"""
    return faiss.read_index(index_path)

def search_faiss_index(query_vector, index, top_k=5):
    """Tìm top_k kết quả gần nhất"""
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]  # Trả về danh sách vị trí và khoảng cách

def get_top_matches(df, query_text, index_path="models/faiss_index.index", top_k=5):
    """Tìm top_k dòng trong df gần nhất với câu truy vấn"""
    query_vec = encode_query(query_text)
    index = load_faiss_index(index_path)
    matched_indices, distances = search_faiss_index(query_vec, index, top_k=top_k)

    results = df.iloc[matched_indices].copy()
    results["🔍 Similarity Score"] = (1 - distances).round(4)  # FAISS dùng L2, mình đổi sang score gần đúng
    return results
