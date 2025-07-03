# backend/embedding.py
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cấu hình OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def row_to_string(row):
    """Ghép các cột thành một câu mô tả dòng dữ liệu"""
    return " | ".join([f"{col}: {str(val)}" for col, val in row.items()])

def get_embedding(text):
    """Gửi đoạn text đến OpenAI để lấy embedding"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def generate_embeddings(df):
    """Tạo danh sách embedding cho từng dòng trong DataFrame"""
    sentences = df.apply(row_to_string, axis=1).tolist()
    embeddings = [get_embedding(text) for text in sentences]
    return np.array(embeddings).astype("float32")

def save_faiss_index(embeddings, index_path="models/faiss_index.index"):
    """Lưu các embedding vào FAISS index"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
