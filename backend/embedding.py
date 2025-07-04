# backend/embedding.py
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def load_model():
    # Chỉ load model 1 lần trong session
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

def row_to_string(row, max_words=100):
    text = " | ".join(f"{col}: {val}" for col, val in row.items())
    return " ".join(text.split()[:max_words])

@st.cache_data(show_spinner=False)
def generate_embeddings(df, batch_size=64, max_rows=None):
    """
    - @st.cache_data để cache kết quả nếu df và max_rows không đổi.
    - Chỉ lấy head(max_rows) để demo nhanh.
    """
    model = load_model()
    data = df.head(max_rows) if max_rows else df
    texts = [row_to_string(r) for _, r in data.iterrows()]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,  # tắt progress bar GUI
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embs.astype("float32")
