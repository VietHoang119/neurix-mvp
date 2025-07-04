# backend/embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer

# Load MiniLM model once
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def row_to_string(row, max_words=100):
    """
    Ghép mỗi dòng DataFrame thành một chuỗi, cắt tối đa max_words từ
    để giảm độ dài input và tăng tốc encoding.
    """
    text = " | ".join(f"{col}: {val}" for col, val in row.items())
    return " ".join(text.split()[:max_words])

def generate_embeddings(df, batch_size=64, max_rows=None):
    """
    Sinh embedding cho từng dòng trong DataFrame.
    - max_rows: nếu không None, chỉ lấy head(max_rows) để demo nhanh.
    - Trả về numpy array shape (n_rows, dim).
    """
    # Giới hạn số dòng nếu cần
    data = df.head(max_rows) if max_rows else df

    # Chuyển thành list văn bản
    texts = [row_to_string(r) for _, r in data.iterrows()]

    # Encode thành embedding
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")
