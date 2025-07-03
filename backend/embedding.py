# backend/embedding.py
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-small-v2")

def row_to_string(row):
    return " | ".join([f"{col}: {str(val)}" for col, val in row.items()])

def generate_embeddings(df, batch_size=32, max_rows=None):
    if max_rows:
        df = df.head(max_rows)

    texts = df.apply(row_to_string, axis=1).tolist()
    texts = [f"passage: {t}" for t in texts]

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
