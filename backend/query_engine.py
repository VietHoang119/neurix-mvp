# backend/query_engine.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def encode_query(query_text):
    vec = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")

def search_index(df, index, query_text, top_k=5):
    qv = encode_query(query_text)
    distances, indices = index.search(qv, top_k)
    # chuyển L2 distance thành score (1/(1+dist))
    scores = (1 / (1 + distances[0])).tolist()
    rows = df.iloc[indices[0]].copy().reset_index(drop=True)
    rows["score"] = scores
    return rows
