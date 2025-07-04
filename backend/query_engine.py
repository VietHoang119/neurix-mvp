# backend/query_engine.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def encode_query(q):
    return model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]

def search_embeddings(embs: np.ndarray, qv: np.ndarray, top_k=5):
    dists = np.linalg.norm(embs - qv, axis=1)
    idxs = np.argsort(dists)[:top_k]
    scores = (1 / (1 + dists[idxs])).round(4)
    return idxs, scores

def get_top_matches(df, embeddings, query_text, top_k=5):
    qv = encode_query(query_text)
    idxs, scores = search_embeddings(embeddings, qv, top_k)
    res = df.iloc[idxs].copy().reset_index(drop=True)
    res["score"] = scores
    return res
