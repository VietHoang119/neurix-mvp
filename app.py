# app.py
import streamlit as st
import pandas as pd
from backend.embedding import generate_embeddings, build_faiss_index
from backend.query_engine import search_index

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine - In-Memory MVP")

# 1. Upload & load data
uploaded = st.file_uploader("📁 Upload CSV/XLSX", type=["csv","xlsx"])
if uploaded and "df" not in st.session_state:
    try:
        st.session_state.df = (
            pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
            else pd.read_excel(uploaded)
        )
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")

df = st.session_state.get("df")
if df is not None:
    st.success(f"File đã load: {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(5))

    # 2. Build in-memory FAISS index
    if st.button("🚀 Build Semantic Memory"):
        with st.spinner("Đang embedding & build index..."):
            embs = generate_embeddings(df, max_rows=500)  # cho MVP chỉ embed 500 dòng đầu
            idx = build_faiss_index(embs)
            st.session_state.faiss_idx = idx
            st.success(f"✅ Tạo xong index với {idx.ntotal} vectors")

# 3. Query
if "faiss_idx" in st.session_state:
    st.subheader("💬 Semantic Query")
    q = st.text_input("Hỏi (VN):")
    if q:
        with st.spinner("Đang tìm kết quả..."):
            res = search_index(st.session_state.df, st.session_state.faiss_idx, q, top_k=5)
        st.write("Kết quả top 5:")
        st.dataframe(res)
