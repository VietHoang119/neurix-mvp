# app.py
import streamlit as st
import pandas as pd
from backend.embedding import generate_embeddings
from backend.query_engine import get_top_matches
from backend.rag import answer_with_rag

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine – In-Memory + RAG")

# 1) Upload & Load Data
uploaded = st.file_uploader("📁 Upload CSV/XLSX", type=["csv","xlsx"])
if uploaded and "df" not in st.session_state:
    try:
        st.session_state.df = (
            pd.read_csv(uploaded)
            if uploaded.name.lower().endswith(".csv")
            else pd.read_excel(uploaded)
        )
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")

df = st.session_state.get("df")
if df is not None:
    st.success(f"✅ Data loaded: {df.shape[0]}×{df.shape[1]}")
    st.dataframe(df.head(5), use_container_width=True)

    # 2) Build embeddings
    if "embeddings" not in st.session_state:
        if st.button("🚀 Build Semantic Memory"):
            with st.spinner("🔄 Generating embeddings…"):
                embs = generate_embeddings(df, max_rows=500)
                st.session_state.embeddings = embs
            st.success(f"✅ Built embeddings: {embs.shape}")

# 3) Semantic Query + RAG Answer
if "embeddings" in st.session_state:
    st.subheader("💬 Semantic Query")
    query = st.text_input("Hỏi dữ liệu (tiếng Việt):")
    if query:
        with st.spinner("🔍 Searching…"):
            results = get_top_matches(
                st.session_state.df,
                st.session_state.embeddings,
                query,
                top_k=5
            )
        st.write("Top 5 kết quả gần nhất:")
        st.dataframe(results, use_container_width=True)
        
        # 4) Natural-language answer
        if st.button("📝 Giải thích tự nhiên"):
            with st.spinner("✍️ Generating natural language answer…"):
                answer = answer_with_rag(results, query)
            st.markdown(f"**Neurix trả lời:** {answer}")
