# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine – In-Memory MVP")

# Cache đọc file, tránh load lại khi chỉnh sửa UI
@st.cache_data(show_spinner=False)
def read_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)

# 1) Upload & Load
uploaded = st.file_uploader("📁 Upload CSV / XLSX", type=["csv", "xlsx"])
if uploaded and "df" not in st.session_state:
    try:
        st.session_state.df = read_file(uploaded)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")

df = st.session_state.get("df")
if df is not None:
    st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(5), use_container_width=True)

    # 2) Build embeddings only khi nhấn nút
    if "embeddings" not in st.session_state:
        if st.button("🚀 Build Semantic Memory"):
            with st.spinner("🔄 Generating embeddings…"):
                # import lazy để không block import app.py
                from backend.embedding import generate_embeddings
                embs = generate_embeddings(df, max_rows=500)
                st.session_state.embeddings = embs
            st.success(f"✅ Built embeddings: shape {embs.shape}")

# 3) Semantic Query
if "embeddings" in st.session_state:
    st.subheader("💬 Semantic Query")
    q = st.text_input("Hỏi dữ liệu (tiếng Việt):")
    if q:
        with st.spinner("🔍 Searching…"):
            from backend.query_engine import get_top_matches
            result = get_top_matches(df, st.session_state.embeddings, q, top_k=5)
        st.write("Top 5 kết quả:")
        st.dataframe(result, use_container_width=True)
