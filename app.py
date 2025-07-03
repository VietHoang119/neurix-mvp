import streamlit as st
import pandas as pd
import os

from backend.embedding import generate_embeddings, save_faiss_index
from backend.query_engine import get_top_matches

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine - MVP")

uploaded_file = st.file_uploader("📁 Tải lên file dữ liệu (.csv hoặc .xlsx)", type=["csv", "xlsx"])

# Bảo lưu df vào session_state
if uploaded_file and "df" not in st.session_state:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {e}")

# Lấy lại df
df = st.session_state.get("df", None)

if df is not None:
    st.success("✅ File đã được tải lên thành công!")
    st.subheader("📌 Thông tin tổng quan:")
    st.write(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
    st.dataframe(df.head(10))

    if st.button("🔎 Phân tích & Lưu FAISS Index"):
        with st.spinner("🔄 Đang sinh embedding và lưu index..."):
            embeddings = generate_embeddings(df, max_rows=500)
            save_faiss_index(embeddings)

        if os.path.exists("models/faiss_index.index"):
            st.success("✅ Đã lưu FAISS index tại models/faiss_index.index")
        else:
            st.warning("⚠️ Chưa tạo được FAISS index. Kiểm tra log!")

# Truy vấn nếu có df và FAISS index
if df is not None and os.path.exists("models/faiss_index.index"):
    st.subheader("💬 Truy vấn ngữ nghĩa")
    query = st.text_input("Hỏi dữ liệu bằng tiếng Việt:")

    if query:
        try:
            results = get_top_matches(df, query)
            st.write(f"✅ Tìm thấy {len(results)} dòng phù hợp:")
            st.dataframe(results)
        except Exception as e:
            st.error(f"❌ Lỗi khi truy vấn: {e}")
