# app.py
import streamlit as st
import pandas as pd
import os

from backend.embedding import generate_embeddings, save_faiss_index

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine – MVP Demo")

st.markdown("Upload dữ liệu và để Neurix tạo trí nhớ ngữ nghĩa thông minh.")

uploaded_file = st.file_uploader("📁 Tải lên file dữ liệu (.csv hoặc .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Đọc dữ liệu
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File đã được tải lên thành công!")

        # Hiển thị thông tin sơ bộ
        st.subheader("📌 Tổng quan dữ liệu:")
        st.write(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
        st.dataframe(df.head(10))

        # Hiển thị schema
        st.subheader("📑 Cấu trúc bảng dữ liệu:")
        schema_info = pd.DataFrame({
            "Tên cột": df.columns,
            "Kiểu dữ liệu": [str(dtype) for dtype in df.dtypes]
        })
        st.table(schema_info)

        # Tạo embedding và lưu index
        if st.button("🚀 Tạo trí nhớ ngữ nghĩa từ dữ liệu"):
            with st.spinner("🔄 Đang tạo vector embeddings..."):
                vectors = generate_embeddings(df)
                save_faiss_index(vectors)
            st.success("✅ Đã lưu FAISS index tại models/faiss_index.index")

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")

else:
    st.info("📂 Vui lòng tải lên file dữ liệu để bắt đầu.")
