# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Neurix MVP", layout="wide")
st.title("🧠 Neurix Memory Engine – MVP Demo")

st.markdown("Upload dữ liệu và để Neurix phân tích tự động.")

# Tải file
uploaded_file = st.file_uploader("📁 Tải lên file dữ liệu (.csv hoặc .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Đọc file CSV hoặc Excel
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Thông báo thành công
        st.success("✅ File đã được tải lên thành công!")

        # Hiển thị thông tin tổng quan
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

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")

else:
    st.info("📂 Vui lòng tải lên file dữ liệu để bắt đầu.")
