# backend/rag.py
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()
# Khởi tạo client theo API v1
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_with_rag(df_rows: pd.DataFrame, query: str) -> str:
    """
    Lấy top-k rows rồi build prompt cho OpenAI ChatCompletion (API v1),
    trả về câu trả lời tự nhiên.
    """
    # Chuyển DataFrame thành CSV nhỏ gọn
    csv_data = df_rows.to_csv(index=False)

    system_prompt = (
        "You are a helpful AI assistant. "
        "Given the following CSV data and a question, "
        "provide a concise, natural-language answer."
    )
    user_prompt = (
        f"CSV data:\n{csv_data}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Sử dụng client.chat.completions.create của v1
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=256
    )
    return resp.choices[0].message.content.strip()
