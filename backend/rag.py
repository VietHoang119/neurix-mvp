# backend/rag.py
import os
import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_with_rag(df_rows: pd.DataFrame, query: str) -> str:
    """
    Lấy top-k rows rồi build prompt cho OpenAI ChatCompletion,
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

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=256
    )
    return resp.choices[0].message["content"].strip()
