import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_context(df: pd.DataFrame) -> str:
    """Build a summary of the dataframe to give the AI context."""
    context = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

First 5 rows:
{df.head().to_string()}

Basic Statistics:
{df.describe().to_string()}

Missing Values:
{df.isnull().sum().to_string()}
"""
    return context


def ask_agent(df: pd.DataFrame, question: str) -> str:
    """Send the question + dataset context to Groq LLM and return the answer."""
    context = build_context(df)

    prompt = f"""You are a helpful data analyst assistant. 
You have been given a dataset with the following information:

{context}

The user asks: {question}

Answer clearly and concisely based on the data provided. 
If the question is about data quality, mention specific column names and numbers.
If you can calculate something from the statistics provided, do so.
Keep your answer under 150 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )

    return response.choices[0].message.content
    