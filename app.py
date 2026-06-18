import streamlit as st
import pandas as pd
from datachat.agent import ask_agent

st.set_page_config(page_title="DataChat", page_icon="🤖", layout="wide")

st.title("🤖 DataChat — Talk to Your Dataset")
st.markdown("Upload any CSV and ask questions in plain English!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    st.divider()
    st.subheader("💬 Ask anything about your data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    question = st.chat_input("e.g. Which column has the most missing values?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_agent(df, question)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            