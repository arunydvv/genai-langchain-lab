from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

import streamlit as st
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Create Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
)

# Convert into ChatModel
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("My AI Chatbot")

user_input = st.text_input("Enter your PROMPT!")

if st.button("Submit"):

    result = model.invoke(user_input)
    st.write(result.content)

