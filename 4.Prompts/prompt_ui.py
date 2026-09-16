import streamlit as st
from dotenv import load_dotenv

from langchain_core.load import load
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()


# -----------------------------
# Load prompt template
# -----------------------------

with open("template.json", "r", encoding="utf-8") as f:
    import json

    data = json.load(f)

summary_template = load(data)


# -----------------------------
# Create Hugging Face endpoint
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    max_new_tokens=2048,
    temperature=0.3,
)


# -----------------------------
# Convert into ChatModel
# -----------------------------

model = ChatHuggingFace(llm=llm)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered",
)

st.title("📚 My AI Academic Paper Summarizer")

st.write(
    "Summarize research papers using "
    "Hugging Face and LangChain."
)


# -----------------------------
# User inputs
# -----------------------------

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)


# -----------------------------
# Summarize button
# -----------------------------

if st.button("Summarize", type="primary"):

    with st.spinner("Generating explanation..."):

        try:

            # Fill the prompt placeholders
            prompt = summary_template.invoke(
                {
                    "paper_input": paper_input,
                    "style_input": style_input,
                    "length_input": length_input,
                }
            )

            # Send prompt to Hugging Face model
            result = model.invoke(prompt)

            # Display output
            st.subheader("Summary")

            st.write(result.content)

        except Exception as e:

            st.error(f"Error: {e}")