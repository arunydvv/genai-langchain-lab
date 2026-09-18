# OneLine AI ✨

Type **one line** — an idea, a thought, a boring sentence, a topic, or a rough
problem — and OneLine AI creatively transforms it into something useful and
impressive. It detects your intent internally and returns a polished result:
a startup concept, a punchier rewrite, a mini story, an action plan, or a
creative concept.

This is **not** a chatbot. The entire experience is: *one line in → one
intelligent transformation out.*

## Stack

- **Streamlit** — the UI
- **LangChain** (`langchain-core`, `langchain-huggingface`) — prompt
  construction and model invocation
- **Hugging Face Inference** — the **only** LLM provider used. No OpenAI,
  Gemini, Anthropic, local models, or paid APIs.

The app calls a hosted model through the Hugging Face inference service — it
does **not** download or run a model locally, so it starts fast and needs no
GPU.

## Setup

### 1. Create and activate a virtual environment

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Create a Hugging Face token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **inference** permission (a "Read" token with
   inference access, or a fine-grained token with "Make calls to Inference
   Providers" enabled).
3. Copy the token — you only see it once.

### 4. Create your `.env` file

    cp .env.example .env

Then edit `.env`:

    HF_TOKEN=hf_your_real_token_here

### 5. Run the app

    streamlit run app.py

## Changing the model

Set it in one place at the top of `app.py`:

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

Alternatives: `mistralai/Mistral-7B-Instruct-v0.3`,
`Qwen/Qwen2.5-7B-Instruct`, `HuggingFaceH4/zephyr-7b-beta`. Some models are
gated and need you to accept their license on the model page first.