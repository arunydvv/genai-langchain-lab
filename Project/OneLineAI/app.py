"""
OneLine AI
==========
Type ONE line — an idea, a thought, a boring sentence, a topic, or a rough
problem — and OneLine AI creatively transforms it into something useful and
impressive. It detects your intent internally and returns a polished result
(a startup concept, a punchier rewrite, a mini story, an action plan, etc.).

Stack:
    - Streamlit          -> UI
    - LangChain          -> prompt construction + model invocation
    - Hugging Face       -> LLM (Inference Providers), the ONLY provider used

The Hugging Face token is loaded from a local ``.env`` file (variable
``HF_TOKEN``) via ``python-dotenv``. It is never hardcoded and never displayed.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Change the model in ONE place. Any current Hugging Face chat model that is
# served through the HF Inference/Inference-Providers endpoint works here.
# Good alternatives: "mistralai/Mistral-7B-Instruct-v0.3",
# "Qwen/Qwen2.5-7B-Instruct", "HuggingFaceH4/zephyr-7b-beta".
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are OneLine AI, a creative transformation engine.

The user gives you exactly ONE line: an idea, a thought, a sentence, a topic, \
a problem, or a rough concept. Do the following silently:

1. Detect the user's intent. Typical mappings:
   - rough idea            -> a sharp startup concept
   - dull/boring sentence  -> a powerful, vivid rewrite
   - a topic               -> a compact, engaging mini story
   - a problem             -> a concrete, actionable plan
   - a plain idea          -> an imaginative creative concept
2. Transform the input into something genuinely useful, concrete, and \
impressive that fits that intent.
3. Never reveal your reasoning, never restate the task, never name the intent \
or your steps, and never apologize or say "as an AI".

Return your answer in EXACTLY this format, with nothing before or after it:

TITLE: <a punchy title, max 8 words>
RESULT: <the transformed output — rich, specific, and well crafted; use short \
paragraphs or a tight list when it helps readability>
INSIGHT: <one or two sentences: either "why this works" or a concrete \
"next move". Keep it optional but prefer to include it.>

Keep it concise but high impact. Do not add any extra headings or commentary."""

EXAMPLES = [
    "an app that helps people remember to drink water",
    "the ocean at night",
    "I keep procrastinating on my side project",
    "make this sound better: our product is good and cheap",
]

# --------------------------------------------------------------------------- #
# Token handling
# --------------------------------------------------------------------------- #


def get_hf_token() -> str | None:
    """Load HF_TOKEN from the environment (.env) or Streamlit secrets.

    The token value itself is never returned to the UI or logged.
    """
    load_dotenv()  # populate os.environ from a local .env file if present

    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    # Optional fallback for Streamlit Cloud deployments. Accessing st.secrets
    # raises if no secrets file exists, so guard it.
    try:
        secret = st.secrets["HF_TOKEN"]  # type: ignore[index]
        if secret:
            return str(secret).strip()
    except Exception:
        pass

    return None


# --------------------------------------------------------------------------- #
# LangChain + Hugging Face chain
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def build_chain(model_name: str, token: str):
    """Build a cached LangChain chain: ChatPromptTemplate | ChatHuggingFace.

    Cached so the model client is created once per session rather than on
    every rerun.
    """
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        huggingfacehub_api_token=token,
    )
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{user_input}"),
        ]
    )
    return prompt | chat_model


def transform(chain, user_input: str) -> str:
    """Invoke the chain and return the raw model text."""
    response = chain.invoke({"user_input": user_input})
    # ChatHuggingFace returns an AIMessage.
    return getattr(response, "content", str(response)).strip()


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #


def parse_output(raw: str) -> dict[str, str]:
    """Parse the model's TITLE/RESULT/INSIGHT format into a dict.

    Falls back gracefully: if no TITLE marker is found, the whole text is
    treated as the result.
    """
    buckets: dict[str, list[str]] = {"title": [], "result": [], "insight": []}
    current: str | None = None

    for line in raw.splitlines():
        upper = line.strip().upper()
        if upper.startswith("TITLE:"):
            current = "title"
            buckets[current].append(line.split(":", 1)[1].strip())
        elif upper.startswith("RESULT:"):
            current = "result"
            buckets[current].append(line.split(":", 1)[1].strip())
        elif upper.startswith("INSIGHT:"):
            current = "insight"
            buckets[current].append(line.split(":", 1)[1].strip())
        elif current is not None:
            buckets[current].append(line)

    parsed = {key: "\n".join(vals).strip() for key, vals in buckets.items()}

    if not parsed["title"] and not parsed["result"]:
        # Model ignored the format; show the raw text as the result.
        parsed["result"] = raw.strip()

    return parsed


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _html_safe(text: str) -> str:
    """Escape HTML and preserve line breaks for safe card rendering."""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return escaped.replace("\n", "<br>")


def render_result(parsed: dict[str, str]) -> None:
    title = parsed.get("title") or "Your transformation"
    result = parsed.get("result", "")
    insight = parsed.get("insight", "")

    insight_html = ""
    if insight:
        insight_html = f"""
        <div class="ol-insight">
            <span class="ol-insight-label">Next move</span>
            <p>{_html_safe(insight)}</p>
        </div>
        """

    st.markdown(
        f"""
        <div class="ol-card">
            <h2 class="ol-card-title">{_html_safe(title)}</h2>
            <div class="ol-card-body">{_html_safe(result)}</div>
            {insight_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

CSS = """
<style>
    .stApp { background: radial-gradient(1200px 600px at 50% -10%,
             #1c1033 0%, #0d0a1a 45%, #08060f 100%); }
    #MainMenu, footer { visibility: hidden; }

    .ol-hero { text-align: center; margin: 0.5rem 0 1.75rem 0; }
    .ol-hero h1 {
        font-size: 3rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #a78bfa, #f0abfc, #7dd3fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.35rem;
    }
    .ol-hero p { color: #b8b3c9; font-size: 1.05rem; margin: 0; }

    .stTextInput > div > div > input {
        font-size: 1.05rem; padding: 0.85rem 1rem; border-radius: 14px;
    }

    div.stButton > button {
        width: 100%; border-radius: 14px; padding: 0.7rem 1rem;
        font-weight: 700; font-size: 1.02rem; border: 0;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed, #d946ef);
        color: white;
    }

    .ol-examples-label { color: #8f8aa3; font-size: 0.85rem;
        text-align: center; margin: 0.25rem 0 0.4rem 0; }

    .ol-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 20px; padding: 1.6rem 1.8rem; margin-top: 1.4rem;
        box-shadow: 0 20px 60px rgba(124,58,237,0.15);
    }
    .ol-card-title {
        color: #f5f3ff; font-size: 1.6rem; font-weight: 800;
        margin: 0 0 0.9rem 0; letter-spacing: -0.01em;
    }
    .ol-card-body { color: #ddd8ea; font-size: 1.05rem; line-height: 1.6; }
    .ol-insight {
        margin-top: 1.25rem; padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    .ol-insight-label {
        display: inline-block; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em; color: #c4b5fd;
        background: rgba(167,139,250,0.12); padding: 0.2rem 0.6rem;
        border-radius: 999px; margin-bottom: 0.4rem;
    }
    .ol-insight p { color: #c8c3d8; margin: 0.3rem 0 0 0; }

    .ol-footer { text-align: center; color: #6f6a80; font-size: 0.82rem;
        margin-top: 2.5rem; }
</style>
"""


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def _use_example(text: str) -> None:
    """Callback: prefill the input with an example (runs before rerun)."""
    st.session_state.user_input = text


def main() -> None:
    st.set_page_config(page_title="OneLine AI", page_icon="✨", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ol-hero">
            <h1>OneLine AI</h1>
            <p>One line in. An intelligent transformation out.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    token = get_hf_token()
    if not token:
        st.error(
            "HF_TOKEN not found. Create a `.env` file with "
            "`HF_TOKEN=your_huggingface_token_here` (see `.env.example`), "
            "then restart the app."
        )
        st.stop()

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    user_input = st.text_input(
        "Your one line",
        key="user_input",
        placeholder="e.g. an app that helps people remember to drink water",
        label_visibility="collapsed",
    )

    go = st.button("Transform ✨", type="primary", use_container_width=True)

    # Example suggestions
    st.markdown('<p class="ol-examples-label">Try one of these</p>',
                unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        with col:
            short = example if len(example) <= 28 else example[:27] + "…"
            st.button(short, key=f"ex_{example}", help=example,
                      on_click=_use_example, args=(example,),
                      use_container_width=True)

    if go:
        text = st.session_state.user_input.strip()
        if not text:
            st.warning("Type a line first, then hit Transform.")
            st.stop()

        try:
            chain = build_chain(MODEL_NAME, token)
            with st.spinner("Transforming…"):
                raw = transform(chain, text)
            render_result(parse_output(raw))
        except Exception as exc:  # surface a clean message, hide internals
            st.error(
                "Something went wrong while contacting Hugging Face. "
                "Check that your token has inference permissions and that the "
                f"model is available.\n\nDetails: {exc}"
            )

    st.markdown(
        '<p class="ol-footer">Powered by Hugging Face 🤗 · '
        "Built with Streamlit &amp; LangChain</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()