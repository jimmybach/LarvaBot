import sys
from pathlib import Path
import gc

import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------
# Paths
# -------------------

ROOT = Path(__file__).resolve().parents[1]  # Larva_Bot/
sys.path.insert(0, str(ROOT))

from src.make_chat import chat_with_arvind as make_chat, clear_chat

ASSETS_DIR = ROOT / "assets"
LARVA_IMG = ASSETS_DIR / "larva.jpg"
YUNGVIND_IMG = ASSETS_DIR / "yungvind.jpg"

# -------------------
# Page config
# -------------------

st.set_page_config(
    page_title="LarvaBot",
    page_icon="🐛",
    layout="wide"
)

MODEL_OPTIONS = {
    "Qwen 0.6B (Faster, less accurate)": {
        "repo": "Qwen/Qwen3-0.6B",
        "subfolder": None,
    },
    "Qwen 4B Finetuned (Slower, more accurate)": {
        "repo": "jimmybach33/larvabot-4b",
        "subfolder": "arvind-merged",
    },
}

@st.cache_resource
def load_model(model_name, subfolder=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        subfolder=subfolder,
    )

    return tokenizer, model

# -------------------
# Session state
# -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name" not in st.session_state:
    st.session_state.name = "You"

# -------------------
# Sidebar
# -------------------

with st.sidebar:
    if YUNGVIND_IMG.exists():
        img = Image.open(YUNGVIND_IMG).rotate(-90, expand=True)
        st.image(img, use_container_width=True)

    st.title("LarvaBot")

    selected_model = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys())
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.7,
        help="Lower = more deterministic. Higher = more creative."
    )

    st.text_input("Your name", key="name")

    if st.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.session_state.messages = []
        st.rerun()

# -------------------
# Model switching
# -------------------

if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model

if selected_model != st.session_state.current_model:
    st.cache_resource.clear()
    gc.collect()
    st.session_state.current_model = selected_model
    st.rerun()

model_name = MODEL_OPTIONS[selected_model]['repo']
model_subfolder = MODEL_OPTIONS[selected_model]['subfolder']

with st.spinner("Loading model..."):
    tokenizer, model = load_model(model_name, subfolder=model_subfolder)

# -------------------
# Main page
# -------------------

if LARVA_IMG.exists():
    st.image(str(LARVA_IMG), width=180)

st.title("LarvaBot")
st.caption("Berman Admin really out here making him AI :(")
st.divider()

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.name}:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Arvind:** {message['content']}")

user_input = st.chat_input("Message Arvind...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.name}:** {user_input}")

    with st.chat_message("assistant"):
        with st.spinner("Arvind is locking in..."):
            response = make_chat(
                tokenizer,
                model,
                user_input,
                temperature=temperature
            )

        st.markdown(f"**Arvind:** {response}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )