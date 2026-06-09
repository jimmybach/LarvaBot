import streamlit as st
import os
print(os.getcwd())
print(os.listdir())
os.chdir('mount/src/larvabot/Larva_Bot/app')
print(os.listdir())
from src.make_chat import chat_with_arvind as make_chat, clear_chat
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from PIL import Image

st.set_page_config(
    page_title="LarvaBot",
    page_icon="🐛",
    layout="wide"
)

MODEL_OPTIONS = {
    "Qwen 0.6B (Faster, less accurate)": "Qwen/Qwen3-0.6B",
    "Qwen 4B Finetuned (Slower, more accurate)": "jimmybach33/larvabot-4b",
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    return tokenizer, model


# -------------------
# Sidebar
# -------------------
st.image("./larva.jpg")

with st.sidebar:
    img=Image.open("yungvind.jpg").rotate(-90,expand=True)
    st.image(img)
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

model_name = MODEL_OPTIONS[selected_model]

with st.spinner("Loading model..."):
    tokenizer, model = load_model(model_name)


# -------------------
# Main page
# -------------------

st.title("LarvaBot")
st.caption("Berman Admin really out here making him AI :(")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name" not in st.session_state:
    st.session_state.name = "You"

with st.expander("User settings", expanded=False):
    st.text_input("Your name", key="name")


# Display previous messages every rerun
for message in st.session_state.messages:
    role = message["role"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.name}:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Arvind:** {message['content']}")


# Chat input
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