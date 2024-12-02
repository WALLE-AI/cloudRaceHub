from models.llm import LLMApi
import streamlit as st
##加载env file
from dotenv import load_dotenv
load_dotenv()

st.title(":sunglasses: _StarChat_")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "qwen-plus"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "system", "content": "你是starchat智能助手"})
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = LLMApi.llm_client("tongyi").chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})