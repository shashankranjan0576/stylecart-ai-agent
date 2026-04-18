import streamlit as st
from agent import ask
import uuid

st.set_page_config(page_title="StyleCart AI", layout="centered")

st.title("🛍️ StyleCart AI Assistant")

# unique session id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
prompt = st.chat_input("Ask something...")

if prompt:
    # show user message
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # get response
    result = ask(prompt, st.session_state.thread_id)
    answer = result["answer"]

    # show bot message
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})