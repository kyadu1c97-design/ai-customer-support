import streamlit as st
import requests

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Header + Clear Button
col1, col2 = st.columns([10, 1])

with col1:
    st.markdown("### 🤖 AI Customer Support Chatbot")

with col2:
    if st.button("🗑"):
        st.session_state.messages = []

# Chat History Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#  User Input
user_input = st.chat_input("Type your question...")

if user_input and user_input.strip():

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    #  Get bot response
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"question": user_input},
                timeout=20
            )

            result = response.json()

            bot_reply = result.get("answer", "No response")
            sources = result.get("sources", [])

            #  Add sources (optional)
            if sources:
                bot_reply += f"\n\n📌 Sources: {', '.join(sources)}"

    except requests.exceptions.ConnectionError:
        bot_reply = "❌ Backend not running"

    except requests.exceptions.Timeout:
        bot_reply = "⏳ Server taking too long to respond"

    except:
        bot_reply = "⚠️ Something went wrong"

    # Show bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)