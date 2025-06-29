import streamlit as st
from backend import get_response
from vector_db import load_vector_db
import os
from emotion import detect_emotion
from tts import generate_tts  # ✅ Your TTS module

# Load vector DB
vector_db = load_vector_db()

# Streamlit settings
st.set_page_config(
    page_title="MindMate",
    page_icon="🧠",
    layout="wide"
)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#Header
st.markdown("""
<div class="header-container">
    <h1>🧠 MindMate</h1>
    <p>I'm here to support you. Ask me anything related to mental health.</p>
</div>
""", unsafe_allow_html=True)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render messages
for sender, message in st.session_state.messages:
    if sender == "You":
        st.markdown(f'<div class="user-bubble">🧍 {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🤖 {message}</div>', unsafe_allow_html=True)

# Close box
st.markdown("</div></div>", unsafe_allow_html=True)

# ✅ Chat input
if user_input := st.chat_input("Ask your question..."):

    # Step 1: Store user input
    st.session_state.messages.append(("You", user_input))

    # Step 2: Detect emotion
    emotion, confidence = detect_emotion(user_input)
    emotion_msg = f"🧠 Detected Emotion: **{emotion.capitalize()}** ({confidence:.2%} confidence)"
    st.session_state.messages.append(("Bot", emotion_msg))

    # Step 3: Get bot reply
    with st.spinner("🤖 Thinking..."):
        response = get_response(user_input, vector_db)

    # Step 4: Add response to chat
    st.session_state.messages.append(("Bot", response))

    # Step 5: Rerun to display text, then add audio
    st.session_state.last_response = response  # 🔄 Save for audio
    st.rerun()

# ✅ Display TTS only AFTER rerun (to ensure full text shown first)
if "last_response" in st.session_state:
    audio_path = generate_tts(st.session_state.last_response)
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/mp3')
    os.remove(audio_path)  # Cleanup
    del st.session_state["last_response"]
