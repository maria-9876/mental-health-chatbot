import streamlit as st
import os
from dotenv import load_dotenv

from backend import get_response
from vector_db import load_vector_db
from emotion import detect_emotion
from tts import generate_tts  # ✅ Your TTS module

# ✅ Load environment variables from .env (only locally)
load_dotenv()

# ✅ Get secrets (Streamlit Cloud preferred, fallback to .env for local)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
GROQ_MODEL = st.secrets.get("GROQ_MODEL", os.getenv("GROQ_MODEL"))
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# ✅ Load vector DB (you must ensure the DB path works in cloud too)
vector_db = load_vector_db()

# ✅ Streamlit settings
st.set_page_config(
    page_title="MindMate",
    page_icon="🧠",
    layout="wide"
)

# ✅ Load custom CSS (ensure assets/style.css exists in your repo)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ✅ Header section
st.markdown("""
<div class="header-container">
    <h1>🧠 MindMate</h1>
    <p>I'm here to support you. Ask me anything related to mental health.</p>
</div>
""", unsafe_allow_html=True)

# ✅ Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Render past chat messages
for sender, message in st.session_state.messages:
    if sender == "You":
        st.markdown(f'<div class="user-bubble">🧍 {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🤖 {message}</div>', unsafe_allow_html=True)

# ✅ Chat input box
if user_input := st.chat_input("Ask your question..."):

    # Step 1: Save user message
    st.session_state.messages.append(("You", user_input))

    # Step 2: Detect emotion
    emotion, confidence = detect_emotion(user_input)
    emotion_msg = f"🧠 Detected Emotion: **{emotion.capitalize()}** ({confidence:.2%} confidence)"
    st.session_state.messages.append(("Bot", emotion_msg))

    # Step 3: Get response from QA backend
    with st.spinner("🤖 Thinking..."):
        response = get_response(user_input, vector_db)

    # Step 4: Save response
    st.session_state.messages.append(("Bot", response))

    # Step 5: Save for TTS after rerun
    st.session_state.last_response = response
    st.rerun()

# ✅ After rerun, generate TTS audio
if "last_response" in st.session_state:
    audio_path = generate_tts(st.session_state.last_response)
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/mp3')
    os.remove(audio_path)
    del st.session_state["last_response"]
