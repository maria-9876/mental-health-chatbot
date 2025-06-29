from gtts import gTTS
import uuid
import os

def generate_tts(text):
    # Generate a unique filename
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    return filename
