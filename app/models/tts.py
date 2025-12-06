# app/models/tts.py

import os

from gtts import gTTS
from IPython.display import Audio

from config import ANSWERS_DIR

class TextToSpeech:
    def __init__(self, output_dir=ANSWERS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _speak(self, text, filename="answer.mp3"):
        path = os.path.join(self.output_dir, filename)
        tts = gTTS(text, lang="es", tld="es")
        tts.save(path)
        return Audio(path, autoplay=True)
