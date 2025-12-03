# app/audio/tts.py

import os

from gtts import gTTS
from IPython.display import Audio

class TextToSpeech:
    def __init__(self, output_dir="audio_out"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def say(self, texto, filename="respuesta.mp3"):
        path = os.path.join(self.output_dir, filename)
        tts = gTTS(texto, lang="es", tld="es")
        tts.save(path)
        return Audio(path, autoplay=True)
