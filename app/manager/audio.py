# app/manager/audio.py

import whisper

from models.tts import TextToSpeech
from config import ANSWERS_DIR


class AudioManager:
    def __init__(self, model_name="base", output_dir=ANSWERS_DIR):
        self.model = whisper.load_model(model_name)
        self.tts = TextToSpeech(output_dir)

    def listen(self, audio_path):
        """Utiliza Whisper para transcribir un audio a texto."""
        return self.model.transcribe(audio_path)["text"]

    def speak(self, text: str, filename: str):
        """Utiliza TTS para transcribir un texto a audio."""
        self.tts._speak(text=text, filename=filename)