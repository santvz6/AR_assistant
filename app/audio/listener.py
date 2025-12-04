# app/audio/listener.py

import whisper

class AudioListener:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def process_audio(self, audio_path):
        return self.model.transcribe(audio_path)["text"]
