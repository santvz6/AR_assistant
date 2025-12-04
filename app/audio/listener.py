# app/audio/listener.py

import whisper

from queue import Queue

from config import AUDIOS_DIR

class AudioListener:
    def __init__(self, queue: Queue, model_name="base"):
        self.queue = queue
        self.model = whisper.load_model(model_name)

    def run(self):
        while True:
            audio_path = AUDIOS_DIR[0]
            resultado = self.model.transcribe(audio_path)
            texto = resultado["text"]
            if texto.strip():
                self.queue.put(texto)
