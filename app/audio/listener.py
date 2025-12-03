# app/audio/listener.py

import whisper
from queue import Queue

class AudioListener:
    def __init__(self, queue: Queue, model_name="base"):
        self.queue = queue
        self.model = whisper.load_model(model_name)

    def run(self):
        """
        Loop continuo para escuchar audio y transcribir.
        Aquí debes reemplazar `ruta_audio` con tu captura en tiempo real.
        """
        while True:
            ruta_audio = "audio.wav"  # placeholder: captura real con micrófono
            resultado = self.model.transcribe(ruta_audio)
            texto = resultado["text"]
            if texto.strip():
                self.queue.put(texto)
