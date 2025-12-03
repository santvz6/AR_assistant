# app/main.py

# 1. Esperar Orden del Usuario: Speech2Text (Whisper)
    # 1.1 Transcribir audio: obtener orden.
    # 1.2 Analizar la orden (ej: contiene palabras como 'buscar', 'marcar', 'qué hay...').
# 2. Explicación del juego: Text2Speech
# 3. Inicio del juego 
    # 3.1 Video Processing
    # 3.2 Object Detection: YOLO
# 4. Esperar preguntas del usuario
    # 4.1 Procesar respuestas: CLIP
# 5. Cada cierto tiempo el Agente dará pistas
    # 5.1 ¿Comparando la puntuacion de cada palabra de nuestro dict clave?

import os

from threading import Thread
from queue import Queue

from audio.listener import AudioListener
from audio.tts import TextToSpeech
from video.player import VideoPlayer
from game.logic import GameLogic


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
audios_path = os.path.join(BASE_DIR, "resources", "sounds")
videos_path = os.path.join(BASE_DIR, "resources", "videos")


def main():
    audio_queue = Queue()
    command_queue = Queue()

    # Inicialización de modulos
    listener = AudioListener(queue=audio_queue)
    tts = TextToSpeech()
    player = VideoPlayer(video_path="resources/videos/demo.mp4")
    game = GameLogic(command_queue=command_queue)

    # Hilo para audio
    Thread(target=listener.run, daemon=True).start()

    # Loop principal del juego
    for frame in player.iter_frames():
        detections = player.detect_objects(frame)
        game.update(detections)

        # Procesar nuevas instrucciones de audio
        while not audio_queue.empty():
            comando = audio_queue.get()
            game.process_command(comando)

        # Emitimos instrucciones TTS si corresponde
        if game.has_instruction():
            tts.say(game.get_instruction())


if __name__ == "__main__":
    main()