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
from config import VIDEOS_DIR

def main():
    audio_queue = Queue()
    command_queue = Queue()

    # Inicializamos módulos
    listener = AudioListener(queue=audio_queue)
    tts = TextToSpeech()
    video_file = os.path.join(VIDEOS_DIR, os.listdir(VIDEOS_DIR)[0])
    player = VideoPlayer(video_path=video_file, skip_seconds=0.01)
    game = GameLogic(command_queue=command_queue)

    # Hilo de audio
    Thread(target=listener.run, daemon=True).start()

    # Procesar y generar video anotado
    output_video = player.process_video(output_path=os.path.join(VIDEOS_DIR, "video_annotated.mp4"))
    print(f"Video anotado generado: {output_video}")

    # Loop principal del video/juego
    for frame in player.iter_frames():
        detections = player.detect_objects(frame)
        game.update(detections)

        while not audio_queue.empty():
            comando = audio_queue.get()
            game.process_command(comando)

        if game.has_instruction():
            tts.say(game.get_instruction())

if __name__ == "__main__":
    main()
